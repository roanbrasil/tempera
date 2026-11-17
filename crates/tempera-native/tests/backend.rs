#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Contract tests for the native backend (Contracts §7) on synthetic data.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchReader, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::json;
use tempera_core::Event;
use tempera_core::events::{ArtifactKind, LogLevel};
use tempera_native::{Manifest, NativeError, PredictManifest, run_fit, run_predict};

/// Small deterministic PRNG (xorshift) so tests need no extra dependency.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.next_f64().max(1e-12), self.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Binary,
    Multi,
    Regression,
}

/// Columns: `x1` Float64, `x2` Int64, `plan` Utf8, `flag` Boolean, plus
/// `__target__` and the ignored `__group__`. Optionally injects nulls and an
/// unseen category.
struct Data {
    x1: Vec<Option<f64>>,
    x2: Vec<Option<i64>>,
    plan: Vec<Option<String>>,
    flag: Vec<Option<bool>>,
    target_s: Vec<String>,
    target_f: Vec<f64>,
}

const PLANS: [&str; 3] = ["basic", "plus", "pro"];

fn synth(kind: Kind, n: usize, seed: u64) -> Data {
    let mut r = Rng(seed);
    let mut d = Data {
        x1: vec![],
        x2: vec![],
        plan: vec![],
        flag: vec![],
        target_s: vec![],
        target_f: vec![],
    };
    for _ in 0..n {
        let x1 = r.normal() * 2.0;
        let x2 = (r.next_f64() * 10.0).floor() as i64;
        let p = (r.next_f64() * 3.0).floor() as usize % 3;
        let flag = r.next_f64() < 0.5;
        let plan_effect = [-1.0, 0.0, 1.5][p];
        match kind {
            Kind::Binary => {
                let score = 1.5 * x1 + 0.4 * (x2 as f64 - 4.5) + plan_effect + if flag { 1.0 } else { -1.0 };
                d.target_s.push(if score > 0.0 { "yes" } else { "no" }.into());
            }
            Kind::Multi => {
                // Three bands of x1: well-separated classes.
                let c = if x1 < -1.0 {
                    "a"
                } else if x1 < 1.0 {
                    "b"
                } else {
                    "c"
                };
                d.target_s.push(c.into());
            }
            Kind::Regression => {
                let y = 3.0 * x1 - 0.5 * x2 as f64
                    + 2.0 * plan_effect
                    + if flag { 1.0 } else { 0.0 }
                    + 0.3 * r.normal();
                d.target_f.push(y);
            }
        }
        d.x1.push(Some(x1));
        d.x2.push(Some(x2));
        d.plan.push(Some(PLANS[p].into()));
        d.flag.push(Some(flag));
    }
    d
}

fn write_parquet(path: &Path, d: &Data, kind: Kind) {
    let n = d.x1.len();
    let target: ArrayRef = if kind == Kind::Regression {
        Arc::new(Float64Array::from(d.target_f.clone()))
    } else {
        Arc::new(StringArray::from_iter_values(d.target_s.iter()))
    };
    let tdt = if kind == Kind::Regression {
        DataType::Float64
    } else {
        DataType::Utf8
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x1", DataType::Float64, true),
        Field::new("x2", DataType::Int64, true),
        Field::new("plan", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
        Field::new("__target__", tdt, false),
        Field::new("__group__", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(d.x1.clone())),
            Arc::new(Int64Array::from(d.x2.clone())),
            Arc::new(StringArray::from(d.plan.clone())),
            Arc::new(BooleanArray::from(d.flag.clone())),
            target,
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|i| format!("g{}", i % 7)),
            )),
        ],
    )
    .unwrap();
    let mut w = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn read_batch(path: &Path) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let schema = reader.schema();
    let batches: Vec<_> = reader.map(Result::unwrap).collect();
    arrow::compute::concat_batches(&schema, &batches).unwrap()
}

fn classes_of(kind: Kind) -> Option<Vec<&'static str>> {
    match kind {
        Kind::Binary => Some(vec!["no", "yes"]),
        Kind::Multi => Some(vec!["a", "b", "c"]),
        Kind::Regression => None,
    }
}

fn task_of(kind: Kind) -> &'static str {
    match kind {
        Kind::Binary => "binary_classification",
        Kind::Multi => "multiclass_classification",
        Kind::Regression => "regression",
    }
}

struct Setup {
    _dir: tempfile::TempDir,
    root: PathBuf,
    valid: Data,
}

fn setup(kind: Kind) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    write_parquet(&root.join("train.parquet"), &synth(kind, 400, 7), kind);
    let valid = synth(kind, 200, 99);
    write_parquet(&root.join("valid.parquet"), &valid, kind);
    Setup {
        _dir: dir,
        root,
        valid,
    }
}

fn manifest(algorithm: &str, kind: Kind, root: &Path, params: serde_json::Value) -> Manifest {
    let mut m = json!({
        "algorithm": algorithm,
        "task": task_of(kind),
        "features": [
            {"name": "x1", "kind": "numeric"},
            {"name": "x2", "kind": "numeric"},
            {"name": "plan", "kind": "categorical"},
            {"name": "flag", "kind": "boolean"}
        ],
        "params": params,
        "seed": 42,
        "train": root.join("train.parquet"),
        "valid": root.join("valid.parquet"),
        "predict": [{"input": root.join("valid.parquet"), "output": root.join("run/valid_pred.parquet")}],
        "work_dir": root.join("run"),
        "export": ["onnx", "native"]
    });
    if let Some(c) = classes_of(kind) {
        m["classes"] = json!(c);
        if kind == Kind::Binary {
            m["positive_class"] = json!("yes");
        }
    }
    Manifest::from_json(&m.to_string()).unwrap()
}

fn parse_events(buf: &[u8]) -> Vec<Event> {
    let text = std::str::from_utf8(buf).unwrap();
    text.lines()
        .map(|l| Event::parse_line(l).unwrap_or_else(|e| panic!("bad event line {l:?}: {e}")))
        .collect()
}

fn accuracy(batch: &RecordBatch, truth: &[String]) -> f64 {
    let pred = batch
        .column_by_name("prediction")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let ok = (0..pred.len()).filter(|&i| pred.value(i) == truth[i]).count();
    ok as f64 / truth.len() as f64
}

fn r2(batch: &RecordBatch, truth: &[f64]) -> f64 {
    let pred = batch
        .column_by_name("prediction")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let ss_tot: f64 = truth.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = truth
        .iter()
        .enumerate()
        .map(|(i, y)| (y - pred.value(i)).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

/// Full fit → schema/quality → predict-from-saved-model round trip.
fn contract(algorithm: &str, kind: Kind, params: serde_json::Value, min_quality: f64) {
    let s = setup(kind);
    let m = manifest(algorithm, kind, &s.root, params);
    let mut out = Vec::new();
    run_fit(&m, &mut out).unwrap();
    let events = parse_events(&out);

    // Event stream: done last, model + predictions artifacts, warn for onnx, importances.
    assert!(matches!(events.last(), Some(Event::Done { .. })));
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Artifact {
            kind: ArtifactKind::Model,
            ..
        }
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        Event::Artifact {
            kind: ArtifactKind::Predictions,
            ..
        }
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Log { level: LogLevel::Warn, msg } if msg.contains("onnx")))
    );
    for f in ["x1", "x2", "plan", "flag"] {
        let name = format!("importance/{f}");
        assert!(
            events.iter().any(|e| matches!(e, Event::Metric { name: n, step: None, value, .. } if *n == name && value.is_finite())),
            "missing {name}"
        );
    }
    assert!(s.root.join("run/model/model.json").exists());

    // Predictions schema and row count.
    let pred_path = s.root.join("run/valid_pred.parquet");
    let batch = read_batch(&pred_path);
    assert_eq!(batch.num_rows(), 200);
    let schema = batch.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    match classes_of(kind) {
        Some(classes) => {
            let mut expected = vec!["prediction".to_owned()];
            expected.extend(classes.iter().map(|c| format!("proba__{c}")));
            assert_eq!(names, expected);
            assert_eq!(schema.field(0).data_type(), &DataType::Utf8);
            for f in &schema.fields()[1..] {
                assert_eq!(f.data_type(), &DataType::Float64);
            }
            // Probabilities sum to one.
            for i in 0..batch.num_rows() {
                let s: f64 = (1..batch.num_columns())
                    .map(|c| {
                        batch
                            .column(c)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .value(i)
                    })
                    .sum();
                assert!((s - 1.0).abs() < 1e-9, "row {i} proba sum {s}");
            }
            let acc = accuracy(&batch, &s.valid.target_s);
            assert!(acc > min_quality, "{algorithm}: accuracy {acc}");
        }
        None => {
            assert_eq!(names, vec!["prediction"]);
            assert_eq!(schema.field(0).data_type(), &DataType::Float64);
            let r2 = r2(&batch, &s.valid.target_f);
            assert!(r2 > min_quality, "{algorithm}: r2 {r2}");
        }
    }

    // Predict from the saved model reproduces fit-time predictions exactly.
    let pm = PredictManifest {
        algorithm: algorithm.into(),
        task: m.task,
        model_dir: s.root.join("run/model"),
        input: s.root.join("valid.parquet"),
        output: s.root.join("run/again.parquet"),
    };
    let mut out2 = Vec::new();
    run_predict(&pm, &mut out2).unwrap();
    let events2 = parse_events(&out2);
    assert!(matches!(events2.last(), Some(Event::Done { .. })));
    assert_eq!(read_batch(&s.root.join("run/again.parquet")), batch);
}

#[test]
fn linear_regression_contract() {
    contract("linear_regression", Kind::Regression, json!({}), 0.8);
}

#[test]
fn ridge_contract() {
    contract("ridge", Kind::Regression, json!({"alpha": 1.0}), 0.8);
}

#[test]
fn logistic_binary_contract() {
    contract(
        "logistic_regression",
        Kind::Binary,
        json!({"alpha": 0.01, "max_iterations": 200}),
        0.85,
    );
}

#[test]
fn logistic_multiclass_contract() {
    contract(
        "logistic_regression",
        Kind::Multi,
        json!({"alpha": 0.01, "max_iterations": 300}),
        0.85,
    );
}

#[test]
fn decision_tree_binary_contract() {
    contract("decision_tree", Kind::Binary, json!({"max_depth": 8}), 0.8);
}

#[test]
fn decision_tree_multiclass_contract() {
    contract("decision_tree", Kind::Multi, json!({}), 0.9);
}

#[test]
fn unknown_algorithm_is_rejected() {
    let s = setup(Kind::Binary);
    let m = manifest("xgboost_turbo", Kind::Binary, &s.root, json!({}));
    let err = run_fit(&m, &mut Vec::new()).unwrap_err();
    assert!(matches!(err, NativeError::UnknownAlgorithm(_)));
    assert_eq!(err.exit_code(), 2);
    assert!(matches!(err.to_event(), Event::Error { .. }));
}

#[test]
fn decision_tree_regression_is_unsupported() {
    let s = setup(Kind::Regression);
    let m = manifest("decision_tree", Kind::Regression, &s.root, json!({}));
    let err = run_fit(&m, &mut Vec::new()).unwrap_err();
    assert!(matches!(err, NativeError::UnsupportedTask { .. }));
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn bad_manifest_json_is_exit_code_2() {
    let err = Manifest::from_json(r#"{"algorithm": "ridge"}"#).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn predict_handles_unseen_category_and_nulls() {
    for (algorithm, kind) in [
        ("logistic_regression", Kind::Binary),
        ("decision_tree", Kind::Multi),
        ("ridge", Kind::Regression),
    ] {
        let s = setup(kind);
        let m = manifest(algorithm, kind, &s.root, json!({}));
        run_fit(&m, &mut Vec::new()).unwrap();

        let mut d = synth(kind, 6, 5);
        d.plan[0] = Some("enterprise".into()); // never seen in train
        d.plan[1] = None;
        d.x1[2] = None;
        d.x2[3] = None;
        d.flag[4] = None;
        d.x1[5] = Some(f64::NAN);
        let input = s.root.join("odd.parquet");
        write_parquet(&input, &d, kind);
        let pm = PredictManifest {
            algorithm: algorithm.into(),
            task: m.task,
            model_dir: s.root.join("run/model"),
            input,
            output: s.root.join("run/odd_pred.parquet"),
        };
        let mut out = Vec::new();
        run_predict(&pm, &mut out).unwrap();
        parse_events(&out);
        let batch = read_batch(&s.root.join("run/odd_pred.parquet"));
        assert_eq!(batch.num_rows(), 6, "{algorithm}");
        for c in 0..batch.num_columns() {
            let col = batch.column(c);
            assert_eq!(col.null_count(), 0);
            if let Some(f) = col.as_any().downcast_ref::<Float64Array>() {
                assert!(
                    f.iter().all(|v| v.unwrap().is_finite()),
                    "{algorithm}: non-finite output"
                );
            }
        }
    }
}

#[test]
fn predict_rejects_mismatched_algorithm() {
    let s = setup(Kind::Regression);
    let m = manifest("linear_regression", Kind::Regression, &s.root, json!({}));
    run_fit(&m, &mut Vec::new()).unwrap();
    let pm = PredictManifest {
        algorithm: "ridge".into(),
        task: m.task,
        model_dir: s.root.join("run/model"),
        input: s.root.join("valid.parquet"),
        output: s.root.join("run/x.parquet"),
    };
    let err = run_predict(&pm, &mut Vec::new()).unwrap_err();
    assert_eq!(err.exit_code(), 2);
}

#[test]
fn manifest_round_trips_documented_example() {
    let text = r#"{"algorithm":"logistic_regression","task":"binary_classification","classes":["no","yes"],"positive_class":"yes","features":[{"name":"tenure","kind":"numeric"},{"name":"plan","kind":"categorical"},{"name":"has_discount","kind":"boolean"}],"params":{},"seed":42,"train":"/abs/train.parquet","valid":"/abs/valid.parquet","predict":[{"input":"/abs/valid.parquet","output":"/abs/run/valid_pred.parquet"}],"work_dir":"/abs/run","export":["onnx","native"]}"#;
    let m = Manifest::from_json(text).unwrap();
    let again = Manifest::from_json(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(m, again);
    let p = PredictManifest::from_json(
        r#"{"algorithm":"ridge","task":"regression","model_dir":"/abs/run/model","input":"/a.parquet","output":"/b.parquet"}"#,
    )
    .unwrap();
    assert_eq!(p.algorithm, "ridge");
}

#[test]
fn categorical_keeps_top_50_plus_other() {
    let s = setup(Kind::Regression);
    // Overwrite train with 60 distinct plans; plan_i appears i+1 times.
    let mut d = synth(Kind::Regression, 0, 1);
    let mut r = Rng(3);
    for i in 0..60usize {
        for _ in 0..=i {
            d.x1.push(Some(r.normal()));
            d.x2.push(Some(1));
            d.plan.push(Some(format!("p{i:02}")));
            d.flag.push(Some(true));
            d.target_f.push(i as f64);
        }
    }
    write_parquet(&s.root.join("train.parquet"), &d, Kind::Regression);
    let m = manifest("ridge", Kind::Regression, &s.root, json!({}));
    run_fit(&m, &mut Vec::new()).unwrap();
    let (_, native) = tempera_native::load_model(&s.root.join("run/model")).unwrap();
    let plan = native
        .preprocessor
        .columns
        .iter()
        .find(|c| c.name() == "plan")
        .unwrap();
    match plan {
        tempera_native::data::ColumnTransform::Categorical {
            categories, other, ..
        } => {
            assert_eq!(categories.len(), 50);
            assert!(*other);
            // Most frequent first; the 10 rarest (p00..p09) fall into `other`.
            assert_eq!(categories[0], "p59");
            assert!(!categories.contains(&"p09".to_owned()));
        }
        _ => panic!("plan should be categorical"),
    }
    assert_eq!(native.preprocessor.width(), 1 + 1 + 51 + 1);
}

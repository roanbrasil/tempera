//! Native Rust algorithms (linfa) for jobs that do not need Python.
//!
//! Implements the backend run contract (Contracts §7): a JSON [`Manifest`]
//! comes in, JSONL [`Event`]s go out, predictions are written as Parquet and
//! the model is saved under `<work_dir>/model/`.
//!
//! | Algorithm | Tasks | linfa crate |
//! |---|---|---|
//! | `linear_regression` | regression | `linfa-linear` (least squares) |
//! | `ridge` | regression | `linfa-elasticnet` with `l1_ratio = 0` |
//! | `logistic_regression` | binary / multiclass | `linfa-logistic` (L2, L-BFGS) |
//! | `decision_tree` | binary / multiclass | `linfa-trees` (CART) |
//!
//! `decision_tree` has no regression variant: linfa-trees 0.8 only ships a
//! classification tree. All four algorithms are deterministic, so `seed` is
//! accepted but has no effect.
//!
//! Saved model: `model.json` (a [`ModelDescriptor`]) plus `native.json`
//! (fitted [`Preprocessor`] and [`FittedModel`] as plain parameters).

pub mod data;
pub mod error;
pub mod manifest;
pub mod model;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tempera_core::Event;
use tempera_core::events::{ArtifactKind, LogLevel};
use tempera_core::spec::{ExportFormat, TaskType};

pub use data::Preprocessor;
pub use error::NativeError;
pub use manifest::{FeatureKind, FeatureSpec, Manifest, PredictManifest, PredictTarget};
pub use model::{Algorithm, FittedModel};

use data::{Predictions, RESERVED_COLUMNS, Table};
use manifest::is_classification;
use model::Target;

/// Algorithm names accepted in manifests.
pub const SUPPORTED_ALGORITHMS: [&str; 4] = [
    "linear_regression",
    "ridge",
    "logistic_regression",
    "decision_tree",
];

/// Version of the saved-model layout.
pub const MODEL_FORMAT_VERSION: u32 = 1;
/// Descriptor file inside the model directory.
pub const MODEL_DESCRIPTOR: &str = "model.json";
/// Native model file inside the model directory.
pub const NATIVE_MODEL_FILE: &str = "native.json";

/// `<model_dir>/model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub format_version: u32,
    pub backend: String,
    pub algorithm: String,
    pub task: TaskType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive_class: Option<String>,
    pub features: Vec<FeatureSpec>,
    /// Files of the model, relative to the model directory.
    pub files: BTreeMap<String, String>,
}

/// `<model_dir>/native.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeModel {
    pub preprocessor: Preprocessor,
    pub model: FittedModel,
}

/// What [`run_fit`] produced.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReport {
    pub model_dir: PathBuf,
    pub summary: BTreeMap<String, f64>,
}

fn emit(events: &mut impl Write, ev: &Event) -> Result<(), NativeError> {
    writeln!(events, "{}", ev.to_line())?;
    events.flush()?;
    Ok(())
}

fn log(events: &mut impl Write, level: LogLevel, msg: impl Into<String>) -> Result<(), NativeError> {
    emit(
        events,
        &Event::Log {
            level,
            msg: msg.into(),
        },
    )
}

fn metric(events: &mut impl Write, name: String, value: f64) -> Result<(), NativeError> {
    emit(
        events,
        &Event::Metric {
            name,
            value,
            step: None,
            epoch: None,
            ts: None,
        },
    )
}

fn validate_manifest(m: &Manifest) -> Result<(Algorithm, Option<Vec<String>>), NativeError> {
    let algorithm = Algorithm::parse(&m.algorithm)?;
    algorithm.check(m.task)?;
    let classes = if is_classification(m.task) {
        Some(m.checked_classes()?)
    } else {
        None
    };
    if m.features.is_empty() {
        return Err(NativeError::Manifest("`features` is empty".into()));
    }
    let mut names = std::collections::HashSet::new();
    for f in &m.features {
        if RESERVED_COLUMNS.contains(&f.name.as_str()) {
            return Err(NativeError::Manifest(format!(
                "`{}` is a reserved column, not a feature",
                f.name
            )));
        }
        if !names.insert(f.name.as_str()) {
            return Err(NativeError::Manifest(format!("duplicate feature `{}`", f.name)));
        }
    }
    Ok((algorithm, classes))
}

/// Fits the model, saves it, writes predictions for every `predict` entry and
/// emits events (ending with `done`) to `events`.
pub fn run_fit(manifest: &Manifest, events: &mut impl Write) -> Result<FitReport, NativeError> {
    let (algorithm, classes) = validate_manifest(manifest)?;
    log(
        events,
        LogLevel::Info,
        format!(
            "native {}: reading {}",
            algorithm.name(),
            manifest.train.display()
        ),
    )?;

    let train = Table::read(&manifest.train)?;
    if train.num_rows() == 0 {
        return Err(NativeError::Data("train set is empty".into()));
    }
    let pre = Preprocessor::fit(&train, &manifest.features)?;
    let x = pre.transform(&train)?;
    log(
        events,
        LogLevel::Info,
        format!("design matrix: {} rows x {} columns", x.nrows(), x.ncols()),
    )?;
    if manifest.valid.is_some() {
        log(
            events,
            LogLevel::Debug,
            "validation set not used (no early stopping in native algorithms)",
        )?;
    }

    let (outcome, loss) = match &classes {
        Some(classes) => {
            let y = train.class_target(classes)?;
            let out = model::fit_classification(algorithm, &x, &y, classes.len(), &manifest.params)?;
            let loss = model::train_loss(&out.model.predict(&x)?, &Target::Classes(&y));
            (out, ("train/log_loss", loss))
        }
        None => {
            let y = train.numeric_target()?;
            let out = model::fit_regression(algorithm, &x, &y, &manifest.params)?;
            let loss = model::train_loss(&out.model.predict(&x)?, &Target::Regression(&y));
            (out, ("train/mse", loss))
        }
    };
    let mut summary = BTreeMap::new();
    if loss.1.is_finite() {
        metric(events, loss.0.to_owned(), loss.1)?;
        summary.insert(loss.0.to_owned(), loss.1);
    }

    // Importance aggregated per original feature.
    let mut importance = vec![0.0; pre.columns.len()];
    for (col, owner) in pre.column_owner().into_iter().enumerate() {
        importance[owner] += outcome.column_importance.get(col).copied().unwrap_or(0.0);
    }
    for (f, v) in manifest.features.iter().zip(importance) {
        metric(events, format!("importance/{}", f.name), v)?;
    }

    // Save the model.
    let model_dir = manifest.work_dir.join("model");
    std::fs::create_dir_all(&model_dir)?;
    let native = NativeModel {
        preprocessor: pre,
        model: outcome.model,
    };
    let native_path = model_dir.join(NATIVE_MODEL_FILE);
    std::fs::write(&native_path, serde_json::to_vec(&native)?)?;
    let descriptor = ModelDescriptor {
        format_version: MODEL_FORMAT_VERSION,
        backend: "tempera-native".into(),
        algorithm: algorithm.name().into(),
        task: manifest.task,
        classes: classes.clone(),
        positive_class: manifest.positive_class.clone(),
        features: manifest.features.clone(),
        files: BTreeMap::from([("native".to_owned(), NATIVE_MODEL_FILE.to_owned())]),
    };
    std::fs::write(
        model_dir.join(MODEL_DESCRIPTOR),
        serde_json::to_vec_pretty(&descriptor)?,
    )?;
    emit(
        events,
        &Event::Artifact {
            kind: ArtifactKind::Model,
            path: native_path,
            format: Some(ExportFormat::Native),
        },
    )?;
    for fmt in &manifest.export {
        if *fmt != ExportFormat::Native {
            log(
                events,
                LogLevel::Warn,
                format!(
                    "export format {} is not supported by the native backend; skipped",
                    serde_json::to_string(fmt)?
                ),
            )?;
        }
    }

    let total = manifest.predict.len() as u64;
    for (i, target) in manifest.predict.iter().enumerate() {
        predict_file(&native, classes.as_deref(), &target.input, &target.output)?;
        emit(
            events,
            &Event::Artifact {
                kind: ArtifactKind::Predictions,
                path: target.output.clone(),
                format: None,
            },
        )?;
        emit(
            events,
            &Event::Progress {
                step: i as u64 + 1,
                total_steps: Some(total),
            },
        )?;
    }

    emit(
        events,
        &Event::Done {
            summary: summary.clone(),
        },
    )?;
    Ok(FitReport { model_dir, summary })
}

fn predict_file(
    native: &NativeModel,
    classes: Option<&[String]>,
    input: &Path,
    output: &Path,
) -> Result<(), NativeError> {
    let table = Table::read(input)?;
    let x = native.preprocessor.transform(&table)?;
    let preds = native.model.predict(&x)?;
    if let (Predictions::Classification { .. }, None) | (Predictions::Regression(_), Some(_)) =
        (&preds, classes)
    {
        return Err(NativeError::Model("model kind does not match the task".into()));
    }
    data::write_predictions(output, &preds, classes)
}

/// Loads `<model_dir>/model.json` and its native model file.
pub fn load_model(model_dir: &Path) -> Result<(ModelDescriptor, NativeModel), NativeError> {
    let read = |p: PathBuf| {
        std::fs::read(&p).map_err(|e| NativeError::Model(format!("cannot read `{}`: {e}", p.display())))
    };
    let descriptor: ModelDescriptor = serde_json::from_slice(&read(model_dir.join(MODEL_DESCRIPTOR))?)?;
    if descriptor.format_version != MODEL_FORMAT_VERSION || descriptor.backend != "tempera-native" {
        return Err(NativeError::Model(format!(
            "unsupported model (backend `{}`, format {})",
            descriptor.backend, descriptor.format_version
        )));
    }
    let file = descriptor
        .files
        .get("native")
        .ok_or_else(|| NativeError::Model("descriptor lists no native file".into()))?;
    if Path::new(file).components().count() != 1 {
        return Err(NativeError::Model(
            "native file must be inside the model directory".into(),
        ));
    }
    let native: NativeModel = serde_json::from_slice(&read(model_dir.join(file))?)?;
    let n_classes = descriptor.classes.as_ref().map_or(0, Vec::len);
    native.model.validate(n_classes, native.preprocessor.width())?;
    Ok((descriptor, native))
}

/// Loads a saved model, predicts `input` into `output` and emits events
/// (ending with `done`).
pub fn run_predict(manifest: &PredictManifest, events: &mut impl Write) -> Result<(), NativeError> {
    let algorithm = Algorithm::parse(&manifest.algorithm)?;
    algorithm.check(manifest.task)?;
    let (descriptor, native) = load_model(&manifest.model_dir)?;
    if descriptor.algorithm != algorithm.name() || descriptor.task != manifest.task {
        return Err(NativeError::Manifest(format!(
            "manifest asks for {} / {:?} but the model is {} / {:?}",
            algorithm.name(),
            manifest.task,
            descriptor.algorithm,
            descriptor.task
        )));
    }
    log(
        events,
        LogLevel::Info,
        format!(
            "native {}: predicting {}",
            algorithm.name(),
            manifest.input.display()
        ),
    )?;
    predict_file(
        &native,
        descriptor.classes.as_deref(),
        &manifest.input,
        &manifest.output,
    )?;
    emit(
        events,
        &Event::Artifact {
            kind: ArtifactKind::Predictions,
            path: manifest.output.clone(),
            format: None,
        },
    )?;
    emit(
        events,
        &Event::Done {
            summary: BTreeMap::new(),
        },
    )
}

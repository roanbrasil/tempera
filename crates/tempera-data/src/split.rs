//! Model-ready data and train/valid/test splits.
//!
//! Order of operations (docs/wiki/Data-Audit.md):
//! 1. drop rows without a target, 2. deduplicate exact duplicates,
//! 3. pick the split strategy (temporal > group > stratified > random),
//! 4. hold out the **test** set first, then carve validation from the rest,
//! 5. write `train/valid/test.parquet` + `split.json`, addressed by a hash of
//!    everything that influences the result (so re-running is free).
//!
//! Backends only ever see the model-ready layout (Contracts.md §7).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempera_core::spec::{JobSpec, SplitStrategy, TaskType};

use crate::error::{DataError, Result};
use crate::ingest::{Dataset, scan_parquet};

pub const TARGET: &str = "__target__";
pub const GROUP: &str = "__group__";
pub const TIME: &str = "__time__";
/// Bump when the split layout or algorithm changes.
pub const SPLIT_VERSION: u32 = 1;

/// The three feature kinds backends understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    Numeric,
    Categorical,
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Feature {
    pub name: String,
    pub kind: FeatureKind,
}

/// How each source column is used. Produced by the audit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FeaturePlan {
    /// Source columns used as-is (after casting).
    pub features: Vec<Feature>,
    /// Datetime source columns expanded into year/month/dayofweek/hour.
    pub datetime: Vec<String>,
    /// Columns not used, with the reason.
    pub dropped: BTreeMap<String, String>,
    /// Remove exact duplicate rows before splitting.
    pub dedupe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    Train,
    Valid,
    Test,
}

/// Result of [`make_split`]; also stored as `split.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Split {
    pub id: String,
    pub version: u32,
    pub dataset_id: String,
    /// Strategy actually used (never `auto`).
    pub strategy: SplitStrategy,
    pub seed: u64,
    pub task: TaskType,
    /// Final feature columns as the backends see them.
    pub features: Vec<Feature>,
    /// Sorted labels (classification only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive_class: Option<String>,
    pub rows: BTreeMap<String, u64>,
    /// Label counts per partition (classification only).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub class_counts: BTreeMap<String, BTreeMap<String, u64>>,
    pub dropped_null_target: u64,
    pub dropped_duplicates: u64,
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Split {
    pub fn path(&self, p: Partition) -> PathBuf {
        self.dir.join(match p {
            Partition::Train => "train.parquet",
            Partition::Valid => "valid.parquet",
            Partition::Test => "test.parquet",
        })
    }
}

/// Resolves `auto` to a concrete strategy.
pub fn resolve_strategy(spec: &JobSpec, task: TaskType) -> SplitStrategy {
    match spec.split.strategy {
        SplitStrategy::Auto if spec.data.time_column.is_some() => SplitStrategy::Temporal,
        SplitStrategy::Auto if spec.data.group_column.is_some() => SplitStrategy::Group,
        SplitStrategy::Auto
            if matches!(
                task,
                TaskType::BinaryClassification | TaskType::MulticlassClassification
            ) =>
        {
            SplitStrategy::Stratified
        }
        SplitStrategy::Auto => SplitStrategy::Random,
        s => s,
    }
}

/// Builds the model-ready splits for `ds` (cached by content).
pub fn make_split(ds: &Dataset, spec: &JobSpec, task: TaskType, plan: &FeaturePlan) -> Result<Split> {
    let target = spec
        .data
        .target
        .clone()
        .ok_or_else(|| DataError::Invalid("a target column is required to split".into()))?;
    let strategy = resolve_strategy(spec, task);
    let id = split_id(ds, spec, task, plan, strategy)?;
    let dir = ds.dir.join("splits").join(&id);
    if let Ok(text) = std::fs::read_to_string(dir.join("split.json"))
        && let Ok(mut s) = serde_json::from_str::<Split>(&text)
    {
        s.dir = dir;
        return Ok(s);
    }

    let classification = matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    );
    let mut df = model_ready(
        scan_parquet(&ds.data_path())?,
        spec,
        &target,
        plan,
        classification,
    )?
    .collect()?;

    let before = df.height();
    df = df.filter(&df.column(TARGET)?.is_not_null())?;
    let dropped_null_target = (before - df.height()) as u64;

    let before = df.height();
    if plan.dedupe {
        df = df.unique_stable(None, UniqueKeepStrategy::First, None)?;
    }
    let dropped_duplicates = (before - df.height()) as u64;
    if df.height() < 10 {
        return Err(DataError::Invalid(format!(
            "only {} usable rows; need at least 10",
            df.height()
        )));
    }

    let assignment = assign(&df, strategy, spec.split.test_size, spec.split.seed)?;
    std::fs::create_dir_all(&dir)?;
    let tmp = tempdir_in(&dir)?;
    let mut rows = BTreeMap::new();
    let mut class_counts = BTreeMap::new();
    for part in [Partition::Train, Partition::Valid, Partition::Test] {
        let mask: BooleanChunked = assignment.iter().map(|&p| p == part).collect();
        let mut part_df = df.filter(&mask)?;
        let name = format!("{part:?}").to_lowercase();
        if part_df.height() == 0 {
            return Err(DataError::Invalid(format!(
                "the {name} partition is empty; add data or change the split"
            )));
        }
        rows.insert(name.clone(), part_df.height() as u64);
        if classification {
            class_counts.insert(name.clone(), label_counts(part_df.column(TARGET)?)?);
        }
        let mut f = std::fs::File::create(tmp.join(format!("{name}.parquet")))?;
        ParquetWriter::new(&mut f).finish(&mut part_df)?;
    }

    let classes = if classification {
        sorted_labels(df.column(TARGET)?)?
    } else {
        Vec::new()
    };
    let positive_class = (task == TaskType::BinaryClassification)
        .then(|| positive_class(&classes, df.column(TARGET)))
        .flatten();
    let features = final_features(&df);
    let split = Split {
        id,
        version: SPLIT_VERSION,
        dataset_id: ds.id.clone(),
        strategy,
        seed: spec.split.seed,
        task,
        features,
        classes,
        positive_class,
        rows,
        class_counts,
        dropped_null_target,
        dropped_duplicates,
        dir: dir.clone(),
    };
    std::fs::write(tmp.join("split.json"), serde_json::to_string_pretty(&split)?)?;
    for f in ["train.parquet", "valid.parquet", "test.parquet", "split.json"] {
        std::fs::rename(tmp.join(f), dir.join(f))?;
    }
    let _ = std::fs::remove_dir(&tmp);
    Ok(split)
}

/// Selects, casts and renames columns into the model-ready layout.
fn model_ready(
    lf: LazyFrame,
    spec: &JobSpec,
    target: &str,
    plan: &FeaturePlan,
    classification: bool,
) -> Result<LazyFrame> {
    let schema = lf.clone().collect_schema()?;
    let has = |c: &str| schema.get(c).is_some();
    if !has(target) {
        return Err(DataError::MissingColumn(target.to_owned()));
    }
    let mut exprs = Vec::new();
    for f in &plan.features {
        if !has(&f.name) {
            return Err(DataError::MissingColumn(f.name.clone()));
        }
        let e = col(f.name.as_str());
        exprs.push(match f.kind {
            FeatureKind::Numeric => e.cast(DataType::Float64),
            FeatureKind::Categorical => e.cast(DataType::String),
            FeatureKind::Boolean => e.cast(DataType::Boolean),
        });
    }
    for d in &plan.datetime {
        if !has(d) {
            return Err(DataError::MissingColumn(d.clone()));
        }
        let c = col(d.as_str());
        exprs.push(
            c.clone()
                .dt()
                .year()
                .cast(DataType::Float64)
                .alias(format!("{d}__year")),
        );
        exprs.push(
            c.clone()
                .dt()
                .month()
                .cast(DataType::Float64)
                .alias(format!("{d}__month")),
        );
        exprs.push(
            c.clone()
                .dt()
                .weekday()
                .cast(DataType::Float64)
                .alias(format!("{d}__dayofweek")),
        );
        if matches!(schema.get(d), Some(DataType::Datetime(..))) {
            exprs.push(c.dt().hour().cast(DataType::Float64).alias(format!("{d}__hour")));
        }
    }
    let t = col(target);
    exprs.push(
        if classification {
            label_expr(t, schema.get(target))
        } else {
            t.cast(DataType::Float64)
        }
        .alias(TARGET),
    );
    if let Some(g) = &spec.data.group_column {
        if !has(g) {
            return Err(DataError::MissingColumn(g.clone()));
        }
        exprs.push(col(g.as_str()).cast(DataType::String).alias(GROUP));
    }
    if let Some(tc) = &spec.data.time_column {
        if !has(tc) {
            return Err(DataError::MissingColumn(tc.clone()));
        }
        exprs.push(col(tc.as_str()).alias(TIME));
    }
    Ok(lf.select(exprs))
}

/// Labels as strings; floats that are whole numbers lose the `.0`.
fn label_expr(e: Expr, dtype: Option<&DataType>) -> Expr {
    match dtype {
        Some(dt) if dt.is_float() => e.cast(DataType::Int64).cast(DataType::String),
        _ => e.cast(DataType::String),
    }
}

fn final_features(df: &DataFrame) -> Vec<Feature> {
    df.columns()
        .iter()
        .filter(|c| !c.name().starts_with("__"))
        .map(|c| Feature {
            name: c.name().to_string(),
            kind: match c.dtype() {
                DataType::Boolean => FeatureKind::Boolean,
                DataType::String => FeatureKind::Categorical,
                _ => FeatureKind::Numeric,
            },
        })
        .collect()
}

fn assign(df: &DataFrame, strategy: SplitStrategy, test_size: f64, seed: u64) -> Result<Vec<Partition>> {
    let n = df.height();
    // Validation is carved from what remains after the test holdout.
    let valid_size = test_size;
    Ok(match strategy {
        SplitStrategy::Temporal => {
            let t = df
                .column(TIME)?
                .as_materialized_series()
                .arg_sort(SortOptions::default());
            let order: Vec<usize> = t.iter().flatten().map(|i| i as usize).collect();
            let n_test = ((n as f64) * test_size).round().max(1.0) as usize;
            let n_valid = (((n - n_test) as f64) * valid_size).round().max(1.0) as usize;
            let mut out = vec![Partition::Train; n];
            for (rank, &row) in order.iter().enumerate() {
                out[row] = if rank >= n - n_test {
                    Partition::Test
                } else if rank >= n - n_test - n_valid {
                    Partition::Valid
                } else {
                    Partition::Train
                };
            }
            out
        }
        SplitStrategy::Group => {
            let groups = df.column(GROUP)?.str()?.clone();
            let mut sizes: HashMap<String, usize> = HashMap::new();
            for g in groups.iter() {
                *sizes.entry(g.unwrap_or("<null>").to_owned()).or_default() += 1;
            }
            let mut keys: Vec<&String> = sizes.keys().collect();
            keys.sort_by_key(|k| (mix(seed, k.as_bytes()), *k));
            let n_test = (n as f64) * test_size;
            let n_valid = (n as f64 - n_test) * valid_size;
            let (mut acc, mut part_of) = (0usize, HashMap::new());
            for k in keys {
                let p = if (acc as f64) < n_test {
                    Partition::Test
                } else if (acc as f64) < n_test + n_valid {
                    Partition::Valid
                } else {
                    Partition::Train
                };
                acc += sizes[k];
                part_of.insert(k.clone(), p);
            }
            groups.iter().map(|g| part_of[g.unwrap_or("<null>")]).collect()
        }
        SplitStrategy::Stratified => {
            let labels = df.column(TARGET)?.str()?.clone();
            let mut by_class: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (i, l) in labels.iter().enumerate() {
                by_class.entry(l.unwrap_or("").to_owned()).or_default().push(i);
            }
            let mut out = vec![Partition::Train; n];
            for (label, mut idx) in by_class {
                shuffle(&mut idx, seed ^ mix(0, label.as_bytes()));
                cut(&idx, test_size, valid_size, &mut out);
            }
            out
        }
        SplitStrategy::Random | SplitStrategy::Auto => {
            let mut idx: Vec<usize> = (0..n).collect();
            shuffle(&mut idx, seed);
            let mut out = vec![Partition::Train; n];
            cut(&idx, test_size, valid_size, &mut out);
            out
        }
    })
}

fn cut(idx: &[usize], test_size: f64, valid_size: f64, out: &mut [Partition]) {
    let k = idx.len();
    let n_test = ((k as f64) * test_size).round() as usize;
    let n_valid = (((k - n_test) as f64) * valid_size).round() as usize;
    for (pos, &row) in idx.iter().enumerate() {
        out[row] = if pos < n_test {
            Partition::Test
        } else if pos < n_test + n_valid {
            Partition::Valid
        } else {
            Partition::Train
        };
    }
}

/// Deterministic Fisher-Yates with SplitMix64 (no RNG dependency, stable across versions).
fn shuffle(v: &mut [usize], seed: u64) {
    let mut state = seed;
    for i in (1..v.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        v.swap(i, (z % (i as u64 + 1)) as usize);
    }
}

fn mix(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(seed.to_le_bytes());
    h.update(bytes);
    let d = h.finalize();
    u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
}

fn sorted_labels(c: &Column) -> Result<Vec<String>> {
    let mut v: Vec<String> = c
        .str()?
        .iter()
        .flatten()
        .map(str::to_owned)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    v.sort();
    Ok(v)
}

fn label_counts(c: &Column) -> Result<BTreeMap<String, u64>> {
    let mut m = BTreeMap::new();
    for l in c.str()?.iter().flatten() {
        *m.entry(l.to_owned()).or_insert(0) += 1;
    }
    Ok(m)
}

/// Picks the positive class: conventional names first, otherwise the minority.
fn positive_class(classes: &[String], target: PolarsResult<&Column>) -> Option<String> {
    const POSITIVE: [&str; 6] = ["1", "true", "yes", "y", "positive", "pos"];
    for p in POSITIVE {
        if let Some(c) = classes.iter().find(|c| c.eq_ignore_ascii_case(p)) {
            return Some(c.clone());
        }
    }
    let counts = label_counts(target.ok()?).ok()?;
    counts.into_iter().min_by_key(|(_, n)| *n).map(|(l, _)| l)
}

fn split_id(
    ds: &Dataset,
    spec: &JobSpec,
    task: TaskType,
    plan: &FeaturePlan,
    strategy: SplitStrategy,
) -> Result<String> {
    let key = serde_json::json!({
        "v": SPLIT_VERSION,
        "dataset": ds.id,
        "target": spec.data.target,
        "group": spec.data.group_column,
        "time": spec.data.time_column,
        "task": task,
        "strategy": strategy,
        "test_size": spec.split.test_size,
        "seed": spec.split.seed,
        "plan": plan,
    });
    let digest = Sha256::digest(serde_json::to_vec(&key)?);
    Ok(hex::encode(&digest[..8]))
}

fn tempdir_in(dir: &Path) -> Result<PathBuf> {
    let p = dir.join(format!(".tmp-{}", std::process::id()));
    std::fs::create_dir_all(&p)?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::ingest::ingest;
    use std::collections::HashSet;

    fn dataset(n: usize) -> (tempfile::TempDir, Dataset) {
        let tmp = tempfile::tempdir().unwrap();
        let mut csv = String::from("customer,day,amount,plan,label\n");
        for i in 0..n {
            let label = if i % 10 == 0 { "yes" } else { "no" };
            csv += &format!(
                "c{},2024-01-{:02},{}.5,{},{}\n",
                i % 37,
                1 + i % 28,
                i,
                ["a", "b"][i % 2],
                label
            );
        }
        // one exact duplicate and one null target
        csv += "c1,2024-01-02,9999.5,b,no\nc1,2024-01-02,9999.5,b,no\nc2,2024-01-03,7.5,a,\n";
        let p = tmp.path().join("d.csv");
        std::fs::write(&p, csv).unwrap();
        let ds = ingest(&p, tmp.path()).unwrap();
        (tmp, ds)
    }

    fn plan() -> FeaturePlan {
        FeaturePlan {
            features: vec![
                Feature {
                    name: "amount".into(),
                    kind: FeatureKind::Numeric,
                },
                Feature {
                    name: "plan".into(),
                    kind: FeatureKind::Categorical,
                },
            ],
            datetime: vec!["day".into()],
            dropped: BTreeMap::new(),
            dedupe: true,
        }
    }

    fn spec(extra: &str) -> JobSpec {
        JobSpec::from_yaml(&format!(
            "version: 1\nname: t\ndata: {{ source: d.csv, target: label{extra} }}\n"
        ))
        .unwrap()
    }

    fn read(p: &Path) -> DataFrame {
        scan_parquet(p).unwrap().collect().unwrap()
    }

    #[test]
    fn stratified_split_preserves_class_balance() {
        let (_t, ds) = dataset(1000);
        let s = make_split(&ds, &spec(""), TaskType::BinaryClassification, &plan()).unwrap();
        assert_eq!(s.strategy, SplitStrategy::Stratified);
        assert_eq!(s.dropped_null_target, 1);
        assert_eq!(s.dropped_duplicates, 1);
        assert_eq!(s.classes, ["no", "yes"]);
        assert_eq!(s.positive_class.as_deref(), Some("yes"));
        assert_eq!(s.rows.values().sum::<u64>(), 1001);
        for part in ["train", "valid", "test"] {
            let c = &s.class_counts[part];
            let share = c["yes"] as f64 / (c["yes"] + c["no"]) as f64;
            assert!((share - 0.1).abs() < 0.02, "{part}: {share}");
        }
        let names: Vec<_> = s.features.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            ["amount", "plan", "day__year", "day__month", "day__dayofweek"]
        );
        let train = read(&s.path(Partition::Train));
        assert!(train.column(TARGET).unwrap().dtype().is_string());
    }

    #[test]
    fn group_split_never_shares_entities() {
        let (_t, ds) = dataset(1000);
        let s = make_split(
            &ds,
            &spec(", group_column: customer"),
            TaskType::BinaryClassification,
            &plan(),
        )
        .unwrap();
        assert_eq!(s.strategy, SplitStrategy::Group);
        let groups = |p| -> HashSet<String> {
            read(&s.path(p))
                .column(GROUP)
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .flatten()
                .map(str::to_owned)
                .collect()
        };
        let (tr, va, te) = (
            groups(Partition::Train),
            groups(Partition::Valid),
            groups(Partition::Test),
        );
        assert!(tr.is_disjoint(&te) && tr.is_disjoint(&va) && va.is_disjoint(&te));
    }

    #[test]
    fn temporal_split_respects_order() {
        let (_t, ds) = dataset(1000);
        let s = make_split(
            &ds,
            &spec(", time_column: day"),
            TaskType::BinaryClassification,
            &plan(),
        )
        .unwrap();
        assert_eq!(s.strategy, SplitStrategy::Temporal);
        let times = |p| -> Vec<i32> {
            read(&s.path(p))
                .column(TIME)
                .unwrap()
                .cast(&DataType::Int32)
                .unwrap()
                .i32()
                .unwrap()
                .iter()
                .flatten()
                .collect()
        };
        let max_train = times(Partition::Train).into_iter().max().unwrap();
        let min_valid = times(Partition::Valid).into_iter().min().unwrap();
        let max_valid = times(Partition::Valid).into_iter().max().unwrap();
        let min_test = times(Partition::Test).into_iter().min().unwrap();
        assert!(max_train <= min_valid && max_valid <= min_test);
    }

    #[test]
    fn same_seed_same_split_and_cached() {
        let (_t, ds) = dataset(500);
        let a = make_split(&ds, &spec(""), TaskType::BinaryClassification, &plan()).unwrap();
        let mtime = std::fs::metadata(a.path(Partition::Test))
            .unwrap()
            .modified()
            .unwrap();
        let b = make_split(&ds, &spec(""), TaskType::BinaryClassification, &plan()).unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(
            std::fs::metadata(b.path(Partition::Test))
                .unwrap()
                .modified()
                .unwrap(),
            mtime
        );
        let mut other = spec("");
        other.split.seed = 7;
        let c = make_split(&ds, &other, TaskType::BinaryClassification, &plan()).unwrap();
        assert_ne!(a.id, c.id);
        assert_ne!(read(&a.path(Partition::Test)), read(&c.path(Partition::Test)));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        for seed in 0..20 {
            let mut v: Vec<usize> = (0..100).collect();
            shuffle(&mut v, seed);
            let mut s = v.clone();
            s.sort_unstable();
            assert_eq!(s, (0..100).collect::<Vec<_>>());
        }
    }
}

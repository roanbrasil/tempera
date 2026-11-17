//! Statistical data audit (M1 checks: A1, A2, A3/A4 split guards, A5, A10) and
//! the resulting [`FeaturePlan`]. Statistics decide; the SLM advisor (M3) only
//! explains. See docs/wiki/Data-Audit.md and ADR-0004.

use std::collections::{BTreeMap, HashMap};

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tempera_core::metrics::roc_auc;
use tempera_core::spec::{AuditPolicy, JobSpec, SplitStrategy, TaskType};
use tempera_data::infer::TaskInference;
use tempera_data::ingest::Dataset;
use tempera_data::profile::{ColumnKind, ColumnProfile, Profile};
use tempera_data::split::{Feature, FeatureKind, FeaturePlan};

/// Rows sampled for the per-feature leakage check.
const LEAKAGE_SAMPLE: u32 = 200_000;
/// Single-feature AUC (or |correlation| for regression) at which a column is
/// almost certainly leaking the target.
const LEAK_RED: f64 = 0.98;
const LEAK_YELLOW: f64 = 0.93;

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error(transparent)]
    Data(#[from] tempera_data::DataError),
    #[error(transparent)]
    Polars(#[from] PolarsError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, AuditError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Green,
    Yellow,
    Red,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Resolution {
    /// Needs a decision (blocks training when red and policy is `block_on_red`).
    Unresolved,
    /// Handled automatically; `action` says how.
    Auto { action: String },
    /// The user acknowledged it in `audit.acknowledge`.
    Acknowledged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// Stable id, e.g. `A1:refund_amount` or `A5`. Used in `audit.acknowledge`.
    pub id: String,
    pub check: String,
    pub severity: Severity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,
    pub title: String,
    pub evidence: String,
    pub suggestion: String,
    pub resolution: Resolution,
}

impl Finding {
    pub fn blocks(&self) -> bool {
        self.severity == Severity::Red && self.resolution == Resolution::Unresolved
    }
}

/// `audit.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub dataset_id: String,
    pub policy: AuditPolicy,
    pub findings: Vec<Finding>,
    pub plan: FeaturePlan,
    /// True when training must not start (see [`Finding::blocks`]).
    pub blocked: bool,
}

impl AuditReport {
    pub fn blocking(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.blocks())
    }
}

/// Runs every M1 check and builds the feature plan.
pub fn audit(
    ds: &Dataset,
    spec: &JobSpec,
    profile: &Profile,
    inference: &TaskInference,
) -> Result<AuditReport> {
    let mut findings = Vec::new();
    let mut plan = FeaturePlan {
        dedupe: true,
        ..FeaturePlan::default()
    };
    let target = spec.data.target.clone().unwrap_or_default();
    let reserved: Vec<&str> = [
        Some(target.as_str()),
        spec.data.group_column.as_deref(),
        spec.data.time_column.as_deref(),
    ]
    .into_iter()
    .flatten()
    .collect();

    // ---- A10: column usability, and the initial plan ----
    for c in &profile.columns {
        if reserved.contains(&c.name.as_str()) {
            continue;
        }
        if spec.data.drop.contains(&c.name) {
            plan.dropped
                .insert(c.name.clone(), "dropped in the spec (`data.drop`)".into());
            continue;
        }
        if let Some((sev, title, why)) = unusable(c) {
            plan.dropped.insert(c.name.clone(), why.clone());
            findings.push(Finding {
                id: format!("A10:{}", c.name),
                check: "A10".into(),
                severity: sev,
                column: Some(c.name.clone()),
                title,
                evidence: why,
                suggestion: "Dropped automatically. Encode it yourself if it carries signal.".into(),
                resolution: Resolution::Auto {
                    action: "column dropped".into(),
                },
            });
            continue;
        }
        match c.kind {
            ColumnKind::Numeric => plan.features.push(Feature {
                name: c.name.clone(),
                kind: FeatureKind::Numeric,
            }),
            ColumnKind::Boolean => plan.features.push(Feature {
                name: c.name.clone(),
                kind: FeatureKind::Boolean,
            }),
            ColumnKind::Categorical => {
                plan.features.push(Feature {
                    name: c.name.clone(),
                    kind: FeatureKind::Categorical,
                });
            }
            ColumnKind::Datetime => plan.datetime.push(c.name.clone()),
            _ => {}
        }
    }

    // ---- A4 / A3: split strategy guards ----
    let explicit_mixing = matches!(
        spec.split.strategy,
        SplitStrategy::Random | SplitStrategy::Stratified
    );
    if let Some(tc) = &spec.data.time_column {
        if explicit_mixing {
            findings.push(Finding {
                id: "A4".into(),
                check: "A4".into(),
                severity: Severity::Red,
                column: Some(tc.clone()),
                title: "Random split on time-ordered data".into(),
                evidence: format!(
                    "`data.time_column: {tc}` is set but `split.strategy` mixes past and future rows."
                ),
                suggestion:
                    "Use `split.strategy: temporal` (or `auto`) so the model is tested on the future.".into(),
                resolution: Resolution::Unresolved,
            });
        }
    } else if let Some(dt) = profile
        .columns
        .iter()
        .find(|c| c.kind == ColumnKind::Datetime && !reserved.contains(&c.name.as_str()))
    {
        findings.push(Finding {
            id: format!("A4:{}", dt.name),
            check: "A4".into(),
            severity: Severity::Yellow,
            column: Some(dt.name.clone()),
            title: "Data has a date column but the split is not temporal".into(),
            evidence: format!(
                "`{}` ranges from {} to {}.",
                dt.name,
                dt.min_time.as_deref().unwrap_or("?"),
                dt.max_time.as_deref().unwrap_or("?")
            ),
            suggestion: format!(
                "If you will predict future rows, set `data.time_column: {}`.",
                dt.name
            ),
            resolution: Resolution::Unresolved,
        });
    }
    if let Some(gc) = &spec.data.group_column
        && explicit_mixing
    {
        findings.push(Finding {
            id: "A3".into(),
            check: "A3".into(),
            severity: Severity::Red,
            column: Some(gc.clone()),
            title: "Same entity can land in train and test".into(),
            evidence: format!("`data.group_column: {gc}` is set but the split ignores groups."),
            suggestion: "Use `split.strategy: group` (or `auto`).".into(),
            resolution: Resolution::Unresolved,
        });
    }

    // ---- A5: class imbalance ----
    if let Some(m) = inference.minority_share
        && m < 0.2
    {
        findings.push(Finding {
            id: "A5".into(),
            check: "A5".into(),
            severity: Severity::Yellow,
            column: Some(target.clone()),
            title: "Imbalanced classes".into(),
            evidence: format!("The rarest class is {:.1}% of the rows.", m * 100.0),
            suggestion: "Stratified split and a metric that is fair to the rare class.".into(),
            resolution: Resolution::Auto {
                action: format!(
                    "stratified split; metric `{}`",
                    serde_json::to_value(inference.metric)?.as_str().unwrap_or("")
                ),
            },
        });
    }

    // ---- target nulls ----
    if let Some(t) = profile.column(&target)
        && t.null_count > 0
    {
        findings.push(Finding {
            id: "target-nulls".into(),
            check: "A13".into(),
            severity: Severity::Yellow,
            column: Some(target.clone()),
            title: "Rows without a target".into(),
            evidence: format!(
                "{} rows ({:.1}%) have no `{target}`.",
                t.null_count,
                t.null_fraction * 100.0
            ),
            suggestion: "They are removed before training. Check why the label is missing.".into(),
            resolution: Resolution::Auto {
                action: "rows removed".into(),
            },
        });
    }

    // ---- A2 + A1 need the data ----
    let lf = ds.scan()?;
    let dup = duplicate_rows(lf.clone())?;
    if dup > 0 {
        findings.push(Finding {
            id: "A2".into(),
            check: "A2".into(),
            severity: Severity::Red,
            column: None,
            title: "Exact duplicate rows".into(),
            evidence: format!("{dup} rows are exact copies of another row; copies split across train and test inflate the score."),
            suggestion: "Deduplicate before splitting.".into(),
            resolution: Resolution::Auto { action: format!("{dup} duplicates removed before the split") },
        });
    }
    if !target.is_empty() {
        findings.extend(leakage(lf, &target, inference.task, &plan)?);
    }

    // ---- acknowledgements and policy ----
    for f in &mut findings {
        if f.resolution == Resolution::Unresolved
            && spec.audit.acknowledge.iter().any(|a| a == &f.id || a == &f.check)
        {
            f.resolution = Resolution::Acknowledged;
        }
    }
    findings.sort_by(|a, b| b.severity.cmp(&a.severity).then_with(|| a.id.cmp(&b.id)));
    let blocked = spec.audit.policy == AuditPolicy::BlockOnRed && findings.iter().any(Finding::blocks);
    Ok(AuditReport {
        dataset_id: ds.id.clone(),
        policy: spec.audit.policy,
        findings,
        plan,
        blocked,
    })
}

/// Writes `audit.json` next to the dataset.
pub fn save(ds: &Dataset, report: &AuditReport) -> Result<std::path::PathBuf> {
    let path = ds.dir.join("audit.json");
    std::fs::write(&path, serde_json::to_string_pretty(report)?)?;
    Ok(path)
}

fn unusable(c: &ColumnProfile) -> Option<(Severity, String, String)> {
    let id_name = {
        let n = c.name.to_ascii_lowercase();
        n == "id"
            || n.ends_with("_id")
            || n.ends_with("id") && n.len() <= 6
            || n.starts_with("id_")
            || n == "uuid"
    };
    match c.kind {
        ColumnKind::Constant => Some((
            Severity::Green,
            "Constant column".into(),
            format!("`{}` has a single value.", c.name),
        )),
        ColumnKind::HighCardinality => Some((
            Severity::Yellow,
            "Identifier-like column".into(),
            format!(
                "`{}` has {} distinct values in {} non-null rows (looks like an ID).",
                c.name,
                c.n_unique,
                (c.n_unique as f64 / c.unique_ratio.max(f64::EPSILON)).round()
            ),
        )),
        ColumnKind::Text => Some((
            Severity::Yellow,
            "Free text".into(),
            format!("`{}` is free text; text features arrive in M5.", c.name),
        )),
        ColumnKind::Numeric if id_name && c.unique_ratio > 0.95 => Some((
            Severity::Yellow,
            "Identifier-like column".into(),
            format!(
                "`{}` is numeric but named like an ID and {:.0}% unique.",
                c.name,
                c.unique_ratio * 100.0
            ),
        )),
        _ if c.null_fraction > 0.95 => Some((
            Severity::Yellow,
            "Almost empty column".into(),
            format!("`{}` is {:.0}% null.", c.name, c.null_fraction * 100.0),
        )),
        _ => None,
    }
}

fn duplicate_rows(lf: LazyFrame) -> Result<u64> {
    let df = lf.collect()?;
    let n = df.height();
    let unique = df.unique_stable(None, UniqueKeepStrategy::First, None)?.height();
    Ok((n - unique) as u64)
}

/// A1: how well does each feature *alone* predict the target?
fn leakage(lf: LazyFrame, target: &str, task: TaskType, plan: &FeaturePlan) -> Result<Vec<Finding>> {
    let mut cols: Vec<Expr> = plan.features.iter().map(|f| col(f.name.as_str())).collect();
    cols.push(col(target));
    let df = lf
        .select(cols)
        .filter(col(target).is_not_null())
        .limit(LEAKAGE_SAMPLE)
        .collect()?;
    if df.height() < 20 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for f in &plan.features {
        let score = match task {
            TaskType::BinaryClassification => binary_auc(&df, &f.name, f.kind, target)?,
            TaskType::Regression => regression_strength(&df, &f.name, f.kind, target)?,
            TaskType::MulticlassClassification => multiclass_purity(&df, &f.name, target)?,
            _ => None,
        };
        let Some(score) = score else { continue };
        let severity = if score >= LEAK_RED {
            Severity::Red
        } else if score >= LEAK_YELLOW {
            Severity::Yellow
        } else {
            continue;
        };
        let measure = match task {
            TaskType::BinaryClassification => "single-feature AUC",
            TaskType::Regression => "single-feature strength (|Spearman| or correlation ratio)",
            _ => "single-feature accuracy gain",
        };
        out.push(Finding {
            id: format!("A1:{}", f.name),
            check: "A1".into(),
            severity,
            column: Some(f.name.clone()),
            title: if severity == Severity::Red { "Probable target leakage".into() } else { "Suspiciously predictive column".into() },
            evidence: format!("`{}` alone predicts `{target}` with {measure} = {score:.3}.", f.name),
            suggestion: format!(
                "If `{}` is only known after the outcome, add it to `data.drop`. If it is legitimate, add `A1:{}` to `audit.acknowledge`.",
                f.name, f.name
            ),
            resolution: Resolution::Unresolved,
        });
    }
    Ok(out)
}

/// Feature values as scores: numeric as-is, categorical/boolean via a smoothed
/// in-sample target rate per category.
fn binary_auc(df: &DataFrame, feature: &str, kind: FeatureKind, target: &str) -> Result<Option<f64>> {
    let y_raw = df.column(target)?.cast(&DataType::String)?;
    let y_str = y_raw.str()?;
    let labels: Vec<Option<&str>> = y_str.iter().collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for l in labels.iter().flatten() {
        *counts.entry(*l).or_default() += 1;
    }
    let Some((&positive, _)) = counts.iter().min_by_key(|(_, n)| **n) else {
        return Ok(None);
    };
    let y: Vec<bool> = labels.iter().map(|l| *l == Some(positive)).collect();
    let scores = feature_scores(
        df,
        feature,
        kind,
        &y.iter().map(|&b| f64::from(u8::from(b))).collect::<Vec<_>>(),
    )?;
    let (ys, ss): (Vec<bool>, Vec<f64>) = y
        .iter()
        .zip(&scores)
        .filter_map(|(y, s)| s.map(|s| (*y, s)))
        .unzip();
    if ys.len() < 20 {
        return Ok(None);
    }
    let auc = roc_auc(&ys, &ss);
    Ok((!auc.is_nan()).then(|| auc.max(1.0 - auc)))
}

fn regression_strength(
    df: &DataFrame,
    feature: &str,
    kind: FeatureKind,
    target: &str,
) -> Result<Option<f64>> {
    let y: Vec<Option<f64>> = df
        .column(target)?
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .collect();
    let y_filled: Vec<f64> = y.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let scores = feature_scores(df, feature, kind, &y_filled)?;
    let (a, b): (Vec<f64>, Vec<f64>) = scores
        .iter()
        .zip(&y)
        .filter_map(|(s, y)| Some((s.as_ref().copied()?, (*y)?)))
        .filter(|(s, y)| s.is_finite() && y.is_finite())
        .unzip();
    if a.len() < 20 {
        return Ok(None);
    }
    Ok(Some(pearson(&ranks(&a), &ranks(&b)).abs()))
}

/// Normalized accuracy gain of mapping each value (or numeric decile) to its majority class.
fn multiclass_purity(df: &DataFrame, feature: &str, target: &str) -> Result<Option<f64>> {
    let x = df.column(feature)?.cast(&DataType::String)?;
    let y = df.column(target)?.cast(&DataType::String)?;
    let (x, y) = (x.str()?, y.str()?);
    let mut table: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    let mut overall: HashMap<&str, usize> = HashMap::new();
    let mut n = 0usize;
    for (xv, yv) in x.iter().zip(y.iter()) {
        let (Some(xv), Some(yv)) = (xv, yv) else { continue };
        *table.entry(xv).or_default().entry(yv).or_default() += 1;
        *overall.entry(yv).or_default() += 1;
        n += 1;
    }
    // Too many distinct values makes the mapping memorize rows.
    if n < 20 || table.len() > n / 10 {
        return Ok(None);
    }
    let base = *overall.values().max().unwrap_or(&0) as f64 / n as f64;
    let hit: usize = table.values().map(|m| *m.values().max().unwrap_or(&0)).sum();
    let acc = hit as f64 / n as f64;
    Ok((base < 1.0).then(|| (acc - base) / (1.0 - base)))
}

fn feature_scores(df: &DataFrame, feature: &str, kind: FeatureKind, y: &[f64]) -> Result<Vec<Option<f64>>> {
    let c = df.column(feature)?;
    Ok(match kind {
        FeatureKind::Numeric => c
            .cast(&DataType::Float64)?
            .f64()?
            .iter()
            .map(|v| v.filter(|x| x.is_finite()))
            .collect(),
        FeatureKind::Categorical | FeatureKind::Boolean => {
            let s = c.cast(&DataType::String)?;
            let s = s.str()?;
            let global = y.iter().filter(|v| v.is_finite()).sum::<f64>() / y.len().max(1) as f64;
            let mut agg: HashMap<&str, (f64, f64)> = HashMap::new();
            for (v, t) in s.iter().zip(y) {
                if let Some(v) = v
                    && t.is_finite()
                {
                    let e = agg.entry(v).or_default();
                    e.0 += t;
                    e.1 += 1.0;
                }
            }
            // Smoothing: rare categories shrink to the global rate, so IDs do not look predictive.
            const M: f64 = 10.0;
            s.iter()
                .map(|v| v.map(|v| agg.get(v).map_or(global, |(sum, n)| (sum + M * global) / (n + M))))
                .collect()
        }
    })
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        for &k in &idx[i..=j] {
            r[k] = (i + j) as f64 / 2.0;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        cov / (va.sqrt() * vb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use tempera_data::{infer_task, ingest, profile_dataset};

    /// Churn-like data with a planted leak (`refund_amount` exists only for churners),
    /// an ID column, a constant and a duplicate row.
    fn fixture(dir: &std::path::Path) -> std::path::PathBuf {
        let mut csv = String::from("customer_id,tenure,plan,refund_amount,country,signup,churned\n");
        for i in 0..2000u64 {
            let h = i.wrapping_mul(2654435761) % 1000;
            let churned = h < 150;
            let tenure = if churned { h % 24 } else { 12 + h % 48 };
            let refund = if churned { 20.0 + (h % 50) as f64 } else { 0.0 };
            csv += &format!(
                "{i},{tenure},{},{refund},BR,2024-{:02}-{:02},{}\n",
                ["basic", "pro", "team"][(h % 3) as usize],
                1 + h % 12,
                1 + h % 28,
                if churned { "yes" } else { "no" }
            );
        }
        csv += "1,25,pro,0,BR,2024-02-02,no\n1,25,pro,0,BR,2024-02-02,no\n";
        let p = dir.join("churn.csv");
        std::fs::write(&p, csv).unwrap();
        p
    }

    fn run(extra: &str) -> AuditReport {
        let tmp = tempfile::tempdir().unwrap();
        let src = fixture(tmp.path());
        let ds = ingest(&src, tmp.path()).unwrap();
        let profile = profile_dataset(&ds).unwrap();
        let spec = JobSpec::from_yaml(&format!(
            "version: 1\nname: t\ndata: {{ source: churn.csv, target: churned }}\n{extra}"
        ))
        .unwrap();
        let inf = infer_task(&spec, &profile).unwrap();
        audit(&ds, &spec, &profile, &inf).unwrap()
    }

    fn ids(r: &AuditReport) -> Vec<&str> {
        r.findings.iter().map(|f| f.id.as_str()).collect()
    }

    #[test]
    fn detects_planted_problems() {
        let r = run("");
        let ids = ids(&r);
        assert!(ids.contains(&"A1:refund_amount"), "{ids:?}");
        assert!(ids.contains(&"A2"), "{ids:?}");
        assert!(ids.contains(&"A5"), "{ids:?}");
        assert!(ids.contains(&"A10:customer_id"), "{ids:?}");
        assert!(ids.contains(&"A10:country"), "{ids:?}");
        assert!(ids.contains(&"A4:signup"), "{ids:?}");
        assert!(!ids.contains(&"A1:plan"), "{ids:?}");
        assert!(r.blocked);
        let blocking: Vec<_> = r.blocking().map(|f| f.id.as_str()).collect();
        assert_eq!(blocking, ["A1:refund_amount"]);
        assert!(r.plan.dropped.contains_key("customer_id"));
        assert_eq!(r.plan.datetime, ["signup"]);
    }

    #[test]
    fn drop_or_acknowledge_unblocks() {
        let r = run("");
        assert!(r.blocked);
        let dropped = {
            let tmp = tempfile::tempdir().unwrap();
            let ds = ingest(&fixture(tmp.path()), tmp.path()).unwrap();
            let profile = profile_dataset(&ds).unwrap();
            let spec = JobSpec::from_yaml(
                "version: 1\nname: t\ndata: { source: churn.csv, target: churned, drop: [refund_amount] }\n",
            )
            .unwrap();
            let inf = infer_task(&spec, &profile).unwrap();
            audit(&ds, &spec, &profile, &inf).unwrap()
        };
        assert!(!dropped.blocked);
        assert!(!dropped.plan.features.iter().any(|f| f.name == "refund_amount"));
        let acked = run("audit: { acknowledge: [\"A1:refund_amount\"] }\n");
        assert!(!acked.blocked);
        let warn_only = run("audit: { policy: warn_only }\n");
        assert!(!warn_only.blocked);
    }

    #[test]
    fn explicit_random_split_with_time_column_blocks() {
        let tmp = tempfile::tempdir().unwrap();
        let ds = ingest(&fixture(tmp.path()), tmp.path()).unwrap();
        let profile = profile_dataset(&ds).unwrap();
        let spec = JobSpec::from_yaml(
            "version: 1\nname: t\ndata: { source: c.csv, target: churned, time_column: signup, drop: [refund_amount] }\nsplit: { strategy: random }\n",
        )
        .unwrap();
        let inf = infer_task(&spec, &profile).unwrap();
        let r = audit(&ds, &spec, &profile, &inf).unwrap();
        assert_eq!(r.blocking().map(|f| f.id.as_str()).collect::<Vec<_>>(), ["A4"]);
    }
}

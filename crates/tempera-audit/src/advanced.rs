//! M3 checks (#30): A3 repeated entities, A6 train/test drift, A12 small
//! samples, A13 informative missingness. See docs/wiki/Data-Audit.md.

use std::collections::HashMap;
use std::path::Path;

use polars::prelude::*;
use tempera_core::metrics::roc_auc;
use tempera_core::spec::{JobSpec, SplitStrategy, TaskType};
use tempera_data::infer::TaskInference;
use tempera_data::profile::Profile;
use tempera_data::split::{Feature, FeatureKind, FeaturePlan};

use crate::{Finding, Resolution, Result, Severity};

const ENTITY_WORDS: [&str; 10] = [
    "id", "customer", "client", "user", "account", "patient", "member", "device", "session", "order",
];

/// A3: a column whose values repeat a few times per row, named like an entity,
/// while the split ignores groups → the same entity lands in train and test.
pub fn repeated_entities(profile: &Profile, spec: &JobSpec, reserved: &[&str]) -> Vec<Finding> {
    if spec.data.group_column.is_some() || matches!(spec.split.strategy, SplitStrategy::Group) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for c in &profile.columns {
        if reserved.contains(&c.name.as_str()) || !(c.dtype.starts_with("str") || c.dtype.starts_with('i')) {
            continue;
        }
        let name = c.name.to_ascii_lowercase();
        let entity_like = ENTITY_WORDS.iter().any(|w| {
            name == *w
                || name.contains(&format!("{w}_"))
                || name.ends_with(&format!("_{w}"))
                || name.ends_with("id")
        });
        let rows_per_value = if c.unique_ratio > 0.0 {
            1.0 / c.unique_ratio
        } else {
            0.0
        };
        if entity_like && (1.3..=200.0).contains(&rows_per_value) && c.n_unique >= 10 {
            out.push(Finding {
                id: format!("A3:{}", c.name),
                check: "A3".into(),
                severity: Severity::Yellow,
                column: Some(c.name.clone()),
                title: "Repeated entity without a group split".into(),
                evidence: format!(
                    "`{}` has {} distinct values for {:.1} rows each on average: the same entity can be in train and test.",
                    c.name, c.n_unique, rows_per_value
                ),
                suggestion: format!("If rows of one `{}` are related, set `data.group_column: {}`.", c.name, c.name),
                resolution: Resolution::Unresolved,
            });
        }
    }
    out
}

/// A12: too little data for the task.
pub fn small_sample(
    profile: &Profile,
    inference: &TaskInference,
    plan: &FeaturePlan,
    target: &str,
) -> Vec<Finding> {
    let mut out = Vec::new();
    let n_features = plan.features.len() + plan.datetime.len() * 3;
    let rows = profile.rows;
    let mut push = |id: &str, evidence: String, suggestion: &str| {
        out.push(Finding {
            id: id.to_owned(),
            check: "A12".into(),
            severity: Severity::Yellow,
            column: None,
            title: "Small sample".into(),
            evidence,
            suggestion: suggestion.to_owned(),
            resolution: Resolution::Unresolved,
        });
    };
    if matches!(
        inference.task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    ) && let Some(col) = profile.column(target)
        && col.top_values.len() as u64 >= col.n_unique
        && let Some(min) = col.top_values.iter().map(|v| v.count).min()
        && min < 50
    {
        push(
            "A12:class",
            format!("The rarest class has only {min} rows; validation and test scores will be noisy."),
            "Collect more examples of the rare class, or treat the score as a rough estimate.",
        );
    }
    if inference.task == TaskType::Regression && rows < 200 {
        push(
            "A12:rows",
            format!("Only {rows} rows for regression."),
            "Prefer simple models (linear, ridge) and more data.",
        );
    }
    if n_features > 0 && (rows as f64) < 10.0 * n_features as f64 {
        push(
            "A12:ratio",
            format!("{rows} rows for {n_features} features (fewer than 10 rows per feature)."),
            "Drop weak features or prefer regularized models; complex models will overfit.",
        );
    }
    out
}

/// A13: rows where a feature is missing have a very different target.
pub fn informative_missingness(
    lf: LazyFrame,
    target: &str,
    task: TaskType,
    plan: &FeaturePlan,
    profile: &Profile,
) -> Result<Vec<Finding>> {
    let candidates: Vec<&Feature> = plan
        .features
        .iter()
        .filter(|f| {
            profile
                .column(&f.name)
                .is_some_and(|c| c.null_fraction >= 0.02 && c.null_fraction <= 0.95)
        })
        .collect();
    if candidates.is_empty() || !matches!(task, TaskType::BinaryClassification | TaskType::Regression) {
        return Ok(Vec::new());
    }
    let mut cols: Vec<Expr> = candidates
        .iter()
        .map(|f| col(f.name.as_str()).is_null().alias(f.name.as_str()))
        .collect();
    cols.push(col(target));
    let df = lf
        .select(cols)
        .filter(col(target).is_not_null())
        .limit(500_000)
        .collect()?;
    let y: Vec<f64> = if task == TaskType::BinaryClassification {
        let s = df.column(target)?.cast(&DataType::String)?;
        let labels: Vec<Option<String>> = s.str()?.iter().map(|v| v.map(str::to_owned)).collect();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for l in labels.iter().flatten() {
            *counts.entry(l.as_str()).or_default() += 1;
        }
        let Some((&pos, _)) = counts.iter().min_by_key(|(_, n)| **n) else {
            return Ok(Vec::new());
        };
        labels
            .iter()
            .map(|l| f64::from(u8::from(l.as_deref() == Some(pos))))
            .collect()
    } else {
        df.column(target)?
            .cast(&DataType::Float64)?
            .f64()?
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect()
    };
    let spread = {
        let finite: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
        let m = finite.iter().sum::<f64>() / finite.len().max(1) as f64;
        (finite.iter().map(|v| (v - m).powi(2)).sum::<f64>() / finite.len().max(1) as f64).sqrt()
    };
    let mut out = Vec::new();
    for f in candidates {
        let missing: Vec<bool> = df
            .column(&f.name)?
            .bool()?
            .iter()
            .map(|v| v.unwrap_or(false))
            .collect();
        let (mut a, mut na, mut b, mut nb) = (0.0, 0usize, 0.0, 0usize);
        for (m, v) in missing.iter().zip(&y) {
            if !v.is_finite() {
                continue;
            }
            if *m {
                a += v;
                na += 1;
            } else {
                b += v;
                nb += 1;
            }
        }
        if na < 30 || nb < 30 {
            continue;
        }
        let (ma, mb) = (a / na as f64, b / nb as f64);
        let (strong, evidence) = if task == TaskType::BinaryClassification {
            (
                (ma - mb).abs() >= 0.15,
                format!(
                    "Positive rate is {:.1}% when `{}` is missing vs {:.1}% when present.",
                    ma * 100.0,
                    f.name,
                    mb * 100.0
                ),
            )
        } else {
            (
                spread > 0.0 && (ma - mb).abs() >= 0.5 * spread,
                format!(
                    "Mean target is {ma:.3} when `{}` is missing vs {mb:.3} when present.",
                    f.name
                ),
            )
        };
        if strong {
            out.push(Finding {
                id: format!("A13:{}", f.name),
                check: "A13".into(),
                severity: Severity::Yellow,
                column: Some(f.name.clone()),
                title: "Missingness predicts the target".into(),
                evidence,
                suggestion: format!(
                    "Check why `{}` is missing. If it is filled only after the outcome, it leaks; otherwise models can use the missing pattern.",
                    f.name
                ),
                resolution: Resolution::Unresolved,
            });
        }
    }
    Ok(out)
}

/// A6: does the test partition look like the training one? Per feature:
/// population stability index (numeric, train deciles) and single-feature
/// adversarial AUC (can a feature alone tell train from test?).
pub fn drift(
    train: &Path,
    test: &Path,
    features: &[Feature],
    strategy: SplitStrategy,
) -> Result<Vec<Finding>> {
    let names: Vec<Expr> = features.iter().map(|f| col(f.name.as_str())).collect();
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let tr = tempera_data::ingest::scan_parquet(train)?
        .select(names.clone())
        .limit(200_000)
        .collect()?;
    let te = tempera_data::ingest::scan_parquet(test)?
        .select(names)
        .limit(200_000)
        .collect()?;
    let mut out = Vec::new();
    for f in features {
        let (psi, auc) = match f.kind {
            FeatureKind::Numeric => {
                let a = finite(tr.column(&f.name)?)?;
                let b = finite(te.column(&f.name)?)?;
                if a.len() < 30 || b.len() < 30 {
                    continue;
                }
                (psi_numeric(&a, &b), adversarial_auc(&a, &b))
            }
            FeatureKind::Categorical | FeatureKind::Boolean => {
                let a = strings(tr.column(&f.name)?)?;
                let b = strings(te.column(&f.name)?)?;
                if a.len() < 30 || b.len() < 30 {
                    continue;
                }
                (psi_categorical(&a, &b), f64::NAN)
            }
        };
        let auc = if auc.is_nan() { 0.5 } else { auc.max(1.0 - auc) };
        if psi >= 0.25 || auc >= 0.7 {
            let expected = if strategy == SplitStrategy::Temporal {
                " Some drift is expected with a temporal split: it is what production will look like."
            } else {
                ""
            };
            out.push(Finding {
                id: format!("A6:{}", f.name),
                check: "A6".into(),
                severity: Severity::Yellow,
                column: Some(f.name.clone()),
                title: "Train and test distributions differ".into(),
                evidence: format!(
                    "`{}`: PSI = {psi:.2}, single-feature train-vs-test AUC = {auc:.2}.{expected}",
                    f.name
                ),
                suggestion: "Scores may not transfer. Check how the data was collected; consider reweighting or a different split.".into(),
                resolution: Resolution::Auto { action: "reported".into() },
            });
        }
    }
    Ok(out)
}

fn finite(c: &Column) -> Result<Vec<f64>> {
    Ok(c.cast(&DataType::Float64)?
        .f64()?
        .iter()
        .flatten()
        .filter(|v| v.is_finite())
        .collect())
}

fn strings(c: &Column) -> Result<Vec<String>> {
    Ok(c.cast(&DataType::String)?
        .str()?
        .iter()
        .map(|v| v.unwrap_or("<null>").to_owned())
        .collect())
}

/// PSI over train deciles (with a small floor so empty bins do not explode).
fn psi_numeric(a: &[f64], b: &[f64]) -> f64 {
    let mut sorted = a.to_vec();
    sorted.sort_by(f64::total_cmp);
    let edges: Vec<f64> = (1..10)
        .map(|k| sorted[(k * sorted.len() / 10).min(sorted.len() - 1)])
        .collect();
    let bin = |v: f64| edges.iter().take_while(|e| v > **e).count();
    let hist = |xs: &[f64]| {
        let mut h = [0f64; 10];
        for &x in xs {
            h[bin(x)] += 1.0;
        }
        h.map(|c| (c / xs.len() as f64).max(1e-4))
    };
    let (p, q) = (hist(a), hist(b));
    p.iter().zip(&q).map(|(p, q)| (q - p) * (q / p).ln()).sum()
}

fn freq(xs: &[String]) -> HashMap<&str, f64> {
    let mut m: HashMap<&str, f64> = HashMap::new();
    for x in xs {
        *m.entry(x.as_str()).or_default() += 1.0 / xs.len() as f64;
    }
    m
}

fn psi_categorical(a: &[String], b: &[String]) -> f64 {
    let (p, q) = (freq(a), freq(b));
    let keys: std::collections::BTreeSet<&str> = p.keys().chain(q.keys()).copied().collect();
    keys.iter()
        .map(|k| {
            let (pk, qk) = (
                p.get(k).copied().unwrap_or(0.0).max(1e-4),
                q.get(k).copied().unwrap_or(0.0).max(1e-4),
            );
            (qk - pk) * (qk / pk).ln()
        })
        .sum()
}

fn adversarial_auc(a: &[f64], b: &[f64]) -> f64 {
    let y: Vec<bool> = std::iter::repeat_n(false, a.len())
        .chain(std::iter::repeat_n(true, b.len()))
        .collect();
    let s: Vec<f64> = a.iter().chain(b).copied().collect();
    roc_auc(&y, &s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psi_detects_shift_and_ignores_identical() {
        let a: Vec<f64> = (0..1000).map(f64::from).collect();
        let same = a.clone();
        let shifted: Vec<f64> = (500..1500).map(f64::from).collect();
        assert!(psi_numeric(&a, &same) < 0.01);
        assert!(psi_numeric(&a, &shifted) > 0.25);
        assert!(adversarial_auc(&a, &shifted) > 0.7);
        let c1: Vec<String> = (0..300).map(|i| ["a", "b", "c"][i % 3].to_owned()).collect();
        let c2: Vec<String> = (0..300)
            .map(|i| if i % 10 == 0 { "a" } else { "c" }.to_owned())
            .collect();
        assert!(psi_categorical(&c1, &c1) < 0.01);
        assert!(psi_categorical(&c1, &c2) > 0.25);
    }
}

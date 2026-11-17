//! Scores a predictions Parquet against the truth (Contracts.md §7).
//!
//! Every backend is evaluated here with the same code (`tempera_core::metrics`).

use std::collections::BTreeMap;
use std::path::Path;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use tempera_core::metrics as m;
use tempera_core::spec::{Metric, TaskType};

use crate::error::{DataError, Result};
use crate::ingest::scan_parquet;
use crate::split::TARGET;

const CURVE_POINTS: usize = 100;
const RESIDUAL_SAMPLE: usize = 500;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub rows: u64,
    /// Metric name (snake_case, as in `Metric`) → value.
    pub metrics: BTreeMap<String, f64>,
    /// `confusion[true][pred]`, classes in `classes` order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub confusion: Vec<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
    /// Binary only: `(fpr, tpr)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roc: Vec<(f64, f64)>,
    /// Binary only: `(recall, precision)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pr: Vec<(f64, f64)>,
    /// Regression only: sample of `(truth, prediction)`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub residuals: Vec<(f64, f64)>,
}

impl Evaluation {
    pub fn get(&self, metric: Metric) -> Option<f64> {
        self.metrics.get(&metric_name(metric)).copied()
    }
}

pub fn metric_name(metric: Metric) -> String {
    serde_json::to_value(metric)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Evaluates `predictions` (same row order as `truth`).
pub fn evaluate(
    task: TaskType,
    classes: &[String],
    positive_class: Option<&str>,
    truth: &Path,
    predictions: &Path,
) -> Result<Evaluation> {
    let t = scan_parquet(truth)?.select([col(TARGET)]).collect()?;
    let p = scan_parquet(predictions)?.collect()?;
    if t.height() != p.height() {
        return Err(DataError::Invalid(format!(
            "predictions have {} rows but the truth has {}",
            p.height(),
            t.height()
        )));
    }
    match task {
        TaskType::BinaryClassification | TaskType::MulticlassClassification => classification(
            &t,
            &p,
            classes,
            positive_class,
            task == TaskType::BinaryClassification,
        ),
        TaskType::Regression => regression(&t, &p),
        other => Err(DataError::Invalid(format!(
            "evaluation of {other:?} is not supported yet"
        ))),
    }
}

fn classification(
    t: &DataFrame,
    p: &DataFrame,
    classes: &[String],
    positive: Option<&str>,
    binary: bool,
) -> Result<Evaluation> {
    let index: BTreeMap<&str, usize> = classes.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
    let lookup = |s: Option<&str>| s.and_then(|s| index.get(s).copied()).unwrap_or(usize::MAX);
    let y: Vec<usize> = t
        .column(TARGET)?
        .cast(&DataType::String)?
        .str()?
        .iter()
        .map(lookup)
        .collect();
    let pred: Vec<usize> = p
        .column("prediction")?
        .cast(&DataType::String)?
        .str()?
        .iter()
        .map(lookup)
        .collect();
    let mut proba: Vec<Vec<f64>> = vec![Vec::with_capacity(classes.len()); y.len()];
    for c in classes {
        let col_name = format!("proba__{c}");
        let s = p
            .column(&col_name)
            .map_err(|_| DataError::MissingColumn(col_name.clone()))?;
        for (row, v) in s.cast(&DataType::Float64)?.f64()?.iter().enumerate() {
            proba[row].push(v.unwrap_or(0.0));
        }
    }

    let n = classes.len();
    let mut metrics = BTreeMap::new();
    metrics.insert("accuracy".into(), m::accuracy(&y, &pred));
    metrics.insert("balanced_accuracy".into(), m::balanced_accuracy(&y, &pred, n));
    metrics.insert("f1".into(), m::f1(&y, &pred, n));
    metrics.insert("log_loss".into(), m::log_loss(&y, &proba));
    let (mut roc, mut pr) = (Vec::new(), Vec::new());
    if binary {
        let pos = positive
            .and_then(|c| index.get(c).copied())
            .unwrap_or(1.min(n.saturating_sub(1)));
        let yb: Vec<bool> = y.iter().map(|&k| k == pos).collect();
        let score: Vec<f64> = proba.iter().map(|p| p.get(pos).copied().unwrap_or(0.0)).collect();
        metrics.insert("roc_auc".into(), m::roc_auc(&yb, &score));
        metrics.insert("pr_auc".into(), m::pr_auc(&yb, &score));
        // F1 of the positive class (the generic call scores class index 1).
        let yi: Vec<usize> = yb.iter().map(|&b| usize::from(b)).collect();
        let pi: Vec<usize> = pred.iter().map(|&k| usize::from(k == pos)).collect();
        metrics.insert("f1".into(), m::f1(&yi, &pi, 2));
        roc = m::roc_curve(&yb, &score, CURVE_POINTS);
        pr = m::pr_curve(&yb, &score, CURVE_POINTS);
    } else {
        // One-vs-rest macro ROC-AUC.
        let aucs: Vec<f64> = (0..n)
            .map(|k| {
                let yb: Vec<bool> = y.iter().map(|&c| c == k).collect();
                let s: Vec<f64> = proba.iter().map(|p| p[k]).collect();
                m::roc_auc(&yb, &s)
            })
            .filter(|a| !a.is_nan())
            .collect();
        if !aucs.is_empty() {
            metrics.insert("roc_auc".into(), aucs.iter().sum::<f64>() / aucs.len() as f64);
        }
    }
    Ok(Evaluation {
        rows: y.len() as u64,
        metrics,
        confusion: m::confusion_matrix(&y, &pred, n),
        classes: classes.to_vec(),
        roc,
        pr,
        residuals: Vec::new(),
    })
}

fn regression(t: &DataFrame, p: &DataFrame) -> Result<Evaluation> {
    let y: Vec<f64> = t
        .column(TARGET)?
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let pred: Vec<f64> = p
        .column("prediction")?
        .cast(&DataType::Float64)?
        .f64()?
        .iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let mut metrics = BTreeMap::new();
    metrics.insert("rmse".into(), m::rmse(&y, &pred));
    metrics.insert("mae".into(), m::mae(&y, &pred));
    metrics.insert("mape".into(), m::mape(&y, &pred));
    metrics.insert("r2".into(), m::r2(&y, &pred));
    let step = (y.len() / RESIDUAL_SAMPLE).max(1);
    let residuals = y.iter().zip(&pred).step_by(step).map(|(a, b)| (*a, *b)).collect();
    Ok(Evaluation {
        rows: y.len() as u64,
        metrics,
        confusion: Vec::new(),
        classes: Vec::new(),
        roc: Vec::new(),
        pr: Vec::new(),
        residuals,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn write(dir: &Path, name: &str, mut df: DataFrame) -> std::path::PathBuf {
        let p = dir.join(name);
        ParquetWriter::new(std::fs::File::create(&p).unwrap())
            .finish(&mut df)
            .unwrap();
        p
    }

    #[test]
    fn binary_evaluation() {
        let tmp = tempfile::tempdir().unwrap();
        let truth = write(
            tmp.path(),
            "t.parquet",
            df!(TARGET => ["no", "no", "yes", "yes"]).unwrap(),
        );
        let pred = write(
            tmp.path(),
            "p.parquet",
            df!(
                "prediction" => ["no", "yes", "no", "yes"],
                "proba__no" => [0.9, 0.6, 0.65, 0.2],
                "proba__yes" => [0.1, 0.4, 0.35, 0.8],
            )
            .unwrap(),
        );
        let classes = vec!["no".to_owned(), "yes".to_owned()];
        let e = evaluate(
            TaskType::BinaryClassification,
            &classes,
            Some("yes"),
            &truth,
            &pred,
        )
        .unwrap();
        assert!((e.get(Metric::RocAuc).unwrap() - 0.75).abs() < 1e-12);
        assert!((e.get(Metric::Accuracy).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(e.confusion, vec![vec![1, 1], vec![1, 1]]);
        assert!(!e.roc.is_empty());
    }

    #[test]
    fn regression_evaluation_and_row_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let truth = write(
            tmp.path(),
            "t.parquet",
            df!(TARGET => [3.0, -0.5, 2.0, 7.0]).unwrap(),
        );
        let pred = write(
            tmp.path(),
            "p.parquet",
            df!("prediction" => [2.5, 0.0, 2.0, 8.0]).unwrap(),
        );
        let e = evaluate(TaskType::Regression, &[], None, &truth, &pred).unwrap();
        assert!((e.get(Metric::Mae).unwrap() - 0.5).abs() < 1e-12);
        let short = write(tmp.path(), "s.parquet", df!("prediction" => [1.0]).unwrap());
        assert!(evaluate(TaskType::Regression, &[], None, &truth, &short).is_err());
    }
}

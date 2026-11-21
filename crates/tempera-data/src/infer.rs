//! Task type and metric inference, with a human-readable justification.

use serde::{Deserialize, Serialize};
use tempera_core::spec::{JobSpec, Metric, TaskType};

use crate::error::{DataError, Result};
use crate::profile::{ColumnKind, ColumnProfile, Profile};

/// Integer targets with at most this many distinct values are treated as classes.
const MAX_INT_CLASSES: u64 = 10;
/// String targets with more classes than this are rejected for classification.
const MAX_CLASSES: u64 = 200;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskInference {
    pub task: TaskType,
    pub metric: Metric,
    /// Why this task and metric were chosen (shown to the user).
    pub reasons: Vec<String>,
    /// Share of the rarest class (classification only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minority_share: Option<f64>,
}

/// Resolves the task and metric: explicit values in the spec win; anything
/// missing is inferred from the target column profile.
pub fn infer_task(spec: &JobSpec, profile: &Profile) -> Result<TaskInference> {
    let Some(target) = &spec.data.target else {
        let task = spec.task.kind.unwrap_or(TaskType::Clustering);
        let default_metric = if task == TaskType::AnomalyDetection {
            Metric::PrAuc
        } else {
            Metric::Silhouette
        };
        let metric = spec.task.metric.unwrap_or(default_metric);
        let mut reasons = vec![format!(
            "No target column → unsupervised task `{}`.",
            snake(&task)
        )];
        if task == TaskType::Clustering {
            reasons
                .push("Candidates are ranked by silhouette (cohesion vs separation of clusters).".to_owned());
        } else if task == TaskType::AnomalyDetection {
            reasons.push(
                "Without labels there is nothing to rank: one detector runs and flags outliers.".to_owned(),
            );
        }
        return Ok(TaskInference {
            task,
            metric,
            reasons,
            minority_share: None,
        });
    };
    let col = profile
        .column(target)
        .ok_or_else(|| DataError::MissingColumn(target.clone()))?;
    let mut reasons = Vec::new();
    let minority = minority_share(col);

    let task = match spec.task.kind {
        Some(t) => {
            reasons.push(format!("Task `{}` set in the spec.", snake(&t)));
            t
        }
        None => {
            let t = infer_from_target(col)?;
            reasons.push(task_reason(t, col));
            t
        }
    };

    let metric = match spec.task.metric {
        Some(m) => {
            reasons.push(format!("Metric `{}` set in the spec.", snake(&m)));
            m
        }
        None => {
            let (m, why) = recommend_metric(task, col, minority);
            reasons.push(why);
            m
        }
    };
    let minority_share = matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    )
    .then_some(minority)
    .flatten();
    Ok(TaskInference {
        task,
        metric,
        reasons,
        minority_share,
    })
}

fn infer_from_target(col: &ColumnProfile) -> Result<TaskType> {
    let n = col.n_unique;
    if n < 2 {
        return Err(DataError::Invalid(format!(
            "target `{}` has {n} distinct value(s); nothing to learn",
            col.name
        )));
    }
    let integer_like = col.dtype.starts_with('i') || col.dtype.starts_with('u');
    Ok(match col.kind {
        ColumnKind::Boolean => TaskType::BinaryClassification,
        _ if n == 2 => TaskType::BinaryClassification,
        ColumnKind::Numeric if integer_like && n <= MAX_INT_CLASSES => TaskType::MulticlassClassification,
        ColumnKind::Numeric => TaskType::Regression,
        ColumnKind::Categorical | ColumnKind::HighCardinality | ColumnKind::Text if n <= MAX_CLASSES => {
            TaskType::MulticlassClassification
        }
        _ => {
            return Err(DataError::Invalid(format!(
                "cannot infer a task for target `{}` ({}, {n} distinct values); set `task.type`",
                col.name, col.dtype
            )));
        }
    })
}

fn task_reason(task: TaskType, col: &ColumnProfile) -> String {
    match task {
        TaskType::BinaryClassification => {
            format!(
                "Target `{}` has exactly 2 distinct values → binary classification.",
                col.name
            )
        }
        TaskType::MulticlassClassification => format!(
            "Target `{}` has {} distinct values → multiclass classification.",
            col.name, col.n_unique
        ),
        TaskType::Regression => format!(
            "Target `{}` is numeric with {} distinct values → regression.",
            col.name, col.n_unique
        ),
        other => format!("Task `{}`.", snake(&other)),
    }
}

/// Share of the least frequent class, from the profile's top values.
fn minority_share(col: &ColumnProfile) -> Option<f64> {
    let total: u64 = col.top_values.iter().map(|v| v.count).sum();
    // Only meaningful when every class is listed in the top values.
    if total == 0 || (col.top_values.len() as u64) < col.n_unique {
        return None;
    }
    let min = col.top_values.iter().map(|v| v.count).min()?;
    Some(min as f64 / total as f64)
}

fn recommend_metric(task: TaskType, col: &ColumnProfile, minority: Option<f64>) -> (Metric, String) {
    match task {
        TaskType::BinaryClassification => match minority {
            Some(m) if m < 0.2 => (
                Metric::PrAuc,
                format!(
                    "Positive class is rare ({:.1}%) → PR-AUC, which is not inflated by the many easy negatives the way ROC-AUC is.",
                    m * 100.0
                ),
            ),
            Some(_) => (
                Metric::RocAuc,
                "Classes are reasonably balanced → ROC-AUC (threshold-independent ranking quality)."
                    .to_owned(),
            ),
            // Never claim balance that was not measured (#86).
            None => (
                Metric::RocAuc,
                "Class balance could not be measured → ROC-AUC; check the class counts in the profile."
                    .to_owned(),
            ),
        },
        TaskType::MulticlassClassification => match minority {
            Some(m) if m < 0.1 => (
                Metric::F1,
                format!(
                    "Some classes are rare ({:.1}%) → macro F1, so every class counts equally.",
                    m * 100.0
                ),
            ),
            Some(_) => (
                Metric::BalancedAccuracy,
                "Multiclass → balanced accuracy.".to_owned(),
            ),
            None => (
                Metric::BalancedAccuracy,
                format!(
                    "Multiclass with more than {} classes: class balance not measured → balanced accuracy, which weighs every class equally.",
                    crate::profile::ALL_VALUES_UP_TO
                ),
            ),
        },
        TaskType::Regression => {
            let heavy_tail = col.numeric.as_ref().is_some_and(|n| {
                let spread = (n.p75 - n.p25).abs().max(f64::EPSILON);
                (n.p99 - n.p50).abs() / spread > 10.0
            });
            if heavy_tail {
                (
                    Metric::Mae,
                    "Target has a heavy tail (extreme values) → MAE, which is less dominated by outliers than RMSE.".to_owned(),
                )
            } else {
                (
                    Metric::Rmse,
                    "Regression → RMSE (same unit as the target, penalizes large errors).".to_owned(),
                )
            }
        }
        TaskType::Clustering => (Metric::Silhouette, "Clustering → silhouette.".to_owned()),
        TaskType::LlmFineTune => (Metric::Perplexity, "LLM fine-tuning → perplexity.".to_owned()),
        TaskType::ImageClassification | TaskType::TextClassification => {
            (Metric::F1, "Classification → macro F1.".to_owned())
        }
        TaskType::TimeSeriesForecast => (Metric::Mase, "Forecasting → MASE (scale-free).".to_owned()),
        TaskType::AnomalyDetection => (Metric::PrAuc, "Anomaly detection → PR-AUC.".to_owned()),
    }
}

fn snake<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::profile::{NumericStats, ValueCount};

    fn col(name: &str, dtype: &str, kind: ColumnKind, n_unique: u64, counts: &[u64]) -> ColumnProfile {
        ColumnProfile {
            name: name.into(),
            dtype: dtype.into(),
            kind,
            null_count: 0,
            null_fraction: 0.0,
            n_unique,
            unique_ratio: 0.0,
            numeric: None,
            top_values: counts
                .iter()
                .enumerate()
                .map(|(i, &c)| ValueCount {
                    value: format!("v{i}"),
                    count: c,
                })
                .collect(),
            mean_length: None,
            min_time: None,
            max_time: None,
        }
    }

    fn run(c: ColumnProfile) -> TaskInference {
        let spec = JobSpec::from_yaml(&format!(
            "version: 1\nname: t\ndata: {{ source: x.csv, target: {} }}\n",
            c.name
        ))
        .unwrap();
        let profile = Profile {
            version: 1,
            dataset_id: "x".into(),
            rows: 100,
            columns: vec![c],
            sample: vec![],
            sampled_rows: None,
        };
        infer_task(&spec, &profile).unwrap()
    }

    #[test]
    fn binary_balanced_and_imbalanced() {
        let r = run(col("y", "str", ColumnKind::Categorical, 2, &[60, 40]));
        assert_eq!(
            (r.task, r.metric),
            (TaskType::BinaryClassification, Metric::RocAuc)
        );
        let r = run(col("y", "bool", ColumnKind::Boolean, 2, &[97, 3]));
        assert_eq!(
            (r.task, r.metric),
            (TaskType::BinaryClassification, Metric::PrAuc)
        );
        assert!((r.minority_share.unwrap() - 0.03).abs() < 1e-9);
    }

    #[test]
    fn multiclass_and_regression() {
        let r = run(col("y", "str", ColumnKind::Categorical, 3, &[50, 45, 5]));
        assert_eq!(
            (r.task, r.metric),
            (TaskType::MulticlassClassification, Metric::F1)
        );
        let r = run(col("y", "i64", ColumnKind::Numeric, 5, &[20, 20, 20, 20, 20]));
        assert_eq!(r.task, TaskType::MulticlassClassification);
        let mut c = col("y", "f64", ColumnKind::Numeric, 900, &[]);
        c.numeric = Some(NumericStats {
            min: 0.0,
            max: 10.0,
            mean: 5.0,
            std: 2.0,
            p01: 0.1,
            p25: 3.0,
            p50: 5.0,
            p75: 7.0,
            p99: 9.9,
            histogram: vec![],
        });
        let r = run(c.clone());
        assert_eq!((r.task, r.metric), (TaskType::Regression, Metric::Rmse));
        c.numeric.as_mut().unwrap().p99 = 500.0;
        assert_eq!(run(c).metric, Metric::Mae);
    }

    #[test]
    fn explicit_spec_wins() {
        let spec = JobSpec::from_yaml(
            "version: 1\nname: t\ndata: { source: x.csv, target: y }\ntask: { type: regression, metric: mae }\n",
        )
        .unwrap();
        let profile = Profile {
            version: 1,
            dataset_id: "x".into(),
            rows: 10,
            columns: vec![col("y", "i64", ColumnKind::Numeric, 3, &[3, 3, 4])],
            sample: vec![],
            sampled_rows: None,
        };
        let r = infer_task(&spec, &profile).unwrap();
        assert_eq!((r.task, r.metric), (TaskType::Regression, Metric::Mae));
    }
}

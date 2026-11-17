//! Which algorithms to try, and on which backend.
//!
//! Default ladder (docs/wiki/Algorithm-Catalog.md): trivial baseline →
//! linear model → trees/boosting. Native Rust is preferred when it has the
//! algorithm; Python covers the rest.

use tempera_core::spec::{Candidates, JobSpec, TaskType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Trivial baselines computed in-process.
    Builtin,
    Native,
    Python,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Native => "native",
            Self::Python => "python",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub algorithm: String,
    pub backend: BackendKind,
    /// Trivial floor every model must beat.
    pub is_baseline: bool,
}

const NATIVE: [&str; 4] = [
    "linear_regression",
    "ridge",
    "logistic_regression",
    "decision_tree",
];
const PYTHON: [&str; 6] = [
    "logistic_regression",
    "linear_regression",
    "ridge",
    "random_forest",
    "lightgbm",
    "xgboost",
];

fn classification(task: TaskType) -> bool {
    matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    )
}

/// Resolves the candidate list for a task.
pub fn candidates(spec: &JobSpec, task: TaskType) -> Result<Vec<Candidate>, String> {
    let trivial = if classification(task) { "majority" } else { "mean" };
    let mut out = Vec::new();
    if spec.strategy.baseline {
        out.push(Candidate {
            algorithm: trivial.into(),
            backend: BackendKind::Builtin,
            is_baseline: true,
        });
    }
    let names: Vec<String> = match &spec.strategy.candidates {
        Candidates::Auto if classification(task) => [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "lightgbm",
            "xgboost",
        ]
        .map(String::from)
        .to_vec(),
        Candidates::Auto => [
            "linear_regression",
            "ridge",
            "random_forest",
            "lightgbm",
            "xgboost",
        ]
        .map(String::from)
        .to_vec(),
        Candidates::List(list) => list.clone(),
    };
    for name in names {
        let (forced, algo) = match name.split_once(':') {
            Some((b, a)) => (Some(b.to_owned()), a.to_owned()),
            None => (None, name.clone()),
        };
        if algo == trivial {
            continue;
        }
        let backend = match forced.as_deref() {
            Some("native") if NATIVE.contains(&algo.as_str()) => BackendKind::Native,
            Some("python") if PYTHON.contains(&algo.as_str()) => BackendKind::Python,
            Some(b) => return Err(format!("`{name}`: backend `{b}` does not provide `{algo}`")),
            None if NATIVE.contains(&algo.as_str()) => BackendKind::Native,
            None if PYTHON.contains(&algo.as_str()) => BackendKind::Python,
            None => return Err(format!("unknown algorithm `{algo}`")),
        };
        if algo == "decision_tree" && !classification(task) {
            continue; // linfa-trees has no regressor (see #15)
        }
        if !out
            .iter()
            .any(|c: &Candidate| c.algorithm == algo && c.backend == backend)
        {
            out.push(Candidate {
                algorithm: algo,
                backend,
                is_baseline: false,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn spec(extra: &str) -> JobSpec {
        JobSpec::from_yaml(&format!(
            "version: 1\nname: t\ndata: {{ source: x.csv, target: y }}\n{extra}"
        ))
        .unwrap()
    }

    #[test]
    fn auto_ladder() {
        let c = candidates(&spec(""), TaskType::BinaryClassification).unwrap();
        let names: Vec<_> = c.iter().map(|c| (c.algorithm.as_str(), c.backend)).collect();
        assert_eq!(names[0], ("majority", BackendKind::Builtin));
        assert!(names.contains(&("logistic_regression", BackendKind::Native)));
        assert!(names.contains(&("lightgbm", BackendKind::Python)));
        let r = candidates(&spec(""), TaskType::Regression).unwrap();
        assert!(!r.iter().any(|c| c.algorithm == "decision_tree"));
        assert_eq!(r[0].algorithm, "mean");
    }

    #[test]
    fn explicit_and_forced_backends() {
        let c = candidates(
            &spec("strategy: { candidates: [\"python:logistic_regression\", lightgbm], baseline: false }"),
            TaskType::BinaryClassification,
        )
        .unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].backend, BackendKind::Python);
        assert!(
            candidates(
                &spec("strategy: { candidates: [catboost] }"),
                TaskType::Regression
            )
            .is_err()
        );
        assert!(
            candidates(
                &spec("strategy: { candidates: [\"native:lightgbm\"] }"),
                TaskType::Regression
            )
            .is_err()
        );
    }
}

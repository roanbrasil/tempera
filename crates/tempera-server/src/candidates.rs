//! Which algorithms to try, and on which backend.
//!
//! Default ladder (docs/wiki/Algorithm-Catalog.md): trivial baseline →
//! linear model → trees/boosting. Native Rust is preferred when it has the
//! algorithm; Python covers the rest. `strategy.candidates: [all]` tries every
//! algorithm that supports the task.

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

/// Task families an algorithm supports.
#[derive(Debug, Clone, Copy)]
struct Algo {
    name: &'static str,
    classification: bool,
    regression: bool,
    clustering: bool,
    anomaly: bool,
}

const fn sup(name: &'static str, classification: bool, regression: bool) -> Algo {
    Algo {
        name,
        classification,
        regression,
        clustering: false,
        anomaly: false,
    }
}

const NATIVE: &[Algo] = &[
    sup("linear_regression", false, true),
    sup("ridge", false, true),
    sup("logistic_regression", true, false),
    sup("decision_tree", true, false), // linfa-trees has no regressor (#15)
];

const PYTHON: &[Algo] = &[
    sup("logistic_regression", true, false),
    sup("linear_regression", false, true),
    sup("ridge", false, true),
    sup("lasso", false, true),
    sup("elastic_net", false, true),
    sup("sgd", true, true),
    sup("random_forest", true, true),
    sup("extra_trees", true, true),
    sup("hist_gradient_boosting", true, true),
    sup("adaboost", true, true),
    sup("lightgbm", true, true),
    sup("xgboost", true, true),
    sup("catboost", true, true),
    sup("knn", true, true),
    sup("svm", true, true),
    sup("naive_bayes", true, false),
    sup("mlp", true, true),
    Algo {
        name: "kmeans",
        classification: false,
        regression: false,
        clustering: true,
        anomaly: false,
    },
    Algo {
        name: "dbscan",
        classification: false,
        regression: false,
        clustering: true,
        anomaly: false,
    },
    Algo {
        name: "pca_kmeans",
        classification: false,
        regression: false,
        clustering: true,
        anomaly: false,
    },
    Algo {
        name: "isolation_forest",
        classification: false,
        regression: false,
        clustering: false,
        anomaly: true,
    },
];

fn classification(task: TaskType) -> bool {
    matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    )
}

fn supports(a: &Algo, task: TaskType) -> bool {
    match task {
        TaskType::BinaryClassification | TaskType::MulticlassClassification => a.classification,
        TaskType::Regression => a.regression,
        TaskType::Clustering => a.clustering,
        TaskType::AnomalyDetection => a.anomaly,
        _ => false,
    }
}

fn find(table: &[Algo], name: &str) -> Option<Algo> {
    table.iter().find(|a| a.name == name).copied()
}

/// One catalog row for the API: key, backend and the tasks it supports.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CatalogEntry {
    pub key: &'static str,
    pub backend: &'static str,
    pub tasks: Vec<&'static str>,
}

pub fn catalog() -> Vec<CatalogEntry> {
    let tasks = |a: &Algo| {
        let mut t = Vec::new();
        if a.classification {
            t.extend(["binary_classification", "multiclass_classification"]);
        }
        if a.regression {
            t.push("regression");
        }
        if a.clustering {
            t.push("clustering");
        }
        if a.anomaly {
            t.push("anomaly_detection");
        }
        t
    };
    let mut out = vec![
        CatalogEntry {
            key: "majority",
            backend: "builtin",
            tasks: vec!["binary_classification", "multiclass_classification"],
        },
        CatalogEntry {
            key: "mean",
            backend: "builtin",
            tasks: vec!["regression"],
        },
    ];
    out.extend(NATIVE.iter().map(|a| CatalogEntry {
        key: a.name,
        backend: "native",
        tasks: tasks(a),
    }));
    out.extend(PYTHON.iter().map(|a| CatalogEntry {
        key: a.name,
        backend: "python",
        tasks: tasks(a),
    }));
    out
}

/// Every algorithm name known to any backend (for error messages and docs).
pub fn known() -> Vec<&'static str> {
    let mut v: Vec<&str> = NATIVE.iter().chain(PYTHON).map(|a| a.name).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Resolves the candidate list for a task.
pub fn candidates(spec: &JobSpec, task: TaskType) -> Result<Vec<Candidate>, String> {
    let supervised = matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification | TaskType::Regression
    );
    let trivial = if classification(task) { "majority" } else { "mean" };
    let mut out = Vec::new();
    if spec.strategy.baseline && supervised {
        out.push(Candidate {
            algorithm: trivial.into(),
            backend: BackendKind::Builtin,
            is_baseline: true,
        });
    }
    let names: Vec<String> = match &spec.strategy.candidates {
        Candidates::List(list) if list.iter().any(|n| n == "all") => {
            let mut v: Vec<String> = NATIVE
                .iter()
                .filter(|a| supports(a, task))
                .map(|a| a.name.to_owned())
                .collect();
            v.extend(
                PYTHON
                    .iter()
                    .filter(|a| supports(a, task))
                    .map(|a| a.name.to_owned()),
            );
            v
        }
        Candidates::List(list) => list.clone(),
        Candidates::Auto => match task {
            TaskType::BinaryClassification | TaskType::MulticlassClassification => {
                vec![
                    "logistic_regression",
                    "decision_tree",
                    "random_forest",
                    "lightgbm",
                    "xgboost",
                    "catboost",
                ]
            }
            TaskType::Regression => vec![
                "linear_regression",
                "ridge",
                "random_forest",
                "lightgbm",
                "xgboost",
                "catboost",
            ],
            TaskType::Clustering => vec!["kmeans", "pca_kmeans", "dbscan"],
            TaskType::AnomalyDetection => vec!["isolation_forest"],
            other => return Err(format!("task `{other:?}` is not supported yet (see the roadmap)")),
        }
        .into_iter()
        .map(String::from)
        .collect(),
    };

    for name in names {
        let (forced, algo) = match name.split_once(':') {
            Some((b, a)) => (Some(b.to_owned()), a.to_owned()),
            None => (None, name.clone()),
        };
        if algo == trivial && supervised {
            continue;
        }
        let native = find(NATIVE, &algo);
        let python = find(PYTHON, &algo);
        let (backend, info) = match (forced.as_deref(), native, python) {
            (Some("native"), Some(a), _) => (BackendKind::Native, a),
            (Some("python"), _, Some(a)) => (BackendKind::Python, a),
            (Some(b), _, _) => return Err(format!("`{name}`: backend `{b}` does not provide `{algo}`")),
            // Prefer native only when it supports this task (e.g. no native tree regressor).
            (None, Some(a), _) if supports(&a, task) => (BackendKind::Native, a),
            (None, _, Some(a)) => (BackendKind::Python, a),
            (None, Some(a), None) => (BackendKind::Native, a),
            (None, None, None) => {
                return Err(format!(
                    "unknown algorithm `{algo}` (known: {})",
                    known().join(", ")
                ));
            }
        };
        if !supports(&info, task) {
            // Explicitly listed but wrong task: an error is clearer than a silent skip.
            if matches!(spec.strategy.candidates, Candidates::List(ref l) if !l.iter().any(|n| n == "all")) {
                return Err(format!("`{algo}` does not support task `{task:?}`"));
            }
            continue;
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
    if out.iter().all(|c| c.is_baseline) {
        return Err("no candidate algorithm supports this task".into());
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
        assert!(names.contains(&("catboost", BackendKind::Python)));
        let r = candidates(&spec(""), TaskType::Regression).unwrap();
        assert!(!r.iter().any(|c| c.algorithm == "decision_tree"));
        assert_eq!(r[0].algorithm, "mean");
    }

    #[test]
    fn all_expands_per_task() {
        let all = spec("strategy: { candidates: [all] }");
        let cls = candidates(&all, TaskType::MulticlassClassification).unwrap();
        assert!(cls.iter().any(|c| c.algorithm == "naive_bayes"));
        assert!(!cls.iter().any(|c| c.algorithm == "lasso"));
        let reg = candidates(&all, TaskType::Regression).unwrap();
        assert!(reg.iter().any(|c| c.algorithm == "lasso"));
        assert!(!reg.iter().any(|c| c.algorithm == "naive_bayes"));
        // native decision_tree is classification-only and Python has no tree regressor named so
        assert!(!reg.iter().any(|c| c.algorithm == "decision_tree"));
        // majority + 2 native + 12 Python-only classifiers (logistic dedupes to native)
        assert_eq!(cls.len(), 15, "{cls:?}");
    }

    #[test]
    fn unsupervised_tasks() {
        let s = JobSpec::from_yaml("version: 1\nname: t\ndata: { source: x.csv }\n").unwrap();
        let c = candidates(&s, TaskType::Clustering).unwrap();
        assert_eq!(
            c.iter().map(|c| c.algorithm.as_str()).collect::<Vec<_>>(),
            ["kmeans", "pca_kmeans", "dbscan"]
        );
        assert!(c.iter().all(|c| !c.is_baseline));
        let a = candidates(&s, TaskType::AnomalyDetection).unwrap();
        assert_eq!(a[0].algorithm, "isolation_forest");
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
        assert!(candidates(&spec("strategy: { candidates: [nope] }"), TaskType::Regression).is_err());
        assert!(
            candidates(
                &spec("strategy: { candidates: [\"native:lightgbm\"] }"),
                TaskType::Regression
            )
            .is_err()
        );
        assert!(
            candidates(
                &spec("strategy: { candidates: [naive_bayes] }"),
                TaskType::Regression
            )
            .is_err()
        );
    }
}

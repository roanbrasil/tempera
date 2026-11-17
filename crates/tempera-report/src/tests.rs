//! Rendering tests with realistic fixtures.

use std::collections::BTreeMap;
use std::path::PathBuf;

use tempera_audit::{AuditReport, Finding, Resolution, Severity};
use tempera_core::spec::{AuditPolicy, Metric, SplitStrategy, TaskType};
use tempera_data::eval::Evaluation;
use tempera_data::split::{Feature, FeatureKind, FeaturePlan, Split};

use super::*;

fn finding(id: &str, sev: Severity, title: &str, resolution: Resolution) -> Finding {
    Finding {
        id: id.into(),
        check: id.split(':').next().unwrap_or(id).into(),
        severity: sev,
        column: id.split_once(':').map(|(_, c)| c.to_owned()),
        title: title.into(),
        evidence: format!("Evidence for {title} (value < 0.5 & \"quoted\")."),
        suggestion: format!("Do something about {title}."),
        resolution,
    }
}

fn run(alg: &str, backend: &str, baseline: bool, status: &str, score: Option<f64>, dur: f64) -> RunSummary {
    let mut valid_metrics = BTreeMap::new();
    if let Some(s) = score {
        valid_metrics.insert("roc_auc".to_owned(), s);
        valid_metrics.insert("pr_auc".to_owned(), s - 0.2);
        valid_metrics.insert("f1".to_owned(), s - 0.15);
        valid_metrics.insert("accuracy".to_owned(), s - 0.05);
    }
    RunSummary {
        run_id: format!("run-{alg}"),
        algorithm: alg.into(),
        backend: backend.into(),
        is_baseline: baseline,
        status: status.into(),
        valid_score: score,
        valid_metrics,
        duration_s: dur,
        error: (status == "failed")
            .then(|| "python worker exited with code 137 (out of memory) <oom>".to_owned()),
    }
}

fn plan() -> FeaturePlan {
    let mut dropped = BTreeMap::new();
    dropped.insert("customer_id".to_owned(), "identifier: unique per row".to_owned());
    dropped.insert(
        "chargeback_amount".to_owned(),
        "leaks the target (AUC 0.99 alone)".to_owned(),
    );
    FeaturePlan {
        features: vec![
            Feature {
                name: "age".into(),
                kind: FeatureKind::Numeric,
            },
            Feature {
                name: "tenure_months".into(),
                kind: FeatureKind::Numeric,
            },
            Feature {
                name: "plan".into(),
                kind: FeatureKind::Categorical,
            },
            Feature {
                name: "has_partner".into(),
                kind: FeatureKind::Boolean,
            },
        ],
        datetime: vec!["signup_date".into()],
        dropped,
        dedupe: true,
    }
}

fn audit(findings: Vec<Finding>, blocked: bool) -> AuditReport {
    AuditReport {
        dataset_id: "ds-abc123".into(),
        policy: AuditPolicy::BlockOnRed,
        findings,
        plan: plan(),
        blocked,
    }
}

fn split(task: TaskType, classes: &[&str]) -> Split {
    let mut rows = BTreeMap::new();
    rows.insert("train".to_owned(), 49_000);
    rows.insert("valid".to_owned(), 10_500);
    rows.insert("test".to_owned(), 10_500);
    let mut class_counts = BTreeMap::new();
    if !classes.is_empty() {
        for (p, n) in [("train", 49_000u64), ("valid", 10_500), ("test", 10_500)] {
            let mut m = BTreeMap::new();
            m.insert(classes[0].to_owned(), n * 73 / 100);
            m.insert(classes[1].to_owned(), n - n * 73 / 100);
            class_counts.insert(p.to_owned(), m);
        }
    }
    Split {
        id: "split-1".into(),
        version: 1,
        dataset_id: "ds-abc123".into(),
        strategy: if classes.is_empty() {
            SplitStrategy::Random
        } else {
            SplitStrategy::Stratified
        },
        seed: 42,
        task,
        features: plan().features,
        classes: classes.iter().map(|c| (*c).to_owned()).collect(),
        positive_class: classes.get(1).map(|c| (*c).to_owned()),
        rows,
        class_counts,
        dropped_null_target: 12,
        dropped_duplicates: 1_234,
        dir: PathBuf::from("/tmp/split"),
    }
}

fn spec_yaml() -> String {
    "version: 1\nname: churn\ndata:\n  source: data/churn.csv\n  target: churned\nsplit:\n  strategy: auto\n  seed: 42\n".into()
}

pub fn binary_fixture() -> ReportData {
    let roc: Vec<(f64, f64)> = (0..=50)
        .map(|i| {
            let x = i as f64 / 50.0;
            (x, 1.0 - (1.0 - x).powf(4.0))
        })
        .collect();
    let pr: Vec<(f64, f64)> = (0..=50)
        .map(|i| {
            let r = i as f64 / 50.0;
            (r, (0.95 - 0.6 * r.powf(2.5)).max(0.27))
        })
        .collect();
    let mut metrics = BTreeMap::new();
    metrics.insert("roc_auc".to_owned(), 0.8712);
    metrics.insert("pr_auc".to_owned(), 0.6634);
    metrics.insert("f1".to_owned(), 0.6123);
    metrics.insert("accuracy".to_owned(), 0.8231);
    metrics.insert("log_loss".to_owned(), 0.3921);
    let importance: Vec<(String, f64)> = (0..26)
        .map(|i| {
            (
                format!("feature_with_a_long_name_{i:02}"),
                1000.0 / (i as f64 + 1.0),
            )
        })
        .chain([("nan_feature".to_owned(), f64::NAN)])
        .collect();
    ReportData {
        job_id: "job-0001".into(),
        job_name: "Customer churn (Q3)".into(),
        created_at: "2026-11-16T14:03:12Z".into(),
        tempera_version: "0.1.0".into(),
        dataset: DatasetSummary {
            id: "ds-abc123".into(),
            source: "data/churn.csv".into(),
            rows: 71_246,
            columns: 21,
        },
        task: TaskType::BinaryClassification,
        metric: Metric::RocAuc,
        task_reasons: vec![
            "Target `churned` has exactly 2 distinct values.".into(),
            "Minority class is 26.5%: ROC-AUC ranks well regardless of threshold.".into(),
        ],
        audit: audit(
            vec![
                finding(
                    "A1:chargeback_amount",
                    Severity::Red,
                    "Target leakage",
                    Resolution::Acknowledged,
                ),
                finding(
                    "A4:signup_date",
                    Severity::Yellow,
                    "Date column, non-temporal split",
                    Resolution::Unresolved,
                ),
                finding(
                    "A5",
                    Severity::Yellow,
                    "Imbalanced classes",
                    Resolution::Auto {
                        action: "stratified split; metric `roc_auc`".into(),
                    },
                ),
                finding(
                    "A10:customer_id",
                    Severity::Green,
                    "Identifier column",
                    Resolution::Auto {
                        action: "column dropped".into(),
                    },
                ),
            ],
            false,
        ),
        split: split(TaskType::BinaryClassification, &["no", "yes"]),
        leaderboard: vec![
            run("lightgbm", "python", false, "succeeded", Some(0.8734), 83.4),
            run(
                "logistic_regression",
                "native",
                false,
                "succeeded",
                Some(0.8411),
                4.2,
            ),
            run("majority_class", "builtin", true, "succeeded", Some(0.5), 0.05),
            run("catboost", "python", false, "failed", None, 312.0),
        ],
        winner: Some(Winner {
            run_id: "run-lightgbm".into(),
            algorithm: "lightgbm".into(),
            test: Evaluation {
                rows: 10_500,
                metrics,
                confusion: vec![vec![7_012, 653], vec![1_204, 1_631]],
                classes: vec!["no".into(), "yes".into()],
                roc,
                pr: pr.into_iter().chain([(f64::NAN, 0.5)]).collect(),
                residuals: Vec::new(),
            },
            importance,
            artifacts: vec![
                ("onnx".into(), "runs/job-0001/model.onnx".into()),
                ("lightgbm".into(), "runs/job-0001/model.txt".into()),
            ],
        }),
        cost: CostSummary {
            wall_time_s: 431.7,
            machine: "local (8 CPU, no GPU)".into(),
            cost_usd: 0.0,
        },
        spec_yaml: spec_yaml(),
        reproduce_command: "tempera run tempera.yaml --seed 42".into(),
    }
}

fn regression_fixture() -> ReportData {
    let mut d = binary_fixture();
    d.job_name = "House prices".into();
    d.task = TaskType::Regression;
    d.metric = Metric::Rmse;
    d.split = split(TaskType::Regression, &[]);
    d.leaderboard = vec![
        RunSummary {
            valid_metrics: [("rmse".to_owned(), 31_250.5), ("r2".to_owned(), 0.81)]
                .into_iter()
                .collect(),
            ..run("lightgbm", "python", false, "succeeded", Some(31_250.5), 120.0)
        },
        RunSummary {
            valid_metrics: [("rmse".to_owned(), 78_100.0), ("r2".to_owned(), 0.0)]
                .into_iter()
                .collect(),
            ..run("mean", "builtin", true, "succeeded", Some(78_100.0), 0.1)
        },
    ];
    let residuals: Vec<(f64, f64)> = (0..200)
        .map(|i| {
            let truth = 100_000.0 + i as f64 * 2_500.0;
            let noise = ((i * 37 % 23) as f64 - 11.0) * 4_000.0;
            (truth, truth + noise)
        })
        .chain([(f64::INFINITY, 1.0), (2.0, f64::NAN)])
        .collect();
    let mut metrics = BTreeMap::new();
    metrics.insert("rmse".to_owned(), 32_004.25);
    metrics.insert("mae".to_owned(), 21_877.0);
    metrics.insert("r2".to_owned(), 0.8012);
    metrics.insert("mape".to_owned(), f64::NAN);
    d.winner = Some(Winner {
        run_id: "run-lightgbm".into(),
        algorithm: "lightgbm".into(),
        test: Evaluation {
            rows: 10_500,
            metrics,
            confusion: Vec::new(),
            classes: Vec::new(),
            roc: Vec::new(),
            pr: Vec::new(),
            residuals,
        },
        importance: vec![
            ("sqft".into(), 0.42),
            ("zip".into(), 0.21),
            ("rooms".into(), -0.05),
        ],
        artifacts: Vec::new(),
    });
    d.cost.cost_usd = 0.37;
    d
}

fn no_winner_fixture() -> ReportData {
    let mut d = binary_fixture();
    d.job_name = "Everything failed".into();
    d.leaderboard = vec![
        run("lightgbm", "python", false, "failed", None, 3.0),
        run("majority_class", "builtin", true, "failed", None, 0.01),
    ];
    d.winner = None;
    d
}

const HEADINGS: [&str; 8] = [
    "Verdict",
    "Leaderboard",
    "Test-set evaluation",
    "Feature importance",
    "Data audit",
    "Split",
    "Reproduce",
    "tempera.yaml",
];

fn check_common(html: &str) {
    assert!(html.starts_with("<!doctype html>"), "doctype");
    assert!(html.trim_end().ends_with("</html>"));
    for h in HEADINGS {
        assert!(html.contains(h), "missing heading {h}");
    }
    for token in html.split(|c: char| !c.is_ascii_alphanumeric()) {
        assert!(
            !matches!(token, "NaN" | "nan" | "inf" | "Infinity" | "infinity"),
            "non-finite token `{token}` in output"
        );
    }
    assert!(
        !html.contains("http://") && !html.contains("https://"),
        "external URL in output"
    );
    assert!(!html.contains("<script"), "script tag in output");
    assert_eq!(
        html.matches("<svg").count(),
        html.matches("</svg>").count(),
        "unbalanced svg"
    );
    assert_eq!(
        html.matches("<section").count(),
        html.matches("</section>").count(),
        "unbalanced section"
    );
    assert_eq!(
        html.matches("<table").count(),
        html.matches("</table>").count(),
        "unbalanced table"
    );
    for svg in html.split("<svg").skip(1) {
        let body = svg.split("</svg>").next().unwrap_or("");
        assert!(body.contains("<title>"), "svg without accessible title");
    }
}

#[test]
fn binary_report() {
    let html = render(&binary_fixture());
    check_common(&html);
    // ROC, PR, confusion matrix, importance.
    assert_eq!(html.matches("<svg").count(), 4);
    assert!(
        html.contains("+0.373 ROC-AUC over the majority-class baseline"),
        "verdict gain phrase"
    );
    assert!(html.contains("0.8712"), "test score");
    assert!(html.contains("US$ 0.00 (local)"));
    assert!(html.contains("7m 12s"));
    assert!(html.contains("71,246 rows × 21 columns"));
    assert!(html.contains("7,012"));
    assert!(html.contains("Blocking") && html.contains("Warning") && html.contains("OK"));
    assert!(html.contains("Acknowledged") && html.contains("Auto action") && html.contains("Unresolved"));
    assert!(html.contains("baseline</span>"));
    assert!(html.contains("out of memory) &lt;oom&gt;"));
    assert!(html.contains("tempera run tempera.yaml --seed 42"));
    assert!(html.contains("runs/job-0001/model.onnx"));
    assert!(html.contains("Top 20 of 27 features"));
    assert!(html.contains("held-out"));
    assert!(html.contains("1,234"), "duplicates removed");
    assert!(
        html.contains("&lt; 0.5 &amp; &quot;quoted&quot;"),
        "evidence escaped"
    );
}

#[test]
fn regression_report() {
    let html = render(&regression_fixture());
    check_common(&html);
    assert!(html.contains("Predicted vs actual"));
    assert!(html.contains("y = x"));
    assert!(html.contains("<circle"));
    assert!(!html.contains("Confusion matrix"));
    assert!(html.contains("lower is better"));
    assert!(html.contains("+46,849.5"), "rmse improvement over mean baseline");
    assert!(html.contains("US$ 0.37"));
}

#[test]
fn no_winner_report() {
    let html = render(&no_winner_fixture());
    check_common(&html);
    assert!(html.contains("No winner"));
    assert!(html.contains("All 2 runs failed"));
    assert_eq!(html.matches("<svg").count(), 0);
}

#[test]
fn empty_report_does_not_panic() {
    let mut d = no_winner_fixture();
    d.leaderboard.clear();
    d.task_reasons.clear();
    d.audit.findings.clear();
    d.split.rows.clear();
    d.split.class_counts.clear();
    d.spec_yaml.clear();
    d.cost.wall_time_s = f64::NAN;
    d.cost.cost_usd = f64::INFINITY;
    let html = render(&d);
    check_common(&html);
    assert!(html.contains("No candidate was trained."));
}

#[test]
fn escapes_malicious_strings() {
    let mut d = binary_fixture();
    d.job_name = "<script>alert(1)</script>".into();
    d.leaderboard[0].algorithm = "<img src=x onerror=alert(2)>".into();
    d.spec_yaml = "name: \"</pre><script>x</script>\"".into();
    if let Some(w) = d.winner.as_mut() {
        w.importance[0].0 = "<b>evil</b>".into();
        w.test.classes[1] = "<yes>".into();
    }
    let html = render(&d);
    assert!(!html.contains("<script"));
    assert!(!html.contains("<img"));
    assert!(!html.contains("<b>evil"));
    assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
}

#[test]
fn write_preview() {
    let html = render(&binary_fixture());
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    let _ = std::fs::create_dir_all(&target);
    let _ = std::fs::write(target.join("report-preview.html"), &html);
    let _ = std::fs::write(
        target.join("report-preview-regression.html"),
        render(&regression_fixture()),
    );
    let _ = std::fs::write(
        target.join("report-preview-no-winner.html"),
        render(&no_winner_fixture()),
    );
}

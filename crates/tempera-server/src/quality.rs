//! Data-quality steps that run on the Python worker (#32):
//! A7 label issues (cleanlab) before training, A8 group fairness (fairlearn)
//! of the winner's test predictions. Both become audit findings, so the CLI,
//! the UI and the report show them without special cases. Failures here never
//! fail the job: the step is reported as skipped.

use std::path::{Path, PathBuf};

use polars::prelude::*;
use serde_json::{Value, json};
use tempera_audit::{Finding, Resolution, Severity};
use tempera_core::spec::TaskType;
use tempera_data::ingest::scan_parquet;
use tempera_data::split::{Partition, SENSITIVE_PREFIX, Split};
use tempera_exec::Hub;
use tempera_proto::v1::Command as JobCommand;
use tempera_track::Tracker;

use crate::candidates::BackendKind;
use crate::orchestrator::dispatch;
use crate::ui;

fn abs(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn features_json(split: &Split) -> Value {
    json!(
        split
            .features
            .iter()
            .map(|f| json!({"name": f.name, "kind": f.kind}))
            .collect::<Vec<_>>()
    )
}

/// A7: suspected label errors in the training partition (classification only).
pub async fn label_issues(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    job_dir: &Path,
    split: &Split,
) -> Option<Finding> {
    if !matches!(
        split.task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    ) {
        return None;
    }
    let out = abs(&job_dir.join("label_issues.json"));
    let manifest = json!({
        "task": split.task,
        "classes": split.classes,
        "features": features_json(split),
        "data": abs(&split.path(Partition::Train)),
        "output": out,
        "folds": 5,
        "seed": split.seed,
        "max_rows": 50_000,
        "top_n": 50,
    });
    let run_id = format!("{job_id}-quality");
    if let Err(e) = dispatch(
        tracker,
        hub,
        job_id,
        &run_id,
        BackendKind::Python,
        JobCommand::Quality,
        manifest.to_string(),
        false,
        None,
    )
    .await
    {
        ui::detail(&format!("label-issue check skipped: {e}"));
        return None;
    }
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&out).ok()?).ok()?;
    let found = v["issues_found"].as_u64().unwrap_or(0);
    let checked = v["rows_checked"].as_u64().unwrap_or(0);
    let rate = v["estimated_noise_rate"].as_f64().unwrap_or(0.0);
    if found == 0 || rate < 0.01 {
        return None;
    }
    let examples: Vec<String> = v["top"]
        .as_array()
        .map(|a| {
            a.iter()
                .take(5)
                .map(|r| {
                    format!(
                        "row {} labeled `{}` looks like `{}`",
                        r["row"],
                        r["given"].as_str().unwrap_or("?"),
                        r["suggested"].as_str().unwrap_or("?")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Some(Finding {
        id: "A7".into(),
        check: "A7".into(),
        severity: Severity::Yellow,
        column: None,
        title: "Possible label errors".into(),
        evidence: format!(
            "{found} of {checked} training rows look mislabeled (estimated noise {:.1}%). E.g. {}.",
            rate * 100.0,
            examples.join("; ")
        ),
        suggestion: format!(
            "Review the suspects in {}. A high rate on an inherently uncertain target (churn, fraud) mostly means ambiguous rows, not typos; a few confident flips usually are data errors. Rows are never removed automatically.",
            out.display()
        ),
        resolution: Resolution::Auto {
            action: "reported for review".into(),
        },
    })
}

/// A8: fairness of the winner's test predictions per sensitive attribute.
pub async fn fairness(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    job_dir: &Path,
    split: &Split,
    test_pred: &Path,
) -> Vec<Finding> {
    let test = split.path(Partition::Test);
    let Ok(schema) = scan_parquet(&test).and_then(|lf| Ok(lf.clone().collect_schema()?)) else {
        return Vec::new();
    };
    let sens: Vec<String> = schema
        .iter_names()
        .filter_map(|n| n.strip_prefix(SENSITIVE_PREFIX).map(str::to_owned))
        .collect();
    if sens.is_empty() || !test_pred.is_file() {
        return Vec::new();
    }
    let sens_path = abs(&job_dir.join("sensitive_test.parquet"));
    let write = || -> anyhow::Result<()> {
        let exprs: Vec<Expr> = sens
            .iter()
            .map(|s| col(format!("{SENSITIVE_PREFIX}{s}").as_str()).alias(s.as_str()))
            .collect();
        let mut df = scan_parquet(&test)?.select(exprs).collect()?;
        ParquetWriter::new(std::fs::File::create(&sens_path)?).finish(&mut df)?;
        Ok(())
    };
    if let Err(e) = write() {
        ui::detail(&format!("fairness check skipped: {e}"));
        return Vec::new();
    }
    let out = abs(&job_dir.join("fairness.json"));
    let manifest = json!({
        "task": split.task,
        "classes": split.classes,
        "positive_class": split.positive_class,
        "truth": abs(&test),
        "predictions": abs(test_pred),
        "sensitive": sens.iter().map(|s| (s.clone(), json!(sens_path))).collect::<serde_json::Map<_, _>>(),
        "output": out,
    });
    let run_id = format!("{job_id}-fairness");
    if let Err(e) = dispatch(
        tracker,
        hub,
        job_id,
        &run_id,
        BackendKind::Python,
        JobCommand::Fairness,
        manifest.to_string(),
        false,
        None,
    )
    .await
    {
        ui::detail(&format!("fairness check skipped: {e}"));
        return Vec::new();
    }
    let Ok(v) =
        std::fs::read_to_string(&out).map(|t| serde_json::from_str::<Value>(&t).unwrap_or(Value::Null))
    else {
        return Vec::new();
    };
    let mut findings = Vec::new();
    for (attr, a) in v["attributes"].as_object().into_iter().flatten() {
        let summary = &a["summary"];
        let dp = summary["demographic_parity_difference"].as_f64();
        let eo = summary["equalized_odds_difference"].as_f64();
        let groups: Vec<String> = a["groups"]
            .as_array()
            .map(|g| {
                g.iter()
                    .map(|x| {
                        let m = &x["metrics"];
                        let main = m["selection_rate"]
                            .as_f64()
                            .or(m["accuracy"].as_f64())
                            .or(m["mae"].as_f64());
                        format!(
                            "{} (n={}{}{})",
                            x["group"].as_str().unwrap_or("?"),
                            x["count"],
                            main.map(|v| format!(", {v:.2}")).unwrap_or_default(),
                            if x["small"].as_bool().unwrap_or(false) {
                                ", small"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let gap = dp.unwrap_or(0.0).max(eo.unwrap_or(0.0));
        let severity = if gap >= 0.2 {
            Severity::Yellow
        } else {
            Severity::Green
        };
        findings.push(Finding {
            id: format!("A8:{attr}"),
            check: "A8".into(),
            severity,
            column: Some(attr.clone()),
            title: if severity == Severity::Yellow { "Unequal outcomes across groups".into() } else { "Fairness check".into() },
            evidence: format!(
                "On the test set by `{attr}`: {}.{}{}",
                groups.join(", "),
                dp.map(|d| format!(" Demographic parity difference {d:.2}.")).unwrap_or_default(),
                eo.map(|d| format!(" Equalized odds difference {d:.2}.")).unwrap_or_default()
            ),
            suggestion: "Large gaps deserve a look at the data and the use case; mitigation (reweighting, thresholds per group) is a decision for you and your compliance team.".into(),
            resolution: Resolution::Auto { action: format!("details in {}", out.display()) },
        });
    }
    findings
}

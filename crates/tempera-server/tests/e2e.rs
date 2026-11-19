//! End-to-end: the real `tempera` CLI + `tempera-agent` on the example datasets.
//! Python candidates run only when `TEMPERA_E2E_PYTHON=1` (they need uv + the tabular extra).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// The agent binary sits next to `tempera`; build it if this test ran alone.
fn agent_bin() -> PathBuf {
    let dir = Path::new(env!("CARGO_BIN_EXE_tempera"))
        .parent()
        .unwrap()
        .to_path_buf();
    let bin = dir.join(format!("tempera-agent{}", std::env::consts::EXE_SUFFIX));
    if !bin.is_file() {
        let ok = Command::new(env!("CARGO"))
            .args(["build", "-p", "tempera-agent"])
            .current_dir(repo())
            .status()
            .unwrap();
        assert!(ok.success());
    }
    bin
}

fn tempera(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tempera"))
        .args(args)
        .current_dir(repo().join("examples"))
        .env("TEMPERA_HOME", home)
        .env("TEMPERA_AGENT_BIN", agent_bin())
        .env("TEMPERA_NUM_THREADS", "2")
        .env("NO_COLOR", "1")
        .output()
        .unwrap()
}

fn write_spec(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    let data = repo().join("examples/data");
    std::fs::write(&p, body.replace("$DATA", &data.display().to_string())).unwrap();
    p
}

fn candidates() -> &'static str {
    if std::env::var_os("TEMPERA_E2E_PYTHON").is_some() {
        "[logistic_regression, decision_tree, lightgbm]"
    } else {
        "[logistic_regression, decision_tree]"
    }
}

#[test]
fn leak_blocks_then_drop_trains_and_reports() {
    let tmp = tempfile::tempdir().unwrap();
    let leaky = write_spec(
        tmp.path(),
        "leaky.yaml",
        "version: 1\nname: churn\ndata: { source: $DATA/churn.csv, target: churned }\n",
    );
    let out = tempera(tmp.path(), &["run", leaky.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(2), "{stdout}");
    assert!(stdout.contains("A1:refund_amount"), "{stdout}");

    let fixed = write_spec(
        tmp.path(),
        "fixed.yaml",
        &format!(
            "version: 1\nname: churn\ndata: {{ source: $DATA/churn.csv, target: churned, time_column: signup_date, drop: [refund_amount] }}\nstrategy: {{ candidates: {} }}\n",
            candidates()
        ),
    );
    let out = tempera(tmp.path(), &["run", fixed.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("evaluated once"), "{stdout}");

    let report = tempera(tmp.path(), &["report"]);
    let path = String::from_utf8_lossy(&report.stdout).trim().to_owned();
    let html = std::fs::read_to_string(&path).unwrap();
    assert!(html.starts_with("<!doctype html>") || html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("logistic_regression"));

    let runs = tempera(tmp.path(), &["runs"]);
    let listing = String::from_utf8_lossy(&runs.stdout);
    assert!(
        listing.contains("blocked") && listing.contains("succeeded"),
        "{listing}"
    );
}

#[test]
fn regression_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = write_spec(
        tmp.path(),
        "houses.yaml",
        "version: 1\nname: houses\ndata: { source: $DATA/houses.csv, target: price }\nstrategy: { candidates: [linear_regression] }\n",
    );
    let out = tempera(tmp.path(), &["run", spec.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Winner    linear_regression"), "{stdout}");
}

#[test]
fn init_writes_a_valid_spec() {
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("tempera.yaml");
    let data = repo().join("examples/data/houses.csv");
    let out = tempera(
        tmp.path(),
        &[
            "init",
            data.to_str().unwrap(),
            "--target",
            "price",
            "-o",
            out_path.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let v = tempera(tmp.path(), &["validate", out_path.to_str().unwrap()]);
    assert!(v.status.success(), "{}", String::from_utf8_lossy(&v.stderr));
}

fn python_enabled() -> bool {
    std::env::var_os("TEMPERA_E2E_PYTHON").is_some()
}

#[test]
fn clustering_end_to_end() {
    if !python_enabled() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let spec = write_spec(
        tmp.path(),
        "segments.yaml",
        "version: 1\nname: segments\ndata: { source: $DATA/segments.csv }\nstrategy: { candidates: [kmeans] }\n",
    );
    let out = tempera(tmp.path(), &["run", spec.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Winner    kmeans"), "{stdout}");
    assert!(stdout.contains("silhouette"), "{stdout}");
}

#[test]
fn anomaly_detection_end_to_end() {
    if !python_enabled() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let spec = write_spec(
        tmp.path(),
        "fraud.yaml",
        "version: 1\nname: fraud\ndata: { source: $DATA/transactions.csv, target: is_fraud }\ntask: { type: anomaly_detection, metric: pr_auc }\n",
    );
    let out = tempera(tmp.path(), &["run", spec.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Winner    isolation_forest"), "{stdout}");
}

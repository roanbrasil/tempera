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
        .env("TEMPERA_STATE_DIR", home.join("state"))
        .env("TEMPERA_AGENT_BIN", agent_bin())
        .env("TEMPERA_NUM_THREADS", "2")
        .env("NO_COLOR", "1")
        .env("TEMPERA_ADVISOR", "off")
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

fn tempera_env(home: &Path, args: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tempera"));
    c.args(args)
        .current_dir(repo().join("examples"))
        .env("TEMPERA_HOME", home)
        .env("TEMPERA_STATE_DIR", home.join("state"))
        .env("TEMPERA_AGENT_BIN", agent_bin())
        .env("TEMPERA_NUM_THREADS", "2")
        .env("NO_COLOR", "1")
        .env("TEMPERA_ADVISOR", "off");
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().unwrap()
}

fn houses_on(dir: &Path, extra: &str) -> PathBuf {
    write_spec(
        dir,
        "cloud.yaml",
        &format!(
            "version: 1\nname: houses\ndata: {{ source: $DATA/houses.csv, target: price }}\nstrategy: {{ candidates: [linear_regression, ridge] }}\n{extra}"
        ),
    )
}

/// M6 without docker: the estimate prints a comparison and the savings line,
/// and a budget below the estimate refuses the job before renting anything.
#[test]
fn cloud_estimate_and_budget_refusal_on_mock() {
    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("mock-cloud.json");
    let env = [("TEMPERA_MOCK_CLOUD_STATE", state.to_str().unwrap())];
    let spec = houses_on(tmp.path(), "compute: { allow: [mock, local] }\n");
    let out = tempera_env(tmp.path(), &["cloud", "estimate", spec.to_str().unwrap()], &env);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("estimated before renting"), "{stdout}");
    assert!(
        stdout.contains("local: $0"),
        "local wins `cheapest` when it fits: {stdout}"
    );
    let prices = tempera_env(tmp.path(), &["cloud", "prices", "mock"], &env);
    assert!(String::from_utf8_lossy(&prices.stdout).contains("mock: 4 offers, updated"));

    let poor = houses_on(
        tmp.path(),
        "compute: { allow: [mock] }\nbudget: { max_cost: { amount: 0.0001, currency: USD } }\n",
    );
    let out = tempera_env(tmp.path(), &["run", poor.to_str().unwrap()], &env);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("exceeds budget.max_cost"), "{stderr}");
    // Nothing was rented.
    let st = std::fs::read_to_string(&state).unwrap_or_default();
    assert!(!st.contains("tempera-"), "{st}");
}

/// M6 end to end: a "rented" instance (mock provider + sshd container) runs
/// the job over SSH, is terminated afterwards, and the report carries the
/// estimated/actual cost and the savings line. Opt-in like the SSH tests.
#[test]
fn cloud_job_end_to_end_on_mock_docker() {
    if std::env::var_os("TEMPERA_SSH_TESTS").is_none()
        || !Command::new("docker")
            .arg("info")
            .output()
            .is_ok_and(|o| o.status.success())
    {
        eprintln!("skipping: needs TEMPERA_SSH_TESTS=1 and docker");
        return;
    }
    let image = "tempera-ssh-test:latest";
    let ctx = repo().join("crates/tempera-exec/tests/ssh");
    let ok = Command::new("docker")
        .args(["build", "-q", "-t", image])
        .arg(&ctx)
        .output()
        .unwrap();
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stderr));

    let tmp = tempfile::tempdir().unwrap();
    let state = tmp.path().join("mock-cloud.json");
    let env = [
        ("TEMPERA_MOCK_CLOUD_STATE", state.to_str().unwrap()),
        ("TEMPERA_MOCK_DOCKER_IMAGE", image),
    ];
    let spec = houses_on(
        tmp.path(),
        "compute: { allow: [mock], spot: false }\nbudget: { max_cost: { amount: 1.0, currency: USD } }\n",
    );
    let out = tempera_env(tmp.path(), &["run", spec.to_str().unwrap()], &env);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("Renting   mock:"), "{stdout}");
    assert!(stdout.contains("(cloud)"), "{stdout}");
    assert!(stdout.contains("terminated mock:"), "{stdout}");
    assert!(stdout.contains("Savings vs. naive run"), "{stdout}");

    // The instance is gone, its container removed, its lease closed.
    let st: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&state).unwrap()).unwrap();
    let insts = st["instances"].as_array().unwrap();
    assert_eq!(insts.len(), 1);
    assert_eq!(insts[0]["state"], "gone");
    assert!(insts[0]["container"].is_null());
    // Rental ledger (per-user state dir, one file per rental): nothing left open.
    let open: Vec<_> = std::fs::read_dir(tmp.path().join("state/rentals"))
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    assert!(open.is_empty(), "{open:?}");
    // The instance carried this installation's owner id.
    let owner = std::fs::read_to_string(tmp.path().join("state/owner")).unwrap();
    assert!(insts[0]["name"].as_str().unwrap().starts_with(&format!("tempera-{}-", owner.trim())));

    let report = std::fs::read_dir(tmp.path().join("jobs"))
        .unwrap()
        .flatten()
        .map(|e| e.path().join("report.json"))
        .find(|p| p.is_file())
        .unwrap();
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(report).unwrap()).unwrap();
    assert!(r["cost"]["estimated_cost_usd"].as_f64().unwrap() > 0.0);
    assert!(r["cost"]["cost_usd"].as_f64().unwrap() > 0.0);
    assert!(
        r["cost"]["savings"]
            .as_str()
            .unwrap()
            .starts_with("Savings vs. naive run")
    );
    assert!(r["cost"]["machine"].as_str().unwrap().contains("(cloud)"));
}

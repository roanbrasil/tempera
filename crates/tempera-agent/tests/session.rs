//! Real agent binary ↔ real hub over gRPC.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use polars::prelude::*;
use tempera_exec::Hub;
use tempera_proto::v1::{AgentMessage, Backend, Command as JobCommand, StartJob, agent_message};

async fn start_agent() -> (Hub, tokio::process::Child) {
    let hub = Hub::new();
    let (addr, _server) = hub.serve("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_tempera-agent"))
        .args([
            "connect",
            "--server",
            &format!("http://{addr}"),
            "--machine-id",
            "t1",
        ])
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let hw = hub.wait_for("t1", Duration::from_secs(20)).await.unwrap();
    assert!(hw.cpu_cores >= 1);
    (hub, child)
}

async fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<AgentMessage>) -> Vec<agent_message::Msg> {
    let mut out = Vec::new();
    while let Ok(Some(m)) = tokio::time::timeout(Duration::from_secs(60), rx.recv()).await {
        if let Some(m) = m.msg {
            out.push(m);
        }
    }
    out
}

fn start(run: &str, manifest: String) -> StartJob {
    StartJob {
        job_id: "job".into(),
        run_id: run.into(),
        backend: Backend::Native as i32,
        command: JobCommand::Fit as i32,
        manifest_json: manifest,
        ..Default::default()
    }
}

#[tokio::test]
async fn native_fit_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y: Vec<&str> = x.iter().map(|v| if *v > 10.0 { "yes" } else { "no" }).collect();
    let train = tmp.path().join("train.parquet");
    ParquetWriter::new(std::fs::File::create(&train).unwrap())
        .finish(&mut df!("x" => &x, "__target__" => &y).unwrap())
        .unwrap();
    let work = tmp.path().join("run");
    let manifest = serde_json::json!({
        "algorithm": "logistic_regression", "task": "binary_classification",
        "classes": ["no", "yes"], "positive_class": "yes",
        "features": [{"name": "x", "kind": "numeric"}], "params": {}, "seed": 1,
        "train": train, "predict": [{"input": train, "output": work.join("pred.parquet")}],
        "work_dir": work, "export": ["native"]
    });

    let (hub, _agent) = start_agent().await;
    let rx = hub.start("t1", start("r1", manifest.to_string())).await.unwrap();
    let msgs = collect(rx).await;
    let last = msgs.last().unwrap();
    let agent_message::Msg::Finished(done) = last else {
        panic!("expected JobFinished, got {last:?}")
    };
    assert_eq!(done.run_id, "r1");
    assert!(done.artifacts.iter().any(|a| a.kind == "model"));
    assert!(work.join("pred.parquet").is_file());
}

#[tokio::test]
async fn bad_manifest_fails_with_exit_code_2() {
    let (hub, _agent) = start_agent().await;
    let rx = hub.start("t1", start("r2", "{not json".into())).await.unwrap();
    let msgs = collect(rx).await;
    let agent_message::Msg::Failed(f) = msgs.last().unwrap() else {
        panic!("expected JobFailed")
    };
    assert_eq!(f.exit_code, Some(2));
}

#[tokio::test]
async fn disconnect_fails_running_jobs() {
    let (hub, mut agent) = start_agent().await;
    // A job that never ends: python backend pointed at `sleep` via TEMPERA_PYTHON is overkill;
    // instead kill the agent right after starting a run.
    let rx = hub.start("t1", start("r3", "{}".into())).await.unwrap();
    agent.kill().await.unwrap();
    let msgs = collect(rx).await;
    assert!(matches!(msgs.last(), Some(agent_message::Msg::Failed(_))));
}

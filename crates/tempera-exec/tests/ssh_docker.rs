//! SSH executor against a real sshd in Docker (tests/ssh/Dockerfile).
//!
//! Opt-in: runs only when `TEMPERA_SSH_TESTS` is set and `docker` works;
//! otherwise it prints why and passes. Needs the agent binary
//! (`cargo build -p tempera-agent`, or `TEMPERA_AGENT_BIN`). Uses a throwaway
//! ed25519 key and known_hosts in a temp dir; never touches `~/.ssh`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use std::time::Duration;

use polars::prelude::*;
use tempera_exec::{Hub, SshMachine, StartOpts};
use tempera_proto::v1::{AgentMessage, Backend, Command as JobCommand, StartJob, agent_message};

const IMAGE: &str = "tempera-ssh-test:latest";
const MACHINE: &str = "docker-ssh";

/// Removes the container even when the test panics.
struct Container(String);

impl Drop for Container {
    fn drop(&mut self) {
        let _ = StdCommand::new("docker").args(["rm", "-f", &self.0]).output();
    }
}

fn skip_reason() -> Option<String> {
    if std::env::var_os("TEMPERA_SSH_TESTS").is_none() {
        return Some("TEMPERA_SSH_TESTS is not set".into());
    }
    match StdCommand::new("docker").arg("info").output() {
        Ok(o) if o.status.success() => None,
        _ => Some("docker is not available".into()),
    }
}

fn agent_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("TEMPERA_AGENT_BIN") {
        return PathBuf::from(p);
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"));
    let bin = target.join("debug/tempera-agent");
    assert!(
        bin.is_file(),
        "{} not found: run `cargo build -p tempera-agent` or set TEMPERA_AGENT_BIN",
        bin.display()
    );
    bin
}

fn run(cmd: &mut StdCommand) -> String {
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "{cmd:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

async fn collect(mut rx: tokio::sync::mpsc::UnboundedReceiver<AgentMessage>) -> Vec<agent_message::Msg> {
    let mut out = Vec::new();
    while let Ok(Some(m)) = tokio::time::timeout(Duration::from_secs(120), rx.recv()).await {
        if let Some(m) = m.msg {
            out.push(m);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn ssh_executor_end_to_end() {
    if let Some(why) = skip_reason() {
        eprintln!("skipping SSH integration test: {why}");
        return;
    }
    let agent_bin = agent_binary();
    let tmp = tempfile::tempdir().unwrap();
    let key = tmp.path().join("id_ed25519");
    run(StdCommand::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-C", "tempera-test", "-f"])
        .arg(&key));
    let pubkey = std::fs::read_to_string(key.with_extension("pub")).unwrap();

    let ctx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ssh");
    run(StdCommand::new("docker")
        .args(["build", "-q", "-t", IMAGE])
        .arg(&ctx));
    let id = run(StdCommand::new("docker")
        .args(["run", "-d", "-p", "127.0.0.1::22", "-e"])
        .arg(format!("AUTHORIZED_KEY={}", pubkey.trim()))
        .arg(IMAGE));
    let _container = Container(id.clone());
    let mapped = run(StdCommand::new("docker").args(["port", &id, "22/tcp"]));
    let port: u16 = mapped
        .lines()
        .next()
        .and_then(|l| l.rsplit(':').next())
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("unexpected `docker port` output {mapped:?}"));

    let mut m = SshMachine::new(MACHINE, "127.0.0.1");
    m.user = Some("tempera".into());
    m.port = Some(port);
    m.identity = Some(key.clone());
    m.ssh_options = vec![
        format!("UserKnownHostsFile={}", tmp.path().join("known_hosts").display()),
        "StrictHostKeyChecking=accept-new".into(),
        "IdentityAgent=none".into(),
    ];

    // sshd needs a moment to come up.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match m.exec("true").await {
            Ok(_) => break,
            Err(e) if std::time::Instant::now() > deadline => panic!("sshd never came up: {e}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(300)).await,
        }
    }

    // Health check and inventory.
    let info = m.check().await.unwrap();
    assert_eq!(info.os, "linux");
    assert_eq!(info.arch, std::env::consts::ARCH);
    assert_eq!(info.home, "/home/tempera");
    assert_eq!(info.remote_dir, "/home/tempera/.tempera-agent");
    assert!(
        info.libc.as_deref().is_some_and(|l| l.starts_with("glibc")),
        "{info:?}"
    );
    assert!(info.uv.is_none(), "the test image has no uv");
    assert!(info.python_ready().is_err());
    assert!(info.agent_version.is_none());
    assert!(info.disk_free_bytes > 0);

    // Agent install: upload, then a no-op when the hash matches.
    let agent_path = m.install_agent(&agent_bin).await.unwrap();
    assert_eq!(agent_path, "/home/tempera/.tempera-agent/bin/tempera-agent");
    let inode = |p: String| {
        let m = m.clone();
        async move {
            m.exec(&format!("stat -c %i {p}"))
                .await
                .unwrap()
                .trim()
                .to_owned()
        }
    };
    let agent_inode = inode(agent_path.clone()).await;
    m.install_agent(&agent_bin).await.unwrap();
    assert_eq!(
        inode(agent_path.clone()).await,
        agent_inode,
        "agent was re-uploaded"
    );
    assert!(m.check().await.unwrap().agent_version.is_some());

    // Worker sync (rsync-like, excludes caches) and its no-op second run.
    let worker = tmp.path().join("python");
    std::fs::create_dir_all(worker.join("pkg/__pycache__")).unwrap();
    std::fs::create_dir_all(worker.join(".venv")).unwrap();
    std::fs::write(worker.join("pyproject.toml"), "[project]\nname = \"w\"\n").unwrap();
    std::fs::write(worker.join("pkg/__pycache__/x.pyc"), "junk").unwrap();
    std::fs::write(worker.join(".venv/junk"), "junk").unwrap();
    let remote_worker = m.sync_worker(&worker).await.unwrap();
    assert_eq!(remote_worker, "/home/tempera/.tempera-agent/workers/python");
    let listing = m
        .exec(&format!("cd {remote_worker} && find . | sort"))
        .await
        .unwrap();
    assert!(listing.contains("./pyproject.toml"), "{listing}");
    assert!(
        !listing.contains("__pycache__") && !listing.contains(".venv"),
        "{listing}"
    );
    m.sync_worker(&worker).await.unwrap();

    // Data cache: the second put of the same file transfers nothing.
    let n = 300;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y: Vec<&str> = x.iter().map(|v| if *v > 15.0 { "yes" } else { "no" }).collect();
    let train = tmp.path().join("train.parquet");
    ParquetWriter::new(std::fs::File::create(&train).unwrap())
        .finish(&mut df!("x" => &x, "__target__" => &y).unwrap())
        .unwrap();
    let first = m.put_cached(&train).await.unwrap();
    assert!(first.uploaded);
    assert_eq!(
        first.remote_path,
        format!("/home/tempera/.tempera-agent/cache/{}.parquet", first.sha256)
    );
    let data_inode = inode(first.remote_path.clone()).await;
    let second = m.put_cached(&train).await.unwrap();
    assert!(!second.uploaded, "second put re-uploaded the file");
    assert_eq!(second.remote_path, first.remote_path);
    assert_eq!(inode(second.remote_path.clone()).await, data_inode);
    assert_eq!(m.put(&train).await.unwrap(), first.remote_path);

    // Agent through the reverse tunnel.
    let hub = Hub::new();
    let (addr, _server) = hub.serve("127.0.0.1:0".parse().unwrap()).await.unwrap();
    let opts = StartOpts {
        agent_bin: Some(agent_bin.clone()),
        ..Default::default()
    };
    let session = m.start(&hub, addr, &opts).await.unwrap();
    assert!((20000..60000).contains(&session.remote_port));
    assert!(session.hardware.cpu_cores >= 1);
    assert_eq!(session.hardware.os, "linux");
    let hw = hub.wait_for(MACHINE, Duration::from_secs(5)).await.unwrap();
    assert_eq!(hw, session.hardware);

    // A python start without uv and without install_uv fails clearly.
    let py = StartOpts {
        agent_bin: Some(agent_bin.clone()),
        worker_dir: Some(worker.clone()),
        ..Default::default()
    };
    let err = m.start(&hub, addr, &py).await.unwrap_err().to_string();
    assert!(err.contains("uv"), "{err}");

    // Native fit whose manifest points at the remote cache.
    let root = m.remote_root().await.unwrap();
    let work = format!("{root}/runs/r1");
    let manifest = serde_json::json!({
        "algorithm": "logistic_regression", "task": "binary_classification",
        "classes": ["no", "yes"], "positive_class": "yes",
        "features": [{"name": "x", "kind": "numeric"}], "params": {}, "seed": 1,
        "train": first.remote_path,
        "predict": [{"input": first.remote_path, "output": format!("{work}/pred.parquet")}],
        "work_dir": work, "export": ["native"]
    });
    let start = StartJob {
        job_id: "job".into(),
        run_id: "r1".into(),
        backend: Backend::Native as i32,
        command: JobCommand::Fit as i32,
        manifest_json: manifest.to_string(),
        ..Default::default()
    };
    let msgs = collect(hub.start(MACHINE, start).await.unwrap()).await;
    let Some(agent_message::Msg::Finished(done)) = msgs.last() else {
        panic!("expected JobFinished, got {msgs:?}")
    };
    assert_eq!(done.run_id, "r1");
    assert!(done.artifacts.iter().any(|a| a.kind == "model"));

    // Outputs back: the whole run dir, and a single file.
    let local_run = tmp.path().join("out/r1");
    m.get(&work, &local_run).await.unwrap();
    let rows = |p: PathBuf| {
        let len = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        ParquetReader::new(std::fs::File::open(&p).unwrap())
            .finish()
            .unwrap_or_else(|e| panic!("{} ({len} bytes): {e}", p.display()))
            .height()
    };
    assert_eq!(rows(local_run.join("pred.parquet")), n);
    let single = tmp.path().join("single/pred.parquet");
    m.get(&format!("{work}/pred.parquet"), &single).await.unwrap();
    assert_eq!(rows(single), n);
    assert!(
        m.get(&format!("{root}/nope"), &tmp.path().join("nope"))
            .await
            .is_err()
    );

    session.stop(&hub).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while hub.connected().await.contains(&MACHINE.to_owned()) {
        assert!(tokio::time::Instant::now() < deadline, "agent did not disconnect");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

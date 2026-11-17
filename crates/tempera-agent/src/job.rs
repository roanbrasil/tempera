//! Runs one `StartJob` as a backend subprocess and forwards its events.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;

use tempera_core::events::{ArtifactKind, Event, LogLevel as EvLevel};
use tempera_proto::v1::{
    AgentMessage, Artifact, Backend, CheckpointReady, Command as JobCommand, JobFailed, JobFinished,
    LogBatch, LogLevel, LogLine, Metric, MetricBatch, Progress, StartJob, agent_message,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, watch};

const STDERR_TAIL: usize = 60;

#[derive(Debug, Clone)]
pub struct JobContext {
    pub worker_dir: Option<PathBuf>,
    pub events: mpsc::Sender<AgentMessage>,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn msg(m: agent_message::Msg) -> AgentMessage {
    AgentMessage { msg: Some(m) }
}

pub async fn reject_busy(tx: &mpsc::Sender<AgentMessage>, start: &StartJob) {
    let _ = tx
        .send(msg(agent_message::Msg::Failed(JobFailed {
            job_id: start.job_id.clone(),
            run_id: start.run_id.clone(),
            reason: "agent is busy with another job".into(),
            retryable: true,
            ..Default::default()
        })))
        .await;
}

/// Executes the job; always ends with exactly one `JobFinished` or `JobFailed`.
pub async fn run(ctx: JobContext, start: StartJob, mut cancel: watch::Receiver<bool>) {
    let tx = ctx.events.clone();
    let (job_id, run_id) = (start.job_id.clone(), start.run_id.clone());
    let fail = |reason: String, retryable: bool, stderr_tail: String, exit_code: Option<i32>| {
        msg(agent_message::Msg::Failed(JobFailed {
            job_id: job_id.clone(),
            run_id: run_id.clone(),
            reason,
            retryable,
            stderr_tail,
            exit_code,
        }))
    };

    let manifest = match write_manifest(&start) {
        Ok(p) => p,
        Err(e) => {
            let _ = tx
                .send(fail(
                    format!("cannot write manifest: {e}"),
                    false,
                    String::new(),
                    None,
                ))
                .await;
            return;
        }
    };
    let mut cmd = match backend_command(&ctx, &start, &manifest) {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(fail(e, false, String::new(), None)).await;
            return;
        }
    };
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (k, v) in &start.env {
        cmd.env(k, v);
    }
    let started = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx
                .send(fail(
                    format!("cannot start backend: {e}"),
                    false,
                    String::new(),
                    None,
                ))
                .await;
            return;
        }
    };
    let pid = child.id();
    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        let _ = tx
            .send(fail(
                "backend pipes unavailable".into(),
                false,
                String::new(),
                None,
            ))
            .await;
        return;
    };

    // stderr → raw logs + tail for the failure report.
    let stderr_task = {
        let tx = tx.clone();
        let (job_id, run_id) = (job_id.clone(), run_id.clone());
        tokio::spawn(async move {
            let mut tail = VecDeque::with_capacity(STDERR_TAIL);
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if tail.len() == STDERR_TAIL {
                    tail.pop_front();
                }
                tail.push_back(line.clone());
                let _ = tx.send(log(&job_id, &run_id, LogLevel::Raw, line)).await;
            }
            tail.into_iter().collect::<Vec<_>>().join("\n")
        })
    };

    let mut out = BufReader::new(stdout).lines();
    let mut summary = None;
    let mut error_event: Option<(String, bool)> = None;
    let mut artifacts = Vec::new();
    let mut cancelled = false;
    loop {
        tokio::select! {
            changed = cancel.changed(), if !cancelled => {
                if changed.is_ok() && *cancel.borrow() {
                    cancelled = true;
                    terminate(pid).await;
                }
            }
            line = out.next_line() => {
                let Ok(Some(line)) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                match Event::parse_line(&line) {
                    Ok(ev) => {
                        if let Some(m) = forward(&job_id, &run_id, ev, &mut summary, &mut error_event, &mut artifacts) {
                            let _ = tx.send(m).await;
                        }
                    }
                    Err(_) => {
                        let _ = tx.send(log(&job_id, &run_id, LogLevel::Raw, line)).await;
                    }
                }
            }
        }
    }
    let status = child.wait().await;
    let stderr_tail = stderr_task.await.unwrap_or_default();
    let _ = std::fs::remove_file(&manifest);
    let code = status.as_ref().ok().and_then(|s| s.code());

    let final_msg = match (code, summary) {
        (Some(0), Some(summary)) => msg(agent_message::Msg::Finished(JobFinished {
            job_id: job_id.clone(),
            run_id: run_id.clone(),
            summary,
            artifacts,
            wall_time_ms: started.elapsed().as_millis() as u64,
        })),
        _ => {
            let (reason, retryable) = match (&error_event, cancelled) {
                (_, true) => ("cancelled".to_owned(), false),
                (Some((m, r)), _) => (m.clone(), *r),
                (None, _) => (
                    match code {
                        Some(c) => format!("backend exited with code {c} without a `done` event"),
                        None => "backend was killed by a signal".to_owned(),
                    },
                    false,
                ),
            };
            fail(reason, retryable, stderr_tail, code)
        }
    };
    let _ = tx.send(final_msg).await;
}

fn forward(
    job_id: &str,
    run_id: &str,
    ev: Event,
    summary: &mut Option<std::collections::HashMap<String, f64>>,
    error_event: &mut Option<(String, bool)>,
    artifacts: &mut Vec<Artifact>,
) -> Option<AgentMessage> {
    match ev {
        Event::Metric {
            name,
            value,
            step,
            epoch,
            ts,
        } => Some(msg(agent_message::Msg::Metrics(MetricBatch {
            job_id: job_id.into(),
            run_id: run_id.into(),
            metrics: vec![Metric {
                name,
                value,
                step,
                epoch,
                unix_ms: ts.map_or_else(now_ms, |t| (t * 1000.0) as i64),
            }],
        }))),
        Event::Log { level, msg: text } => Some(log(
            job_id,
            run_id,
            match level {
                EvLevel::Debug => LogLevel::Debug,
                EvLevel::Info => LogLevel::Info,
                EvLevel::Warn => LogLevel::Warn,
                EvLevel::Error => LogLevel::Error,
            },
            text,
        )),
        Event::Progress { step, total_steps } => Some(msg(agent_message::Msg::Progress(Progress {
            job_id: job_id.into(),
            run_id: run_id.into(),
            step,
            total_steps,
        }))),
        Event::Checkpoint { path, step } => Some(msg(agent_message::Msg::Checkpoint(CheckpointReady {
            job_id: job_id.into(),
            run_id: run_id.into(),
            step,
            uri: path.display().to_string(),
            ..Default::default()
        }))),
        Event::Artifact { kind, path, format } => {
            artifacts.push(Artifact {
                kind: match kind {
                    ArtifactKind::Model => "model",
                    ArtifactKind::Report => "report",
                    ArtifactKind::Predictions => "predictions",
                    ArtifactKind::Other => "other",
                }
                .into(),
                format: format
                    .and_then(|f| serde_json::to_value(f).ok())
                    .and_then(|v| v.as_str().map(str::to_owned))
                    .unwrap_or_default(),
                uri: path.display().to_string(),
                sha256: String::new(),
            });
            None
        }
        Event::Done { summary: s } => {
            *summary = Some(s.into_iter().collect::<BTreeMap<_, _>>().into_iter().collect());
            None
        }
        Event::Error { msg: m, retryable } => {
            *error_event = Some((m.clone(), retryable));
            Some(log(job_id, run_id, LogLevel::Error, m))
        }
    }
}

fn log(job_id: &str, run_id: &str, level: LogLevel, message: String) -> AgentMessage {
    msg(agent_message::Msg::Logs(LogBatch {
        job_id: job_id.into(),
        run_id: run_id.into(),
        lines: vec![LogLine {
            level: level as i32,
            message,
            unix_ms: now_ms(),
        }],
    }))
}

fn write_manifest(start: &StartJob) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join("tempera-agent");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("manifest-{}.json", sanitize(&start.run_id)));
    std::fs::write(&path, &start.manifest_json)?;
    Ok(path)
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn backend_command(
    ctx: &JobContext,
    start: &StartJob,
    manifest: &std::path::Path,
) -> Result<Command, String> {
    let action = match JobCommand::try_from(start.command) {
        Ok(JobCommand::Fit) => "fit",
        Ok(JobCommand::Predict) => "predict",
        _ => return Err("StartJob.command must be FIT or PREDICT".into()),
    };
    match Backend::try_from(start.backend) {
        Ok(Backend::Native) => {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let mut c = Command::new(exe);
            c.arg("native").arg(action).arg("--manifest").arg(manifest);
            Ok(c)
        }
        Ok(Backend::Python) => {
            // An explicit interpreter (with tempera_worker installed) wins over uv.
            if let Ok(python) = std::env::var("TEMPERA_PYTHON") {
                let mut c = Command::new(python);
                c.args(["-m", "tempera_worker", action, "--manifest"])
                    .arg(manifest);
                return Ok(c);
            }
            let dir = ctx
                .worker_dir
                .clone()
                .ok_or("Python backend requested but no --worker-dir is set")?;
            let mut c = Command::new("uv");
            c.arg("run")
                .arg("--project")
                .arg(&dir)
                .args([
                    "--extra",
                    "tabular",
                    "--quiet",
                    "python",
                    "-m",
                    "tempera_worker",
                    action,
                    "--manifest",
                ])
                .arg(manifest);
            Ok(c)
        }
        _ => Err("StartJob.backend must be NATIVE or PYTHON".into()),
    }
}

/// Asks the backend to stop (SIGTERM) so it can checkpoint; kill_on_drop is the fallback.
async fn terminate(pid: Option<u32>) {
    if let Some(pid) = pid {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .status()
            .await;
    }
}

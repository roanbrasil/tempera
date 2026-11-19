//! `tempera-agent`: runs on a worker machine, dials the server and executes
//! jobs as backend subprocesses (docs/wiki/Execution-and-Machines.md).
//!
//! * `tempera-agent connect --server URL` keeps a gRPC session open and runs
//!   `StartJob`s one at a time.
//! * `tempera-agent native fit|predict --manifest FILE` is the Rust backend; it
//!   speaks the same JSONL contract as the Python worker, so the agent treats
//!   both backends identically (ADR-0003).

mod job;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand};
use tempera_proto::v1::agent_control_client::AgentControlClient;
use tempera_proto::v1::{AgentMessage, Heartbeat, Register, agent_message, server_message};
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Debug, Parser)]
#[command(version, about = "Têmpera worker agent")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Connect to a server and execute jobs.
    Connect {
        /// Server gRPC endpoint, e.g. http://127.0.0.1:7071
        #[arg(long, env = "TEMPERA_SERVER")]
        server: String,
        #[arg(long, env = "TEMPERA_MACHINE_ID", default_value = "local")]
        machine_id: String,
        #[arg(long, env = "TEMPERA_AGENT_TOKEN", default_value = "")]
        token: String,
        /// Directory of the Python worker project (workers/python).
        #[arg(long, env = "TEMPERA_WORKER_DIR")]
        worker_dir: Option<PathBuf>,
    },
    /// Native (Rust) backend: same CLI contract as `python -m tempera_worker`.
    Native {
        #[command(subcommand)]
        action: NativeAction,
    },
}

#[derive(Debug, Subcommand)]
enum NativeAction {
    Fit {
        #[arg(long)]
        manifest: PathBuf,
    },
    Predict {
        #[arg(long)]
        manifest: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Native { action } => std::process::exit(native(action)),
        Command::Connect {
            server,
            machine_id,
            token,
            worker_dir,
        } => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
                )
                .init();
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(connect(server, machine_id, token, worker_dir))
        }
    }
}

/// Runs the native backend, printing JSONL events on stdout. Returns the exit code.
fn native(action: NativeAction) -> i32 {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let result = match action {
        NativeAction::Fit { manifest } => std::fs::read_to_string(&manifest)
            .map_err(|e| tempera_native::NativeError::Manifest(format!("{}: {e}", manifest.display())))
            .and_then(|t: String| tempera_native::Manifest::from_json(&t))
            .and_then(|m| tempera_native::run_fit(&m, &mut out).map(|_| ())),
        NativeAction::Predict { manifest } => std::fs::read_to_string(&manifest)
            .map_err(|e| tempera_native::NativeError::Manifest(format!("{}: {e}", manifest.display())))
            .and_then(|t: String| tempera_native::PredictManifest::from_json(&t))
            .and_then(|m| tempera_native::run_predict(&m, &mut out)),
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            use std::io::Write;
            let _ = writeln!(out, "{}", e.to_event().to_line());
            e.exit_code()
        }
    }
}

async fn connect(
    server: String,
    machine_id: String,
    token: String,
    worker_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut client = AgentControlClient::connect(server.clone())
        .await
        .with_context(|| format!("cannot reach server at {server}"))?;
    let (tx, rx) = mpsc::channel::<AgentMessage>(4096);
    tx.send(msg(agent_message::Msg::Register(Register {
        protocol_version: tempera_proto::PROTOCOL_VERSION,
        agent_version: env!("CARGO_PKG_VERSION").to_owned(),
        machine_id: machine_id.clone(),
        auth_token: token,
        hardware: Some(tempera_exec::system::hardware()),
        running_jobs: Vec::new(),
    })))
    .await?;
    let mut inbound = client.session(ReceiverStream::new(rx)).await?.into_inner();
    tracing::info!(%server, %machine_id, "connected");

    let mut heartbeat_every = Duration::from_secs(10);
    let mut heartbeat = tokio::time::interval(heartbeat_every);
    let mut running: Option<(String, watch::Sender<bool>, tokio::task::JoinHandle<()>)> = None;
    let ctx = job::JobContext {
        worker_dir,
        events: tx.clone(),
    };

    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                let _ = tx.send(msg(agent_message::Msg::Heartbeat(Heartbeat {
                    unix_ms: job::now_ms(),
                    ..Default::default()
                }))).await;
            }
            next = inbound.message() => {
                let Some(message) = next? else { break };
                match message.msg {
                    Some(server_message::Msg::Registered(r)) => {
                        if r.heartbeat_interval_s > 0 && Duration::from_secs(r.heartbeat_interval_s.into()) != heartbeat_every {
                            heartbeat_every = Duration::from_secs(r.heartbeat_interval_s.into());
                            heartbeat = tokio::time::interval(heartbeat_every);
                        }
                    }
                    Some(server_message::Msg::Start(start)) => {
                        if running.as_ref().is_some_and(|(_, _, h)| !h.is_finished()) {
                            job::reject_busy(&tx, &start).await;
                            continue;
                        }
                        let (cancel_tx, cancel_rx) = watch::channel(false);
                        let id = start.job_id.clone();
                        let handle = tokio::spawn(job::run(ctx.clone(), start, cancel_rx));
                        running = Some((id, cancel_tx, handle));
                    }
                    Some(server_message::Msg::Cancel(c)) => {
                        if let Some((id, cancel, _)) = &running
                            && *id == c.job_id {
                                let _ = cancel.send(true);
                            }
                    }
                    Some(server_message::Msg::Checkpoint(_)) => {}
                    Some(server_message::Msg::Shutdown(s)) => {
                        tracing::info!(reason = %s.reason, "shutdown requested");
                        break;
                    }
                    None => {}
                }
            }
        }
    }
    if let Some((_, cancel, handle)) = running {
        let _ = cancel.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(15), handle).await;
    }
    Ok(())
}

fn msg(m: agent_message::Msg) -> AgentMessage {
    AgentMessage { msg: Some(m) }
}

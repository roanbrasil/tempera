//! Runs an agent as a child process on this machine.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tempera_proto::v1::Hardware;
use tokio::process::{Child, Command};

use crate::{ExecError, Hub};

pub const LOCAL_MACHINE: &str = "local";

/// Starts `tempera-agent connect` locally and waits until it registers.
#[derive(Debug)]
pub struct LocalExecutor {
    child: Option<Child>,
    pub hardware: Hardware,
}

impl LocalExecutor {
    pub async fn start(hub: &Hub, server_url: &str, worker_dir: Option<&Path>) -> Result<Self, ExecError> {
        let bin = agent_binary().ok_or(ExecError::AgentBinaryNotFound)?;
        let mut cmd = Command::new(bin);
        cmd.args(["connect", "--server", server_url, "--machine-id", LOCAL_MACHINE])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .env(
                "RUST_LOG",
                std::env::var("TEMPERA_AGENT_LOG").unwrap_or_else(|_| "warn".into()),
            )
            .kill_on_drop(true);
        if let Some(dir) = worker_dir {
            cmd.arg("--worker-dir").arg(dir);
        }
        let child = cmd.spawn()?;
        let hardware = hub.wait_for(LOCAL_MACHINE, Duration::from_secs(20)).await?;
        Ok(Self {
            child: Some(child),
            hardware,
        })
    }

    /// Asks the agent to exit, then makes sure the process is gone.
    pub async fn stop(mut self, hub: &Hub) {
        let _ = hub.shutdown(LOCAL_MACHINE, "job finished").await;
        if let Some(mut child) = self.child.take()
            && tokio::time::timeout(Duration::from_secs(10), child.wait())
                .await
                .is_err()
        {
            let _ = child.kill().await;
        }
    }
}

/// `TEMPERA_AGENT_BIN`, then next to the current executable, then `PATH`.
pub fn agent_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("TEMPERA_AGENT_BIN") {
        return Some(PathBuf::from(p));
    }
    let name = format!("tempera-agent{}", std::env::consts::EXE_SUFFIX);
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1).take(3) {
            let candidate = dir.join(&name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(&name))
            .find(|p| p.is_file())
    })
}

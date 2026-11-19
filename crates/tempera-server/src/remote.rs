//! Running a job on an SSH machine (M4): manifests are rewritten so inputs
//! come from the remote content-addressed cache and outputs land in a remote
//! mirror of the local job directory; after each command the outputs are
//! pulled back, so everything downstream (evaluation, report) stays local.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::Value;
use tempera_exec::SshMachine;
use tempera_exec::ssh::shell_quote;

#[derive(Debug)]
pub struct Remote {
    pub ssh: SshMachine,
    pub machine_id: String,
    /// Local job directory and its mirror on the remote machine.
    pub job_dir: PathBuf,
    pub remote_job_dir: String,
}

static REMOTES: LazyLock<Mutex<HashMap<String, Arc<Remote>>>> = LazyLock::new(Default::default);

pub fn set(job_id: &str, r: Remote) {
    if let Ok(mut m) = REMOTES.lock() {
        m.insert(job_id.to_owned(), Arc::new(r));
    }
}

pub fn get(job_id: &str) -> Option<Arc<Remote>> {
    REMOTES.lock().ok()?.get(job_id).cloned()
}

pub fn remove(job_id: &str) {
    if let Ok(mut m) = REMOTES.lock() {
        m.remove(job_id);
    }
}

/// Fields holding input files (uploaded to the cache) and outputs (mirrored).
const INPUT_FILES: [&str; 5] = ["train", "valid", "data", "truth", "predictions"];
const OUTPUT_FILES: [&str; 1] = ["output"];
const DIRS: [&str; 2] = ["work_dir", "model_dir"];

/// A local path → remote path translation plan for one manifest.
#[derive(Debug, Default)]
pub struct Plan {
    /// (remote, local) pairs to download after the command.
    pub pulls: Vec<(String, PathBuf)>,
}

impl Remote {
    fn mirror(&self, local: &Path) -> Option<String> {
        let abs = std::path::absolute(local).ok()?;
        let job = std::path::absolute(&self.job_dir).ok()?;
        let rel = abs.strip_prefix(&job).ok()?;
        Some(format!("{}/{}", self.remote_job_dir, rel.display()))
    }

    async fn upload(&self, v: &mut Value) -> anyhow::Result<()> {
        if let Some(p) = v.as_str().map(PathBuf::from)
            && p.is_file()
        {
            *v = Value::String(self.ssh.put(&p).await?);
        }
        Ok(())
    }

    /// Rewrites a manifest in place and returns what to pull back afterwards.
    pub async fn rewrite(&self, m: &mut Value) -> anyhow::Result<Plan> {
        let mut plan = Plan::default();
        let mut mkdirs: Vec<String> = Vec::new();
        let Some(obj) = m.as_object_mut() else {
            return Ok(plan);
        };
        for k in INPUT_FILES {
            if let Some(v) = obj.get_mut(k) {
                self.upload(v).await?;
            }
        }
        if let Some(Value::Object(sens)) = obj.get_mut("sensitive") {
            for v in sens.values_mut() {
                self.upload(v).await?;
            }
        }
        // `input` of a predict manifest, and every predict[].input/output of a fit manifest.
        if let Some(v) = obj.get_mut("input") {
            self.upload(v).await?;
        }
        let mut pull_work_dir = false;
        for k in DIRS {
            if let Some(local) = obj.get(k).and_then(Value::as_str).map(PathBuf::from)
                && let Some(remote) = self.mirror(&local)
            {
                if k == "work_dir" {
                    mkdirs.push(remote.clone());
                    plan.pulls.push((remote.clone(), local));
                    pull_work_dir = true;
                }
                obj.insert(k.to_owned(), Value::String(remote));
            }
        }
        for k in OUTPUT_FILES {
            if let Some(local) = obj.get(k).and_then(Value::as_str).map(PathBuf::from)
                && let Some(remote) = self.mirror(&local)
            {
                if let Some(parent) = remote.rsplit_once('/').map(|(d, _)| d.to_owned()) {
                    mkdirs.push(parent);
                }
                if !pull_work_dir {
                    plan.pulls.push((remote.clone(), local));
                }
                obj.insert(k.to_owned(), Value::String(remote));
            }
        }
        if let Some(Value::Array(targets)) = obj.get_mut("predict") {
            for t in targets {
                if let Some(v) = t.get_mut("input") {
                    self.upload(v).await?;
                }
                if let Some(local) = t.get("output").and_then(Value::as_str).map(PathBuf::from)
                    && let Some(remote) = self.mirror(&local)
                {
                    t["output"] = Value::String(remote);
                }
            }
        }
        if !mkdirs.is_empty() {
            let cmd = format!(
                "mkdir -p {}",
                mkdirs
                    .iter()
                    .map(|d| shell_quote(d))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            self.ssh.exec(&cmd).await?;
        }
        Ok(plan)
    }

    pub async fn pull(&self, plan: &Plan) -> anyhow::Result<()> {
        for (remote, local) in &plan.pulls {
            if remote.ends_with(".json") || remote.ends_with(".parquet") {
                if let Some(dir) = local.parent() {
                    std::fs::create_dir_all(dir)?;
                }
            } else {
                std::fs::create_dir_all(local)?;
            }
            self.ssh.get(remote, local).await?;
        }
        Ok(())
    }
}

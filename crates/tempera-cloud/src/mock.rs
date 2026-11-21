//! A fake provider for CI and demos. Offers are fixed; instances live in a
//! small JSON state file so separate `tempera` processes (server, `cloud
//! reap`) see the same "account".
//!
//! With `TEMPERA_MOCK_DOCKER_IMAGE` set, `launch` starts a real container
//! from that image (an sshd that authorizes `AUTHORIZED_KEY`/`PUBLIC_KEY`,
//! e.g. `crates/tempera-exec/tests/ssh/Dockerfile`) and reports its mapped
//! SSH port, so a "cloud" job runs through the exact SSH path a real
//! instance would. `preempt` simulates a spot interruption.
//!
//! Environment: `TEMPERA_MOCK_CLOUD_STATE` (state file),
//! `TEMPERA_MOCK_DOCKER_IMAGE`, `TEMPERA_MOCK_SSH_USER` (default `tempera`).

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::{BootSpec, CloudError, Instance, InstanceState, Offer, Provider, ResourceRequest, Result, now_s};

const NAME: &str = "mock";

#[derive(Debug, Default, Serialize, Deserialize)]
struct State {
    next_id: u64,
    instances: Vec<MockInstance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MockInstance {
    id: String,
    name: String,
    state: InstanceState,
    usd_per_hour: f64,
    #[serde(default)]
    container: Option<String>,
    #[serde(default)]
    ssh_port: Option<u16>,
}

#[derive(Debug)]
pub struct Mock {
    state_path: PathBuf,
    docker_image: Option<String>,
    ssh_user: String,
    lock: Mutex<()>,
}

impl Mock {
    pub fn from_env() -> Self {
        let state_path = std::env::var_os("TEMPERA_MOCK_CLOUD_STATE")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("tempera-mock-cloud.json"));
        Self {
            state_path,
            docker_image: std::env::var("TEMPERA_MOCK_DOCKER_IMAGE")
                .ok()
                .filter(|s| !s.is_empty()),
            ssh_user: std::env::var("TEMPERA_MOCK_SSH_USER").unwrap_or_else(|_| "tempera".into()),
            lock: Mutex::new(()),
        }
    }

    pub fn with_state(path: impl Into<PathBuf>) -> Self {
        Self {
            state_path: path.into(),
            docker_image: None,
            ssh_user: "tempera".into(),
            lock: Mutex::new(()),
        }
    }

    pub fn catalog(fetched_at: u64) -> Vec<Offer> {
        let offer = |id: &str, gpu: &str, vram: f64, cores: u32, ram: f64, price: f64, spot: bool| Offer {
            provider: NAME.into(),
            id: id.into(),
            gpu_name: gpu.into(),
            gpu_count: u32::from(vram > 0.0),
            vram_gb: vram,
            cpu_cores: cores,
            ram_gb: ram,
            usd_per_hour: price,
            spot,
            region: Some("local".into()),
            reliability: Some(0.999),
            inet_down_mbps: Some(1000.0),
            disk_gb: Some(100.0),
            fetched_at,
        };
        vec![
            offer("cpu-8", "CPU only", 0.0, 8, 16.0, 0.05, false),
            offer("rtx4090-spot", "RTX 4090", 24.0, 16, 62.0, 0.20, true),
            offer("rtx4090", "RTX 4090", 24.0, 16, 62.0, 0.40, false),
            offer("a100-80", "A100 80GB", 80.0, 16, 125.0, 1.60, false),
        ]
    }

    fn load(&self) -> State {
        std::fs::read_to_string(&self.state_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    fn save(&self, s: &State) -> Result<()> {
        if let Some(dir) = self.state_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.state_path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(s)?)?;
        std::fs::rename(tmp, &self.state_path)?;
        Ok(())
    }

    fn with_state_mut<T>(&self, f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
        let _g = self
            .lock
            .lock()
            .map_err(|_| CloudError::Refused("mock state poisoned".into()))?;
        let mut s = self.load();
        let out = f(&mut s)?;
        self.save(&s)?;
        Ok(out)
    }

    /// Simulates a spot interruption: the instance (and its container) disappears.
    pub fn preempt(&self, id: &str) -> Result<()> {
        self.with_state_mut(|s| {
            if let Some(i) = s.instances.iter_mut().find(|i| i.id == id) {
                i.state = InstanceState::Gone;
                if let Some(c) = i.container.take() {
                    let _ = Command::new("docker").args(["rm", "-f", &c]).output();
                }
            }
            Ok(())
        })
    }

    fn to_instance(&self, m: &MockInstance) -> Instance {
        let ssh = m.ssh_port.filter(|_| m.state == InstanceState::Running);
        Instance {
            provider: NAME.into(),
            id: m.id.clone(),
            name: m.name.clone(),
            state: m.state,
            ssh_host: ssh.map(|_| "127.0.0.1".to_owned()),
            ssh_port: ssh,
            ssh_user: Some(self.ssh_user.clone()),
            usd_per_hour: Some(m.usd_per_hour),
        }
    }
}

fn docker(args: &[&str]) -> Result<String> {
    let out = Command::new("docker").args(args).output()?;
    if !out.status.success() {
        return Err(CloudError::Api {
            provider: NAME.into(),
            message: format!(
                "docker {}: {}",
                args.first().unwrap_or(&""),
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

impl Provider for Mock {
    fn name(&self) -> &str {
        NAME
    }

    fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>> {
        let mut v: Vec<Offer> = Self::catalog(now_s())
            .into_iter()
            .filter(|o| req.accepts(o))
            .collect();
        v.sort_by(|a, b| a.usd_per_hour.total_cmp(&b.usd_per_hour));
        Ok(v)
    }

    fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<String> {
        let (container, ssh_port) = match &self.docker_image {
            Some(image) => {
                let key = boot.ssh_public_key.trim();
                let id = docker(&[
                    "run",
                    "-d",
                    "--label",
                    "tempera.mock=1",
                    "-p",
                    "127.0.0.1::22",
                    "-e",
                    &format!("AUTHORIZED_KEY={key}"),
                    "-e",
                    &format!("PUBLIC_KEY={key}"),
                    image,
                ])?;
                let mapped = docker(&["port", &id, "22/tcp"])?;
                let port = mapped
                    .lines()
                    .next()
                    .and_then(|l| l.rsplit(':').next())
                    .and_then(|p| p.parse().ok());
                (Some(id), port)
            }
            None => (None, None),
        };
        self.with_state_mut(|s| {
            s.next_id += 1;
            let id = format!("mock-{}", s.next_id);
            s.instances.push(MockInstance {
                id: id.clone(),
                name: boot.name.clone(),
                state: InstanceState::Running,
                usd_per_hour: offer.usd_per_hour,
                container,
                ssh_port,
            });
            Ok(id)
        })
    }

    fn status(&self, id: &str) -> Result<Instance> {
        let s = self.load();
        Ok(s.instances
            .iter()
            .find(|i| i.id == id)
            .map(|m| self.to_instance(m))
            .unwrap_or(Instance {
                provider: NAME.into(),
                id: id.to_owned(),
                name: String::new(),
                state: InstanceState::Gone,
                ssh_host: None,
                ssh_port: None,
                ssh_user: None,
                usd_per_hour: None,
            }))
    }

    fn instances(&self) -> Result<Vec<Instance>> {
        Ok(self
            .load()
            .instances
            .iter()
            .filter(|i| i.state != InstanceState::Gone)
            .map(|m| self.to_instance(m))
            .collect())
    }

    fn terminate(&self, id: &str) -> Result<()> {
        self.with_state_mut(|s| {
            if let Some(i) = s.instances.iter_mut().find(|i| i.id == id) {
                i.state = InstanceState::Gone;
                if let Some(c) = i.container.take() {
                    let _ = Command::new("docker").args(["rm", "-f", &c]).output();
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn launch_status_terminate_and_preempt() {
        let dir = tempfile::tempdir().unwrap();
        let m = Mock::with_state(dir.path().join("s.json"));
        let offers = m.list_offers(&ResourceRequest::default()).unwrap();
        assert_eq!(
            offers[0].gpu_name, "RTX 4090",
            "cheapest GPU offer first (CPU-only filtered)"
        );
        let a = m
            .launch(&offers[0], &BootSpec::for_job("0000beef", "j1", "k"))
            .unwrap();
        let b = m
            .launch(&offers[0], &BootSpec::for_job("0000beef", "j2", "k"))
            .unwrap();
        assert_eq!(m.instances().unwrap().len(), 2);
        assert_eq!(m.status(&a).unwrap().state, InstanceState::Running);
        m.preempt(&a).unwrap();
        assert_eq!(m.status(&a).unwrap().state, InstanceState::Gone);
        m.terminate(&b).unwrap();
        m.terminate(&b).unwrap(); // idempotent
        assert!(m.instances().unwrap().is_empty());
        // State survives a new client (another process).
        let m2 = Mock::with_state(dir.path().join("s.json"));
        assert_eq!(m2.status(&b).unwrap().state, InstanceState::Gone);
    }
}

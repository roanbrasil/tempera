//! Machine registry: `<TEMPERA_HOME>/machines.yaml`.
//!
//! ```yaml
//! machines:
//!   - name: living-room-pc
//!     kind: ssh
//!     host: 192.168.0.42
//!     user: bruno
//!     identity: ~/.ssh/id_ed25519   # optional; ssh-agent / ~/.ssh/config otherwise
//!     tags: [gpu, rtx3090]
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ExecError;
use crate::ssh::{DEFAULT_REMOTE_DIR, SshMachine};

pub const MACHINES_FILE: &str = "machines.yaml";

/// `<home>/machines.yaml`.
pub fn machines_path(tempera_home: &Path) -> PathBuf {
    tempera_home.join(MACHINES_FILE)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineKind {
    #[default]
    Ssh,
}

/// One registered machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineConfig {
    pub name: String,
    #[serde(default)]
    pub kind: MachineKind,
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Private key file; `key` is accepted as an alias.
    #[serde(default, alias = "key", skip_serializing_if = "Option::is_none")]
    pub identity: Option<PathBuf>,
    /// Remote state directory (default `~/.tempera-agent`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Extra `ssh -o` options (`Key=Value`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ssh_options: Vec<String>,
}

impl MachineConfig {
    pub fn new(name: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: MachineKind::Ssh,
            host: host.into(),
            user: None,
            port: None,
            identity: None,
            remote_dir: None,
            tags: Vec::new(),
            ssh_options: Vec::new(),
        }
    }

    /// The SSH handle for this machine (`~/` in `identity` expanded locally).
    pub fn to_ssh(&self) -> Result<SshMachine, ExecError> {
        let mut m = SshMachine::new(&self.name, &self.host);
        m.user = self.user.clone();
        m.port = self.port;
        m.identity = self.identity.as_deref().map(expand_tilde);
        m.remote_dir = self
            .remote_dir
            .clone()
            .unwrap_or_else(|| DEFAULT_REMOTE_DIR.to_owned());
        m.ssh_options = self.ssh_options.clone();
        m.validate()?;
        Ok(m)
    }
}

/// The whole registry file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachinesFile {
    #[serde(default)]
    pub machines: Vec<MachineConfig>,
}

impl MachinesFile {
    /// Loads the registry; a missing file is an empty registry.
    pub fn load(path: &Path) -> Result<Self, ExecError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e.into()),
        };
        Self::from_yaml(&text).map_err(|e| ExecError::Config(format!("{}: {e}", path.display())))
    }

    pub fn from_yaml(text: &str) -> Result<Self, ExecError> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let file: Self = serde_saphyr::from_str(text).map_err(|e| ExecError::Config(e.to_string()))?;
        let mut seen = std::collections::HashSet::new();
        for m in &file.machines {
            if !seen.insert(m.name.as_str()) {
                return Err(ExecError::Config(format!(
                    "machine `{}` is defined twice",
                    m.name
                )));
            }
        }
        Ok(file)
    }

    pub fn to_yaml(&self) -> Result<String, ExecError> {
        serde_saphyr::to_string(self).map_err(|e| ExecError::Config(e.to_string()))
    }

    /// Writes the registry atomically (temp file + rename), creating the directory.
    pub fn save(&self, path: &Path) -> Result<(), ExecError> {
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        std::fs::write(&tmp, self.to_yaml()?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&MachineConfig> {
        self.machines.iter().find(|m| m.name == name)
    }

    /// Inserts or replaces by name; returns `true` when it replaced one.
    pub fn upsert(&mut self, machine: MachineConfig) -> bool {
        match self.machines.iter_mut().find(|m| m.name == machine.name) {
            Some(existing) => {
                *existing = machine;
                true
            }
            None => {
                self.machines.push(machine);
                false
            }
        }
    }

    /// Removes by name; returns the removed machine.
    pub fn remove(&mut self, name: &str) -> Option<MachineConfig> {
        let i = self.machines.iter().position(|m| m.name == name)?;
        Some(self.machines.remove(i))
    }
}

fn expand_tilde(p: &Path) -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    match p.strip_prefix("~") {
        Ok(rest) => home().map_or_else(|| p.to_path_buf(), |h| h.join(rest)),
        Err(_) => p.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn round_trips_through_yaml_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = machines_path(&tmp.path().join("home"));
        assert_eq!(MachinesFile::load(&path).unwrap(), MachinesFile::default());

        let mut file = MachinesFile::default();
        let mut pc = MachineConfig::new("living-room-pc", "192.168.0.42");
        pc.user = Some("bruno".into());
        pc.port = Some(2222);
        pc.identity = Some("~/.ssh/id_ed25519".into());
        pc.tags = vec!["gpu".into(), "rtx3090".into()];
        assert!(!file.upsert(pc.clone()));
        assert!(!file.upsert(MachineConfig::new("nas", "nas.lan")));
        file.save(&path).unwrap();

        let loaded = MachinesFile::load(&path).unwrap();
        assert_eq!(loaded, file);
        assert_eq!(loaded.get("living-room-pc"), Some(&pc));

        let ssh = pc.to_ssh().unwrap();
        assert_eq!(ssh.destination(), "bruno@192.168.0.42");
        assert_eq!(ssh.port, Some(2222));
        assert_eq!(ssh.remote_dir, DEFAULT_REMOTE_DIR);
        assert!(!ssh.identity.unwrap().starts_with("~"));

        let mut edited = loaded;
        let mut nas = MachineConfig::new("nas", "10.0.0.2");
        nas.remote_dir = Some("/srv/tempera".into());
        assert!(edited.upsert(nas));
        assert_eq!(edited.remove("living-room-pc").unwrap().name, "living-room-pc");
        assert!(edited.remove("nope").is_none());
        edited.save(&path).unwrap();
        let again = MachinesFile::load(&path).unwrap();
        assert_eq!(again.machines.len(), 1);
        assert_eq!(again.machines[0].host, "10.0.0.2");
        assert_eq!(again.machines[0].to_ssh().unwrap().remote_dir, "/srv/tempera");
    }

    #[test]
    fn reads_the_wiki_format() {
        let text = "machines:\n  - name: living-room-pc\n    kind: ssh\n    host: 192.168.0.42\n    \
                    user: bruno\n    key: ~/.ssh/id_ed25519\n    tags: [gpu, rtx3090]\n";
        let f = MachinesFile::from_yaml(text).unwrap();
        assert_eq!(
            f.machines[0].identity.as_deref(),
            Some(Path::new("~/.ssh/id_ed25519"))
        );
        assert_eq!(f.machines[0].kind, MachineKind::Ssh);
        assert!(MachinesFile::from_yaml("").unwrap().machines.is_empty());
    }

    #[test]
    fn rejects_duplicates_and_bad_hosts() {
        let dup = "machines:\n  - {name: a, host: h}\n  - {name: a, host: h2}\n";
        assert!(MachinesFile::from_yaml(dup).is_err());
        assert!(MachineConfig::new("a", "-oProxyCommand=x").to_ssh().is_err());
    }
}

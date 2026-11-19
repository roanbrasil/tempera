//! Remote machines over SSH (M4, ADR-0007).
//!
//! Everything goes through the **system OpenSSH client** (`ssh`, plus local
//! `tar` for directory transfers), so `~/.ssh/config`, `ssh-agent`,
//! `ProxyJump` and `known_hosts` behave exactly as they do for the user.
//! Every invocation passes `BatchMode=yes` (never prompt), `ConnectTimeout=10`
//! and `ServerAliveInterval=15`; host key checking is never relaxed here
//! (tests opt in through [`SshMachine::ssh_options`]).
//!
//! What lives on the remote side, under [`SshMachine::remote_dir`]
//! (default `~/.tempera-agent`):
//!
//! ```text
//! bin/tempera-agent        uploaded only when its sha256 differs
//! cache/<sha256>.<ext>     content-addressed data cache (put)
//! workers/<name>/          synced Python worker project (.venv kept between syncs)
//! ```
//!
//! The agent is started with `ssh -R <port>:<hub>` so it dials the local hub
//! through the SSH connection itself: nothing listens on the network on
//! either side. Remote shell commands are always run as `sh -c <script>`, and
//! every interpolated value goes through [`shell_quote`].

use std::collections::VecDeque;
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};
use tempera_proto::v1::Hardware;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, OnceCell};

use crate::{ExecError, Hub};

/// Default remote state directory (`~` is resolved on the remote side).
pub const DEFAULT_REMOTE_DIR: &str = "~/.tempera-agent";

/// Options passed to every `ssh` invocation. On the command line the first
/// value of an option wins, so these cannot be overridden by `ssh_options`.
const BASE_OPTIONS: [&str; 3] = ["BatchMode=yes", "ConnectTimeout=10", "ServerAliveInterval=15"];

/// Never shipped to a worker: virtualenvs and tool caches.
const WORKER_EXCLUDES: [&str; 6] = [
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    SYNC_MARKER,
];
const SYNC_MARKER: &str = ".tempera-sync-hash";

/// Prepended to remote commands that need user-local tools (`uv`).
const REMOTE_PATH: &str = r#"PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; export PATH"#;

/// Official uv installer (only used with [`StartOpts::install_uv`]).
const UV_INSTALLER: &str = "https://astral.sh/uv/install.sh";

/// Range for the remote end of the reverse tunnel (`-R 0` is not portable).
const TUNNEL_PORTS: std::ops::Range<u16> = 20000..60000;
const TUNNEL_ATTEMPTS: usize = 6;

/// One machine reachable with `ssh [user@]host`.
///
/// Build it with [`SshMachine::new`] and set the public fields; it caches the
/// remote `$HOME` after the first call, so clone it rather than rebuilding it.
#[derive(Debug, Clone)]
pub struct SshMachine {
    /// Machine id: the agent registers with the hub under this name.
    pub name: String,
    /// Host name, IP or `~/.ssh/config` alias.
    pub host: String,
    pub user: Option<String>,
    pub port: Option<u16>,
    /// Private key (`-i`, with `IdentitiesOnly=yes`); `None` uses ssh-agent/config.
    pub identity: Option<PathBuf>,
    /// Remote state directory; `~/` is expanded on the remote side.
    pub remote_dir: String,
    /// Extra `-o` options, e.g. `UserKnownHostsFile=/tmp/kh` in tests.
    pub ssh_options: Vec<String>,
    home: Arc<OnceCell<String>>,
}

/// What [`SshMachine::check`] found on the remote machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RemoteInfo {
    /// `linux` or `macos` (other `uname -s` values are lower-cased as is).
    pub os: String,
    /// `x86_64` or `aarch64` (normalized from `uname -m`).
    pub arch: String,
    /// e.g. `glibc 2.36` or `musl`; `None` when it cannot be told (macOS).
    pub libc: Option<String>,
    /// Remote `$HOME`.
    pub home: String,
    /// Absolute remote state directory.
    pub remote_dir: String,
    /// Path to `uv`, looking also in `~/.local/bin` and `~/.cargo/bin`.
    pub uv: Option<String>,
    pub python3: Option<String>,
    /// `curl` or `wget`, needed to install uv.
    pub downloader: Option<String>,
    /// Free bytes on the filesystem holding `$HOME`.
    pub disk_free_bytes: u64,
    /// Version of the installed agent, if any.
    pub agent_version: Option<String>,
}

impl RemoteInfo {
    /// Error when the Python backend cannot run there (no `uv`).
    pub fn python_ready(&self) -> Result<(), String> {
        if self.uv.is_some() {
            return Ok(());
        }
        let how = if self.downloader.is_some() {
            "start it with install_uv enabled (`--install-uv`) to install it user-locally"
        } else {
            "install it manually (neither curl nor wget is available there)"
        };
        Err(format!(
            "`uv` is not installed on the remote machine, so Python backends cannot run; {how}"
        ))
    }

    /// Human-readable problems worth showing after a check.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if let Err(e) = self.python_ready() {
            w.push(e);
        }
        if self.disk_free_bytes > 0 && self.disk_free_bytes < 1 << 30 {
            w.push(format!(
                "less than 1 GiB free on the remote disk ({} MiB)",
                self.disk_free_bytes >> 20
            ));
        }
        w
    }
}

/// Result of [`SshMachine::put_cached`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFile {
    /// Absolute remote path, `<remote_dir>/cache/<sha256>[.<ext>]`.
    pub remote_path: String,
    pub sha256: String,
    /// `false` when the file was already in the remote cache.
    pub uploaded: bool,
}

/// Options for [`SshMachine::start`].
#[derive(Debug, Clone)]
pub struct StartOpts {
    /// Local agent binary to install; defaults to [`crate::local::agent_binary`].
    pub agent_bin: Option<PathBuf>,
    /// Local Python worker project (`workers/python`). When set it is synced
    /// and the agent gets `--worker-dir`, so Python backends run with
    /// `uv run --project <remote worker dir>`.
    pub worker_dir: Option<PathBuf>,
    /// Install `uv` user-locally with the official installer when missing
    /// (only matters with `worker_dir`). Default `false`.
    pub install_uv: bool,
    /// How long to wait for the agent to register.
    pub timeout: Duration,
    /// `RUST_LOG` for the remote agent.
    pub agent_log: String,
}

impl Default for StartOpts {
    fn default() -> Self {
        Self {
            agent_bin: None,
            worker_dir: None,
            install_uv: false,
            timeout: Duration::from_secs(60),
            agent_log: std::env::var("TEMPERA_AGENT_LOG").unwrap_or_else(|_| "warn".into()),
        }
    }
}

/// A running remote agent: the `ssh` process carrying the tunnel and the agent.
#[derive(Debug)]
pub struct SshSession {
    pub machine_id: String,
    /// Remote end of the reverse tunnel (`127.0.0.1:<port>` on the machine).
    pub remote_port: u16,
    /// Inventory the agent sent in `Register`.
    pub hardware: Hardware,
    pub info: RemoteInfo,
    /// Absolute remote path of the agent binary.
    pub agent_path: String,
    /// Absolute remote worker dir, when a worker was synced.
    pub remote_worker_dir: Option<String>,
    child: Option<Child>,
}

impl SshSession {
    /// Asks the agent to exit, then makes sure the `ssh` process is gone
    /// (closing the tunnel also ends a stuck agent: its gRPC stream breaks).
    pub async fn stop(mut self, hub: &Hub) {
        let _ = hub.shutdown(&self.machine_id, "session stopped").await;
        if let Some(mut child) = self.child.take()
            && tokio::time::timeout(Duration::from_secs(10), child.wait())
                .await
                .is_err()
        {
            let _ = child.kill().await;
        }
    }
}

impl SshMachine {
    pub fn new(name: impl Into<String>, host: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            host: host.into(),
            user: None,
            port: None,
            identity: None,
            remote_dir: DEFAULT_REMOTE_DIR.to_owned(),
            ssh_options: Vec::new(),
            home: Arc::default(),
        }
    }

    /// `user@host` (or `host`).
    pub fn destination(&self) -> String {
        match &self.user {
            Some(u) => format!("{u}@{}", self.host),
            None => self.host.clone(),
        }
    }

    /// Rejects values that `ssh` would read as options (argument injection).
    pub fn validate(&self) -> Result<(), ExecError> {
        let bad = |what: &str, v: &str| ExecError::Config(format!("invalid {what} `{v}` for machine"));
        if self.name.is_empty()
            || !self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err(bad("name (use letters, digits, `-`, `_`, `.`)", &self.name));
        }
        if self.host.is_empty()
            || self.host.starts_with('-')
            || self
                .host
                .chars()
                .any(|c| c.is_whitespace() || c.is_control() || c == '@')
        {
            return Err(bad("host", &self.host));
        }
        if let Some(u) = &self.user
            && (u.is_empty() || u.starts_with('-') || u.chars().any(|c| c.is_whitespace() || c == '@'))
        {
            return Err(bad("user", u));
        }
        if self.remote_dir.trim().is_empty() {
            return Err(bad("remote_dir", &self.remote_dir));
        }
        for o in &self.ssh_options {
            if o.starts_with('-') || !o.contains('=') {
                return Err(bad("ssh option (expected Key=Value)", o));
            }
        }
        Ok(())
    }

    fn err(&self, message: impl Into<String>) -> ExecError {
        ExecError::Ssh {
            machine: self.name.clone(),
            message: message.into(),
        }
    }

    /// `ssh` with the base options, port, identity and extra options; the
    /// caller appends more `-o`s, then [`Self::destination`] and the command.
    fn ssh(&self) -> Result<Command, ExecError> {
        self.validate()?;
        let mut c = Command::new("ssh");
        c.arg("-T");
        for o in BASE_OPTIONS {
            c.arg("-o").arg(o);
        }
        if let Some(p) = self.port {
            c.arg("-p").arg(p.to_string());
        }
        if let Some(i) = &self.identity {
            c.arg("-i").arg(i).arg("-o").arg("IdentitiesOnly=yes");
        }
        for o in &self.ssh_options {
            c.arg("-o").arg(o);
        }
        c.kill_on_drop(true);
        Ok(c)
    }

    /// `ssh ... <dest> sh -c <script>` with the given stdin; stdout/stderr piped.
    fn script_cmd(&self, script: &str, stdin: Stdio) -> Result<Command, ExecError> {
        let mut c = self.ssh()?;
        c.arg(self.destination())
            .arg(format!("sh -c {}", shell_quote(script)))
            .stdin(stdin)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok(c)
    }

    /// Runs a script remotely and returns its stdout; non-zero exit is an error.
    async fn run(&self, what: &str, script: &str, stdin: Stdio) -> Result<String, ExecError> {
        let out = self.script_cmd(script, stdin)?.output().await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                self.err("the `ssh` client is not installed (OpenSSH is required)")
            } else {
                self.err(format!("{what}: cannot run ssh: {e}"))
            }
        })?;
        self.check_status(what, out.status, &out.stderr)?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    fn check_status(
        &self,
        what: &str,
        status: std::process::ExitStatus,
        stderr: &[u8],
    ) -> Result<(), ExecError> {
        if status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(stderr);
        let stderr = stderr.trim();
        Err(match status.code() {
            Some(255) => self.err(format!(
                "cannot connect to {} over ssh ({what}): {stderr}",
                self.destination()
            )),
            Some(c) => self.err(format!("{what} failed (exit code {c}): {stderr}")),
            None => self.err(format!("{what} was killed by a signal: {stderr}")),
        })
    }

    /// Runs `cmd` with `sh -c` on the remote machine and returns its stdout.
    /// `cmd` is used verbatim: quote interpolated values with [`shell_quote`].
    pub async fn exec(&self, cmd: &str) -> Result<String, ExecError> {
        self.run("remote command", cmd, Stdio::null()).await
    }

    /// Remote `$HOME` (cached).
    pub async fn home(&self) -> Result<String, ExecError> {
        self.home
            .get_or_try_init(|| async {
                let home = self
                    .run("read $HOME", r#"printf '%s' "$HOME""#, Stdio::null())
                    .await?;
                let home = home.trim().to_owned();
                if !home.starts_with('/') {
                    return Err(self.err(format!("unexpected remote $HOME `{home}`")));
                }
                Ok(home)
            })
            .await
            .cloned()
    }

    /// Resolves a remote path: `~` and `~/x` against the remote `$HOME`,
    /// relative paths against `$HOME` too; absolute paths unchanged.
    pub async fn resolve(&self, path: &str) -> Result<String, ExecError> {
        if path.starts_with('/') {
            return Ok(path.to_owned());
        }
        let home = self.home().await?;
        let rest = path.strip_prefix('~').map_or(path, |r| r.trim_start_matches('/'));
        Ok(if rest.is_empty() {
            home
        } else {
            format!("{}/{rest}", home.trim_end_matches('/'))
        })
    }

    /// Absolute remote state directory.
    pub async fn remote_root(&self) -> Result<String, ExecError> {
        let root = self.resolve(&self.remote_dir).await?;
        Ok(root.trim_end_matches('/').to_owned())
    }

    /// Health check and inventory: OS, arch, libc, uv/python, free disk and
    /// the installed agent version. Fails only if the machine is unreachable;
    /// missing `uv` is reported by [`RemoteInfo::python_ready`]/`warnings`.
    pub async fn check(&self) -> Result<RemoteInfo, ExecError> {
        let root = self.remote_root().await?;
        let agent = format!("{root}/bin/tempera-agent");
        let script = format!(
            r#"{REMOTE_PATH}
echo "os=$(uname -s)"
echo "arch=$(uname -m)"
echo "libc=$(ldd --version 2>&1 | head -n 1)"
echo "home=$HOME"
echo "uv=$(command -v uv)"
echo "python3=$(command -v python3)"
echo "downloader=$(command -v curl || command -v wget)"
echo "disk_free_kb=$(df -Pk "$HOME" 2>/dev/null | awk 'NR==2 {{print $4}}')"
if [ -x {a} ]; then echo "agent_version=$({a} --version 2>/dev/null | head -n 1)"; fi
exit 0"#,
            a = shell_quote(&agent)
        );
        let out = self.run("health check", &script, Stdio::null()).await?;
        let mut info = parse_remote_info(&out);
        info.remote_dir = root;
        if info.os.is_empty() || info.arch.is_empty() {
            return Err(self.err(format!("unexpected health check output: {out:?}")));
        }
        for w in info.warnings() {
            tracing::debug!(machine = %self.name, "{w}");
        }
        Ok(info)
    }

    /// Installs the agent binary at `<remote_dir>/bin/tempera-agent`. Uploads
    /// only when the remote sha256 differs; the upload is verified and moved
    /// into place atomically. Returns the absolute remote path.
    pub async fn install_agent(&self, local_agent: &Path) -> Result<String, ExecError> {
        let info = self.check().await?;
        self.install_agent_with(&info, local_agent).await
    }

    async fn install_agent_with(&self, info: &RemoteInfo, local_agent: &Path) -> Result<String, ExecError> {
        if !local_agent.is_file() {
            return Err(self.err(format!("agent binary {} not found", local_agent.display())));
        }
        if let Some((os, arch)) = binary_platform(local_agent)
            && (os != info.os || arch != info.arch)
        {
            return Err(self.err(format!(
                "{} is built for {os}/{arch} but the machine is {}/{}; build tempera-agent for that target",
                local_agent.display(),
                info.os,
                info.arch
            )));
        }
        let sha = sha256_file(local_agent).await?;
        let remote = format!("{}/bin/tempera-agent", info.remote_dir);
        let current = self
            .run(
                "hash remote agent",
                &format!(
                    "if [ -f {p} ]; then {}; fi",
                    sha256_cmd(&shell_quote(&remote)),
                    p = shell_quote(&remote)
                ),
                Stdio::null(),
            )
            .await?;
        if current.trim() == sha {
            tracing::debug!(machine = %self.name, "agent is up to date");
            return Ok(remote);
        }
        tracing::info!(machine = %self.name, %remote, "uploading tempera-agent");
        self.upload_file(local_agent, &remote, &sha, "755").await?;
        Ok(remote)
    }

    /// Streams a local file to `remote` (absolute) through `cat`, verifying
    /// the sha256 remotely before the atomic `mv`.
    async fn upload_file(&self, local: &Path, remote: &str, sha: &str, mode: &str) -> Result<(), ExecError> {
        let dir = parent_dir(remote);
        let script = format!(
            r#"set -e
mkdir -p {dir}
tmp={remote}.part.$$
trap 'rm -f "$tmp"' EXIT
cat > "$tmp"
got=$({hash})
if [ "$got" != {sha} ]; then echo "checksum mismatch after upload (got $got)" >&2; exit 3; fi
chmod {mode} "$tmp"
mv -f "$tmp" {remote}"#,
            dir = shell_quote(dir),
            remote = shell_quote(remote),
            hash = sha256_cmd(r#""$tmp""#),
            sha = shell_quote(sha),
            mode = shell_quote(mode),
        );
        let file = std::fs::File::open(local)?;
        self.run(&format!("upload {}", local.display()), &script, Stdio::from(file))
            .await
            .map(|_| ())
    }

    /// Installs `uv` user-locally (`~/.local/bin`) with the official installer.
    /// Returns the path of the installed binary.
    pub async fn install_uv(&self) -> Result<String, ExecError> {
        let script = format!(
            r#"set -e
{REMOTE_PATH}
if command -v curl >/dev/null 2>&1; then curl -LsSf {url} | env UV_NO_MODIFY_PATH=1 sh >&2
elif command -v wget >/dev/null 2>&1; then wget -qO- {url} | env UV_NO_MODIFY_PATH=1 sh >&2
else echo "neither curl nor wget is available to download uv" >&2; exit 4; fi
command -v uv"#,
            url = shell_quote(UV_INSTALLER)
        );
        let out = self.run("install uv", &script, Stdio::null()).await?;
        Ok(out.trim().to_owned())
    }

    /// Syncs a local worker project to `<remote_dir>/workers/<dir name>`,
    /// excluding `.venv`, `__pycache__`, `.pytest_cache` and tool caches.
    /// Skipped when the content hash matches the last sync; the remote
    /// `.venv` is kept so uv's environment stays warm. Returns the remote dir.
    pub async fn sync_worker(&self, local_worker_dir: &Path) -> Result<String, ExecError> {
        if !local_worker_dir.is_dir() {
            return Err(self.err(format!("worker dir {} not found", local_worker_dir.display())));
        }
        let base = local_worker_dir
            .canonicalize()?
            .file_name()
            .map(|n| sanitize_component(&n.to_string_lossy()))
            .unwrap_or_else(|| "worker".into());
        let remote = format!("{}/workers/{base}", self.remote_root().await?);
        let local = local_worker_dir.to_path_buf();
        let hash = tokio::task::spawn_blocking(move || hash_dir(&local))
            .await
            .map_err(|e| self.err(format!("hashing worker dir: {e}")))??;
        let marker = format!("{remote}/{SYNC_MARKER}");
        let current = self
            .run(
                "read worker sync marker",
                &format!("cat {} 2>/dev/null; exit 0", shell_quote(&marker)),
                Stdio::null(),
            )
            .await?;
        if current.trim() == hash {
            tracing::debug!(machine = %self.name, "worker is up to date");
            return Ok(remote);
        }
        tracing::info!(machine = %self.name, %remote, "syncing worker");
        let mut tar = Command::new("tar");
        tar.arg("-C").arg(local_worker_dir);
        for ex in WORKER_EXCLUDES {
            tar.arg(format!("--exclude={ex}"));
        }
        tar.args(["-cf", "-", "."])
            .env("COPYFILE_DISABLE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut tar = tar.spawn()?;
        let tar_out: Stdio = tar
            .stdout
            .take()
            .ok_or_else(|| self.err("tar stdout unavailable"))?
            .try_into()?;
        let script = format!(
            r#"set -e
mkdir -p {d}
find {d} -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {{}} +
tar -C {d} -xf -
printf '%s' {h} > {m}"#,
            d = shell_quote(&remote),
            h = shell_quote(&hash),
            m = shell_quote(&marker),
        );
        let ssh = self.script_cmd(&script, tar_out)?.spawn()?;
        let (tar_res, ssh_res) = tokio::join!(tar.wait_with_output(), ssh.wait_with_output());
        let (tar_res, ssh_res) = (tar_res?, ssh_res?);
        self.check_status("sync worker", ssh_res.status, &ssh_res.stderr)?;
        if !tar_res.status.success() {
            return Err(self.err(format!(
                "local tar failed: {}",
                String::from_utf8_lossy(&tar_res.stderr).trim()
            )));
        }
        Ok(remote)
    }

    /// Uploads `local` into the content-addressed cache and returns the
    /// absolute remote path. See [`Self::put_cached`].
    pub async fn put(&self, local: &Path) -> Result<String, ExecError> {
        Ok(self.put_cached(local).await?.remote_path)
    }

    /// Content-addressed upload to `<remote_dir>/cache/<sha256>[.<ext>]`.
    /// Nothing is transferred when the file is already cached (its mtime is
    /// refreshed, for a future LRU cleanup); uploads are verified remotely.
    pub async fn put_cached(&self, local: &Path) -> Result<CachedFile, ExecError> {
        if !local.is_file() {
            return Err(self.err(format!("{} is not a file", local.display())));
        }
        let sha = sha256_file(local).await?;
        let ext = local
            .extension()
            .map(|e| {
                e.to_string_lossy()
                    .chars()
                    .filter(char::is_ascii_alphanumeric)
                    .collect::<String>()
            })
            .filter(|e| !e.is_empty())
            .map(|e| format!(".{e}"))
            .unwrap_or_default();
        let remote = format!("{}/cache/{sha}{ext}", self.remote_root().await?);
        let q = shell_quote(&remote);
        let state = self
            .run(
                "look up data cache",
                &format!("if [ -f {q} ]; then touch {q}; echo hit; else echo miss; fi"),
                Stdio::null(),
            )
            .await?;
        let uploaded = state.trim() != "hit";
        if uploaded {
            tracing::info!(machine = %self.name, file = %local.display(), %remote, "uploading to data cache");
            self.upload_file(local, &remote, &sha, "644").await?;
        }
        Ok(CachedFile {
            remote_path: remote,
            sha256: sha,
            uploaded,
        })
    }

    /// Downloads a remote file or directory (recursively) to `local`.
    /// A directory's contents land inside `local` (created if needed).
    pub async fn get(&self, remote: &str, local: &Path) -> Result<(), ExecError> {
        let remote = self.resolve(remote).await?;
        let q = shell_quote(&remote);
        let kind = self
            .run(
                "stat remote path",
                &format!(
                    "if [ -d {q} ]; then echo dir; elif [ -f {q} ]; then echo file; else echo missing; fi"
                ),
                Stdio::null(),
            )
            .await?;
        match kind.trim() {
            "file" => {
                if let Some(parent) = local.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                let mut part = local.as_os_str().to_owned();
                part.push(".part");
                let part = PathBuf::from(part);
                let file = std::fs::File::create(&part)?;
                let mut cmd = self.script_cmd(&format!("cat {q}"), Stdio::null())?;
                // Not `output()`: tokio's `output()` forces stdout back to a pipe.
                cmd.stdout(Stdio::from(file));
                let out = cmd.spawn()?.wait_with_output().await?;
                if let Err(e) = self.check_status(&format!("download {remote}"), out.status, &out.stderr) {
                    let _ = std::fs::remove_file(&part);
                    return Err(e);
                }
                std::fs::rename(&part, local)?;
                Ok(())
            }
            "dir" => {
                std::fs::create_dir_all(local)?;
                let mut ssh = self
                    .script_cmd(&format!("tar -C {q} -cf - ."), Stdio::null())?
                    .spawn()?;
                let ssh_out: Stdio = ssh
                    .stdout
                    .take()
                    .ok_or_else(|| self.err("ssh stdout unavailable"))?
                    .try_into()?;
                let tar = Command::new("tar")
                    .arg("-C")
                    .arg(local)
                    .args(["-xf", "-"])
                    .stdin(ssh_out)
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()?;
                let (ssh_res, tar_res) = tokio::join!(ssh.wait_with_output(), tar.wait_with_output());
                let (ssh_res, tar_res) = (ssh_res?, tar_res?);
                self.check_status(&format!("download {remote}"), ssh_res.status, &ssh_res.stderr)?;
                if !tar_res.status.success() {
                    return Err(self.err(format!(
                        "local tar failed: {}",
                        String::from_utf8_lossy(&tar_res.stderr).trim()
                    )));
                }
                Ok(())
            }
            _ => Err(self.err(format!("remote path {remote} does not exist"))),
        }
    }

    /// Prepares the machine and starts `tempera-agent connect` over
    /// `ssh -R <port>:<hub>`; returns once the agent registered with `hub`.
    ///
    /// Steps: [`check`](Self::check) → [`install_agent`](Self::install_agent)
    /// → (with `opts.worker_dir`) ensure `uv` and [`sync_worker`](Self::sync_worker)
    /// → tunnel + agent. The remote port is random in 20000..60000 and is
    /// retried when the remote side cannot bind it (`ExitOnForwardFailure`).
    pub async fn start(
        &self,
        hub: &Hub,
        hub_addr: SocketAddr,
        opts: &StartOpts,
    ) -> Result<SshSession, ExecError> {
        let info = self.check().await?;
        let local_agent = match &opts.agent_bin {
            Some(p) => p.clone(),
            None => crate::local::agent_binary().ok_or(ExecError::AgentBinaryNotFound)?,
        };
        let agent_path = self.install_agent_with(&info, &local_agent).await?;
        let mut info = info;
        let remote_worker_dir = match &opts.worker_dir {
            Some(dir) => {
                if info.uv.is_none() {
                    if !opts.install_uv {
                        return Err(self.err(info.python_ready().err().unwrap_or_default()));
                    }
                    tracing::info!(machine = %self.name, "installing uv");
                    info.uv = Some(self.install_uv().await?);
                }
                Some(self.sync_worker(dir).await?)
            }
            None => None,
        };
        if hub.connected().await.contains(&self.name) {
            return Err(self.err("an agent with this machine id is already connected to the hub"));
        }

        let mut last_error = String::new();
        for _ in 0..TUNNEL_ATTEMPTS {
            let port = random_port();
            let script = agent_script(
                &agent_path,
                port,
                &self.name,
                remote_worker_dir.as_deref(),
                &opts.agent_log,
            );
            let mut cmd = self.ssh()?;
            cmd.args(["-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveCountMax=3"])
                .arg("-R")
                .arg(forward_spec(port, hub_addr))
                .arg(self.destination())
                .arg(format!("sh -c {}", shell_quote(&script)))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            let mut child = cmd.spawn()?;
            let tail = Arc::new(Mutex::new(VecDeque::new()));
            let reader = child.stderr.take().map(|stderr| {
                let tail = tail.clone();
                let name = self.name.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(stderr).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        tracing::debug!(machine = %name, "ssh: {line}");
                        let mut t = tail.lock().await;
                        if t.len() == 40 {
                            t.pop_front();
                        }
                        t.push_back(line);
                    }
                })
            });
            tokio::select! {
                hw = hub.wait_for(&self.name, opts.timeout) => match hw {
                    Ok(hardware) => {
                        tracing::info!(machine = %self.name, port, "remote agent connected");
                        return Ok(SshSession {
                            machine_id: self.name.clone(),
                            remote_port: port,
                            hardware,
                            info,
                            agent_path,
                            remote_worker_dir,
                            child: Some(child),
                        });
                    }
                    Err(e) => {
                        let _ = child.kill().await;
                        if let Some(r) = reader {
                            let _ = r.await;
                        }
                        let t = tail.lock().await.iter().cloned().collect::<Vec<_>>().join("\n");
                        return Err(self.err(format!("{e}; ssh output:\n{t}")));
                    }
                },
                status = child.wait() => {
                    if let Some(r) = reader {
                        let _ = r.await;
                    }
                    let t = tail.lock().await.iter().cloned().collect::<Vec<_>>().join("\n");
                    let code = status.ok().and_then(|s| s.code());
                    if t.contains("remote port forwarding failed") || t.contains("forwarding failed for listen port") {
                        tracing::debug!(machine = %self.name, port, "remote port busy, retrying");
                        last_error = t;
                        continue;
                    }
                    return Err(self.err(format!(
                        "ssh exited (code {}) before the agent connected:\n{t}",
                        code.map_or_else(|| "signal".to_owned(), |c| c.to_string())
                    )));
                }
            }
        }
        Err(self.err(format!(
            "remote port forwarding failed {TUNNEL_ATTEMPTS} times (is AllowTcpForwarding enabled in sshd?):\n{last_error}"
        )))
    }
}

/// Quotes `s` as one word for a POSIX shell. Safe characters pass through;
/// anything else is single-quoted with `'` written as `'\''`.
pub fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, '_' | '-' | '.' | '/' | ':' | '=' | '+' | ',' | '@' | '%')
        })
    {
        return s.to_owned();
    }
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// The command run on the remote side inside the tunnel.
fn agent_script(agent: &str, port: u16, machine_id: &str, worker_dir: Option<&str>, log: &str) -> String {
    let mut s = format!(
        "{REMOTE_PATH}; RUST_LOG={log}; export RUST_LOG; exec {agent} connect --server http://127.0.0.1:{port} --machine-id {id}",
        log = shell_quote(log),
        agent = shell_quote(agent),
        id = shell_quote(machine_id),
    );
    if let Some(wd) = worker_dir {
        s.push_str(" --worker-dir ");
        s.push_str(&shell_quote(wd));
    }
    s
}

/// `-R` argument: remote loopback port → the hub's local address.
fn forward_spec(port: u16, hub: SocketAddr) -> String {
    let ip = match hub.ip() {
        ip if ip.is_unspecified() && ip.is_ipv4() => "127.0.0.1".to_owned(),
        ip if ip.is_unspecified() => "[::1]".to_owned(),
        std::net::IpAddr::V6(ip) => format!("[{ip}]"),
        ip => ip.to_string(),
    };
    format!("127.0.0.1:{port}:{ip}:{}", hub.port())
}

/// Shell snippet printing the sha256 of `quoted_path` (GNU or BSD tools).
fn sha256_cmd(quoted_path: &str) -> String {
    format!(
        "if command -v sha256sum >/dev/null 2>&1; then sha256sum {p}; else shasum -a 256 {p}; fi | cut -d ' ' -f 1",
        p = quoted_path
    )
}

fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) => "/",
        Some(i) => &path[..i],
        None => ".",
    }
}

fn sanitize_component(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.chars().all(|c| c == '.') {
        "worker".into()
    } else {
        s
    }
}

fn parse_remote_info(out: &str) -> RemoteInfo {
    let mut info = RemoteInfo::default();
    let some = |v: &str| (!v.trim().is_empty()).then(|| v.trim().to_owned());
    for line in out.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k {
            "os" => {
                info.os = match v.trim() {
                    "Linux" => "linux".into(),
                    "Darwin" => "macos".into(),
                    other => other.to_lowercase(),
                }
            }
            "arch" => {
                info.arch = match v.trim() {
                    "arm64" | "aarch64" => "aarch64".into(),
                    "amd64" | "x86_64" => "x86_64".into(),
                    other => other.into(),
                }
            }
            "libc" => {
                info.libc = if v.contains("musl") {
                    Some("musl".into())
                } else if v.contains("GLIBC") || v.contains("GNU libc") {
                    v.split_whitespace().last().map(|ver| format!("glibc {ver}"))
                } else {
                    None
                }
            }
            "home" => info.home = v.trim().to_owned(),
            "uv" => info.uv = some(v),
            "python3" => info.python3 = some(v),
            "downloader" => info.downloader = some(v),
            "disk_free_kb" => info.disk_free_bytes = v.trim().parse::<u64>().unwrap_or(0) * 1024,
            "agent_version" => info.agent_version = v.split_whitespace().last().map(str::to_owned),
            _ => {}
        }
    }
    info
}

/// Target `(os, arch)` of an ELF or Mach-O executable, when recognizable.
fn binary_platform(path: &Path) -> Option<(&'static str, &'static str)> {
    let mut head = [0u8; 20];
    std::fs::File::open(path).ok()?.read_exact(&mut head).ok()?;
    if head[..4] == *b"\x7fELF" {
        // e_machine, little-endian (both supported targets are LE).
        let arch = match u16::from_le_bytes([head[18], head[19]]) {
            0x3E => "x86_64",
            0xB7 => "aarch64",
            _ => return None,
        };
        return Some(("linux", arch));
    }
    // 64-bit Mach-O (little-endian magic), cputype follows the magic.
    if head[..4] == [0xCF, 0xFA, 0xED, 0xFE] {
        let arch = match u32::from_le_bytes([head[4], head[5], head[6], head[7]]) {
            0x0100_0007 => "x86_64",
            0x0100_000C => "aarch64",
            _ => return None,
        };
        return Some(("macos", arch));
    }
    None
}

/// Hex sha256 of a file (off the async runtime).
pub async fn sha256_file(path: &Path) -> Result<String, ExecError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || -> std::io::Result<String> {
        let mut f = std::fs::File::open(&path)?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Ok(hex::encode(h.finalize()))
    })
    .await
    .map_err(|e| ExecError::Io(std::io::Error::other(e)))?
    .map_err(ExecError::Io)
}

/// Content hash of a directory tree (relative paths + bytes), skipping
/// [`WORKER_EXCLUDES`]. Symlinks are hashed by their target path.
fn hash_dir(root: &Path) -> std::io::Result<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if WORKER_EXCLUDES.iter().any(|ex| name == *ex) {
                continue;
            }
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(root, &path, out)?;
            } else {
                out.push(
                    path.strip_prefix(root)
                        .map_or_else(|_| path.clone(), Path::to_path_buf),
                );
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(root, root, &mut files)?;
    files.sort();
    let mut h = Sha256::new();
    for rel in files {
        let full = root.join(&rel);
        h.update(rel.to_string_lossy().as_bytes());
        h.update([0]);
        let meta = std::fs::symlink_metadata(&full)?;
        if meta.file_type().is_symlink() {
            h.update(b"L");
            h.update(std::fs::read_link(&full)?.to_string_lossy().as_bytes());
        } else {
            let bytes = std::fs::read(&full)?;
            h.update((bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
        h.update([0]);
    }
    Ok(hex::encode(h.finalize()))
}

/// A pseudo-random port in [`TUNNEL_PORTS`] (no RNG dependency needed).
fn random_port() -> u16 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut x = nanos
        ^ (u64::from(std::process::id()) << 32)
        ^ COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // xorshift64*
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    let span = u64::from(TUNNEL_PORTS.end - TUNNEL_PORTS.start);
    TUNNEL_PORTS.start + (x % span) as u16
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Runs `sh -c "printf '%s' <quoted>"` and returns what the shell saw.
    fn roundtrip(s: &str) -> String {
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("printf '%s' {}", shell_quote(s)))
            .output()
            .unwrap();
        assert!(out.status.success());
        String::from_utf8(out.stdout).unwrap()
    }

    #[test]
    fn quote_passes_safe_words_through() {
        assert_eq!(shell_quote("abc"), "abc");
        assert_eq!(
            shell_quote("/home/u/.tempera-agent/bin/x"),
            "/home/u/.tempera-agent/bin/x"
        );
        assert_eq!(shell_quote("a=b,c:d@e%f+g"), "a=b,c:d@e%f+g");
    }

    #[test]
    fn quote_escapes_everything_else() {
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        assert_eq!(shell_quote("~/x"), "'~/x'");
        for s in [
            "",
            "a b",
            "it's",
            "'",
            "''",
            "$HOME",
            "`id`",
            "$(rm -rf /)",
            "a;b|c&d>e<f",
            "new\nline",
            "tab\there",
            "back\\slash",
            "\"double\"",
            "*?[]{}!#~",
            "ünïcödé ç",
        ] {
            assert_eq!(roundtrip(s), s, "round trip of {s:?}");
        }
    }

    #[test]
    fn agent_script_quotes_arguments() {
        let s = agent_script("/h/a b/tempera-agent", 23456, "pc-1", Some("/h/w's"), "info");
        assert!(s.contains("exec '/h/a b/tempera-agent' connect --server http://127.0.0.1:23456"));
        assert!(s.contains("--machine-id pc-1 --worker-dir '/h/w'\\''s'"));
        assert!(s.starts_with(REMOTE_PATH));
    }

    #[test]
    fn forward_spec_formats() {
        assert_eq!(
            forward_spec(30000, "127.0.0.1:7071".parse().unwrap()),
            "127.0.0.1:30000:127.0.0.1:7071"
        );
        assert_eq!(
            forward_spec(30000, "0.0.0.0:7071".parse().unwrap()),
            "127.0.0.1:30000:127.0.0.1:7071"
        );
        assert_eq!(
            forward_spec(30000, "[::1]:7071".parse().unwrap()),
            "127.0.0.1:30000:[::1]:7071"
        );
    }

    #[test]
    fn random_ports_stay_in_range() {
        for _ in 0..1000 {
            assert!(TUNNEL_PORTS.contains(&random_port()));
        }
    }

    #[test]
    fn validate_rejects_option_injection() {
        let mut m = SshMachine::new("pc", "host.lan");
        assert!(m.validate().is_ok());
        m.host = "-oProxyCommand=evil".into();
        assert!(m.validate().is_err());
        m.host = "host".into();
        m.user = Some("-x".into());
        assert!(m.validate().is_err());
        m.user = Some("bruno".into());
        m.name = "bad name".into();
        assert!(m.validate().is_err());
        m.name = "pc".into();
        m.ssh_options = vec!["-F/dev/null".into()];
        assert!(m.validate().is_err());
        m.ssh_options = vec!["StrictHostKeyChecking=accept-new".into()];
        assert!(m.validate().is_ok());
        assert_eq!(m.destination(), "bruno@host");
    }

    #[test]
    fn parses_check_output() {
        let out = "os=Linux\narch=aarch64\nlibc=ldd (Debian GLIBC 2.36-9+deb12u7) 2.36\nhome=/home/t\n\
                   uv=\npython3=/usr/bin/python3\ndownloader=/usr/bin/curl\ndisk_free_kb=2048\n\
                   agent_version=tempera-agent 0.1.0\n";
        let info = parse_remote_info(out);
        assert_eq!(info.os, "linux");
        assert_eq!(info.arch, "aarch64");
        assert_eq!(info.libc.as_deref(), Some("glibc 2.36"));
        assert_eq!(info.uv, None);
        assert_eq!(info.python3.as_deref(), Some("/usr/bin/python3"));
        assert_eq!(info.disk_free_bytes, 2048 * 1024);
        assert_eq!(info.agent_version.as_deref(), Some("0.1.0"));
        assert!(info.python_ready().unwrap_err().contains("install_uv"));
        let musl = parse_remote_info("os=Darwin\narch=arm64\nlibc=musl libc (x86_64)\n");
        assert_eq!((musl.os.as_str(), musl.arch.as_str()), ("macos", "aarch64"));
        assert_eq!(musl.libc.as_deref(), Some("musl"));
    }

    #[test]
    fn detects_own_binary_platform() {
        let exe = std::env::current_exe().unwrap();
        if let Some((os, arch)) = binary_platform(&exe) {
            assert_eq!(arch, std::env::consts::ARCH);
            assert_eq!(os, std::env::consts::OS);
        }
    }

    #[test]
    fn dir_hash_ignores_caches_and_tracks_content() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("pkg/__pycache__")).unwrap();
        std::fs::create_dir_all(tmp.path().join(".venv/lib")).unwrap();
        std::fs::write(tmp.path().join("pkg/a.py"), "x = 1").unwrap();
        let h1 = hash_dir(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("pkg/__pycache__/a.pyc"), "junk").unwrap();
        std::fs::write(tmp.path().join(".venv/lib/x"), "junk").unwrap();
        assert_eq!(hash_dir(tmp.path()).unwrap(), h1);
        std::fs::write(tmp.path().join("pkg/a.py"), "x = 2").unwrap();
        assert_ne!(hash_dir(tmp.path()).unwrap(), h1);
    }

    #[test]
    fn small_helpers() {
        assert_eq!(parent_dir("/a/b/c"), "/a/b");
        assert_eq!(parent_dir("/c"), "/");
        assert_eq!(sanitize_component("python"), "python");
        assert_eq!(sanitize_component("my worker"), "my_worker");
        assert_eq!(sanitize_component(".."), "worker");
    }
}

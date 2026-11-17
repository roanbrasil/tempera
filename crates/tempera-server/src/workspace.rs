//! Where Têmpera keeps its state: `$TEMPERA_HOME` or `./.tempera`.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
}

impl Workspace {
    pub fn discover() -> Self {
        let root = std::env::var_os("TEMPERA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".tempera"));
        Self { root }
    }

    pub fn db_path(&self) -> PathBuf {
        self.root.join("tempera.db")
    }

    pub fn job_dir(&self, job_id: &str) -> PathBuf {
        self.root.join("jobs").join(job_id)
    }
}

/// The Python worker project (`workers/python`), if it can be found:
/// `$TEMPERA_WORKER_DIR`, then ancestors of the executable and of the cwd.
pub fn worker_dir() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("TEMPERA_WORKER_DIR") {
        return Some(PathBuf::from(p));
    }
    let probe = |base: &Path| {
        base.ancestors()
            .map(|a| a.join("workers/python"))
            .find(|p| p.join("pyproject.toml").is_file())
    };
    std::env::current_exe()
        .ok()
        .and_then(|e| probe(&e))
        .or_else(|| std::env::current_dir().ok().and_then(|d| probe(&d)))
}

/// Whether Python backends can run: an explicit interpreter, or uv + the worker project.
pub fn python_available(worker: Option<&Path>) -> Result<(), String> {
    if std::env::var_os("TEMPERA_PYTHON").is_some() {
        return Ok(());
    }
    if worker.is_none() {
        return Err("Python worker project not found (set TEMPERA_WORKER_DIR)".into());
    }
    let has_uv = std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|p| p.join("uv").is_file()));
    if has_uv {
        Ok(())
    } else {
        Err("`uv` is not installed (https://docs.astral.sh/uv/)".into())
    }
}

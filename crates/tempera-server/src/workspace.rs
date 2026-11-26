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
/// `$TEMPERA_WORKER_DIR`, else a location fixed relative to the executable.
///
/// The current directory is never searched: running `tempera` inside a cloned repository or a
/// downloaded dataset folder must not execute a `workers/python` that happens to be there (#94).
pub fn worker_dir() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("TEMPERA_WORKER_DIR") {
        return Some(PathBuf::from(p));
    }
    worker_near(&std::env::current_exe().ok()?)
}

/// `workers/python` next to the executable (release archive), or at the root of the source
/// tree when the executable is a build output under `target/` (development). Nothing else:
/// an executable in `~/.cargo/bin` does not look in `~` or `/`.
fn worker_near(exe: &Path) -> Option<PathBuf> {
    let found = |root: &Path| {
        let p = root.join("workers").join("python");
        p.join("pyproject.toml").is_file().then_some(p)
    };
    let dir = exe.parent()?;
    found(dir).or_else(|| {
        dir.ancestors()
            .find(|a| a.file_name().is_some_and(|n| n == "target"))
            .and_then(Path::parent)
            .and_then(found)
    })
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn worker_at(root: &Path) -> PathBuf {
        let w = root.join("workers").join("python");
        std::fs::create_dir_all(&w).unwrap();
        std::fs::write(w.join("pyproject.toml"), "[project]\nname = \"x\"\n").unwrap();
        w
    }

    #[test]
    fn worker_is_found_beside_a_release_binary_or_above_target() {
        let tmp = tempfile::tempdir().unwrap();
        // Release archive: tempera-0.1.0/{tempera, workers/python}
        let release = tmp.path().join("tempera-0.1.0");
        let w = worker_at(&release);
        assert_eq!(worker_near(&release.join("tempera")), Some(w));
        // Source tree: repo/target/x86_64-unknown-linux-gnu/release/tempera
        let repo = tmp.path().join("repo");
        let w = worker_at(&repo);
        let exe = repo.join("target/x86_64-unknown-linux-gnu/release/tempera");
        assert_eq!(worker_near(&exe), Some(w));
    }

    /// #94: an executable installed elsewhere never picks up a `workers/python` found in a
    /// parent directory (a home folder, a cloned repository, a dataset folder).
    #[test]
    fn worker_is_not_found_in_unrelated_ancestors() {
        let tmp = tempfile::tempdir().unwrap();
        worker_at(tmp.path()); // e.g. ~/workers/python
        let exe = tmp.path().join(".cargo").join("bin").join("tempera");
        assert_eq!(worker_near(&exe), None);
    }
}

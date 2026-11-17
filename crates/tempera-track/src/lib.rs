//! Tracking store: jobs, runs, metrics, artifacts (SQLite; ADR-0005).
//!
//! Metrics arrive at high frequency, so [`Tracker::add_metrics`] writes a whole
//! batch in one transaction.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::{Row, migrate::Migrator};

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

#[derive(Debug, thiserror::Error)]
pub enum TrackError {
    #[error(transparent)]
    Db(#[from] sqlx::Error),
    #[error(transparent)]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("the test set of job {job} was already used by run {run}; it can only be evaluated once")]
    TestAlreadyUsed { job: String, run: String },
}

pub type Result<T> = std::result::Result<T, TrackError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRow {
    pub id: String,
    pub name: String,
    pub status: String,
    pub task: Option<String>,
    pub metric: Option<String>,
    pub dataset_id: Option<String>,
    pub winner_run_id: Option<String>,
    pub report_path: Option<String>,
    pub error: Option<String>,
    pub created_at: String,
    pub finished_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRow {
    pub id: String,
    pub job_id: String,
    pub algorithm: String,
    pub backend: String,
    pub is_baseline: bool,
    pub status: String,
    pub valid_score: Option<f64>,
    pub valid_metrics: BTreeMap<String, f64>,
    pub model_dir: Option<String>,
    pub error: Option<String>,
    pub duration_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub step: Option<i64>,
    pub value: f64,
    pub unix_ms: i64,
}

#[derive(Debug, Clone)]
pub struct Tracker {
    pool: SqlitePool,
}

pub fn now() -> String {
    // RFC 3339 in UTC without pulling a date crate.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Howard Hinnant's days → civil date.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

impl Tracker {
    /// Opens (and migrates) `<path>`, creating it if needed.
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(opts)
            .await?;
        MIGRATOR.run(&pool).await?;
        Ok(Self { pool })
    }

    pub async fn create_job(&self, id: &str, name: &str, spec_yaml: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO jobs (id, name, spec_yaml, status, created_at) VALUES (?, ?, ?, 'created', ?)",
        )
        .bind(id)
        .bind(name)
        .bind(spec_yaml)
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_job_context(
        &self,
        id: &str,
        dataset_id: &str,
        split_id: Option<&str>,
        task: &str,
        metric: &str,
    ) -> Result<()> {
        sqlx::query("UPDATE jobs SET dataset_id = ?, split_id = ?, task = ?, metric = ? WHERE id = ?")
            .bind(dataset_id)
            .bind(split_id)
            .bind(task)
            .bind(metric)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn set_job_status(&self, id: &str, status: &str, error: Option<&str>) -> Result<()> {
        let terminal = matches!(status, "succeeded" | "failed" | "blocked");
        sqlx::query("UPDATE jobs SET status = ?, error = ?, finished_at = CASE WHEN ? THEN ? ELSE finished_at END WHERE id = ?")
            .bind(status)
            .bind(error)
            .bind(terminal)
            .bind(now())
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn finish_job(
        &self,
        id: &str,
        winner: Option<&str>,
        test_json: Option<&str>,
        report: Option<&str>,
    ) -> Result<()> {
        sqlx::query("UPDATE jobs SET winner_run_id = ?, test_json = ?, report_path = ? WHERE id = ?")
            .bind(winner)
            .bind(test_json)
            .bind(report)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn start_run(
        &self,
        id: &str,
        job_id: &str,
        algorithm: &str,
        backend: &str,
        is_baseline: bool,
        params: &serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO runs (id, job_id, algorithm, backend, is_baseline, params_json, status, started_at) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
        )
        .bind(id)
        .bind(job_id)
        .bind(algorithm)
        .bind(backend)
        .bind(is_baseline)
        .bind(params.to_string())
        .bind(now())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn finish_run(
        &self,
        id: &str,
        status: &str,
        valid_score: Option<f64>,
        valid_metrics: &BTreeMap<String, f64>,
        model_dir: Option<&str>,
        error: Option<&str>,
        duration_s: f64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE runs SET status = ?, valid_score = ?, valid_metrics_json = ?, model_dir = ?, error = ?, finished_at = ?, duration_s = ? WHERE id = ?",
        )
        .bind(status)
        .bind(valid_score.filter(|v| v.is_finite()))
        .bind(serde_json::to_string(&finite(valid_metrics))?)
        .bind(model_dir)
        .bind(error)
        .bind(now())
        .bind(duration_s)
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Writes a batch of metric points in one transaction.
    pub async fn add_metrics(&self, run_id: &str, points: &[MetricPoint]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        for p in points.iter().filter(|p| p.value.is_finite()) {
            sqlx::query("INSERT INTO metrics (run_id, name, step, value, unix_ms) VALUES (?, ?, ?, ?, ?)")
                .bind(run_id)
                .bind(&p.name)
                .bind(p.step)
                .bind(p.value)
                .bind(p.unix_ms)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn add_artifact(&self, run_id: &str, kind: &str, format: &str, path: &str) -> Result<()> {
        sqlx::query("INSERT INTO artifacts (run_id, kind, format, path) VALUES (?, ?, ?, ?)")
            .bind(run_id)
            .bind(kind)
            .bind(format)
            .bind(path)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Records that `run_id` reads the job's test set. Fails if any run already did.
    pub async fn claim_test_set(&self, job_id: &str, run_id: &str) -> Result<()> {
        let res = sqlx::query(
            "INSERT INTO test_access (job_id, run_id, at) VALUES (?, ?, ?) ON CONFLICT(job_id) DO NOTHING",
        )
        .bind(job_id)
        .bind(run_id)
        .bind(now())
        .execute(&self.pool)
        .await?;
        if res.rows_affected() == 0 {
            let run: String = sqlx::query("SELECT run_id FROM test_access WHERE job_id = ?")
                .bind(job_id)
                .fetch_one(&self.pool)
                .await?
                .try_get("run_id")?;
            return Err(TrackError::TestAlreadyUsed {
                job: job_id.to_owned(),
                run,
            });
        }
        Ok(())
    }

    pub async fn jobs(&self, limit: i64) -> Result<Vec<JobRow>> {
        let rows = sqlx::query("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?")
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(job_row).collect()
    }

    /// Finds a job by full id or unique prefix.
    pub async fn job(&self, id_or_prefix: &str) -> Result<JobRow> {
        let rows = sqlx::query("SELECT * FROM jobs WHERE id LIKE ? || '%' LIMIT 2")
            .bind(id_or_prefix)
            .fetch_all(&self.pool)
            .await?;
        match rows.as_slice() {
            [one] => job_row(one),
            [] => Err(TrackError::NotFound(format!("job `{id_or_prefix}`"))),
            _ => Err(TrackError::NotFound(format!(
                "job prefix `{id_or_prefix}` is ambiguous"
            ))),
        }
    }

    pub async fn runs(&self, job_id: &str) -> Result<Vec<RunRow>> {
        let rows = sqlx::query("SELECT * FROM runs WHERE job_id = ? ORDER BY started_at, rowid")
            .bind(job_id)
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| {
                let metrics: Option<String> = r.try_get("valid_metrics_json")?;
                Ok(RunRow {
                    id: r.try_get("id")?,
                    job_id: r.try_get("job_id")?,
                    algorithm: r.try_get("algorithm")?,
                    backend: r.try_get("backend")?,
                    is_baseline: r.try_get("is_baseline")?,
                    status: r.try_get("status")?,
                    valid_score: r.try_get("valid_score")?,
                    valid_metrics: metrics
                        .map(|m| serde_json::from_str(&m))
                        .transpose()?
                        .unwrap_or_default(),
                    model_dir: r.try_get("model_dir")?,
                    error: r.try_get("error")?,
                    duration_s: r.try_get("duration_s")?,
                })
            })
            .collect()
    }

    pub async fn metrics(&self, run_id: &str, name: Option<&str>) -> Result<Vec<MetricPoint>> {
        let rows = sqlx::query(
            "SELECT name, step, value, unix_ms FROM metrics WHERE run_id = ? AND (? IS NULL OR name = ?) ORDER BY rowid",
        )
        .bind(run_id)
        .bind(name)
        .bind(name)
        .fetch_all(&self.pool)
        .await?;
        rows.iter()
            .map(|r| {
                Ok(MetricPoint {
                    name: r.try_get("name")?,
                    step: r.try_get("step")?,
                    value: r.try_get("value")?,
                    unix_ms: r.try_get("unix_ms")?,
                })
            })
            .collect()
    }

    pub async fn artifacts(&self, run_id: &str) -> Result<Vec<(String, String, String)>> {
        let rows = sqlx::query("SELECT kind, format, path FROM artifacts WHERE run_id = ? ORDER BY rowid")
            .bind(run_id)
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| Ok((r.try_get("kind")?, r.try_get("format")?, r.try_get("path")?)))
            .collect()
    }

    pub async fn spec_yaml(&self, job_id: &str) -> Result<String> {
        Ok(sqlx::query("SELECT spec_yaml FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_one(&self.pool)
            .await?
            .try_get("spec_yaml")?)
    }
}

fn finite(m: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    m.iter()
        .filter(|(_, v)| v.is_finite())
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

fn job_row(r: &sqlx::sqlite::SqliteRow) -> Result<JobRow> {
    Ok(JobRow {
        id: r.try_get("id")?,
        name: r.try_get("name")?,
        status: r.try_get("status")?,
        task: r.try_get("task")?,
        metric: r.try_get("metric")?,
        dataset_id: r.try_get("dataset_id")?,
        winner_run_id: r.try_get("winner_run_id")?,
        report_path: r.try_get("report_path")?,
        error: r.try_get("error")?,
        created_at: r.try_get("created_at")?,
        finished_at: r.try_get("finished_at")?,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    async fn tracker() -> (tempfile::TempDir, Tracker) {
        let tmp = tempfile::tempdir().unwrap();
        let t = Tracker::open(&tmp.path().join("t.db")).await.unwrap();
        (tmp, t)
    }

    #[tokio::test]
    async fn job_and_runs_round_trip() {
        let (_tmp, t) = tracker().await;
        t.create_job("job-abc", "churn", "version: 1").await.unwrap();
        t.start_run(
            "r1",
            "job-abc",
            "lightgbm",
            "python",
            false,
            &serde_json::json!({"lr": 0.1}),
        )
        .await
        .unwrap();
        let mut m = BTreeMap::new();
        m.insert("roc_auc".to_owned(), 0.9);
        m.insert("bad".to_owned(), f64::NAN);
        t.finish_run("r1", "succeeded", Some(0.9), &m, Some("/m"), None, 1.5)
            .await
            .unwrap();
        t.set_job_status("job-abc", "succeeded", None).await.unwrap();
        let job = t.job("job-a").await.unwrap();
        assert_eq!(job.status, "succeeded");
        assert!(job.finished_at.is_some());
        let runs = t.runs("job-abc").await.unwrap();
        assert_eq!(runs[0].valid_metrics.len(), 1);
        assert_eq!(runs[0].valid_score, Some(0.9));
        assert!(t.job("nope").await.is_err());
    }

    #[tokio::test]
    async fn metrics_batch_and_query() {
        let (_tmp, t) = tracker().await;
        t.create_job("j", "n", "y").await.unwrap();
        t.start_run("r", "j", "a", "native", false, &serde_json::json!({}))
            .await
            .unwrap();
        let pts: Vec<_> = (0..10_000)
            .map(|i| MetricPoint {
                name: "loss".into(),
                step: Some(i),
                value: 1.0 / (i + 1) as f64,
                unix_ms: i,
            })
            .collect();
        let start = std::time::Instant::now();
        t.add_metrics("r", &pts).await.unwrap();
        assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
        assert_eq!(t.metrics("r", Some("loss")).await.unwrap().len(), 10_000);
        assert!(t.metrics("r", Some("other")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_set_is_claimed_once() {
        let (_tmp, t) = tracker().await;
        t.create_job("j", "n", "y").await.unwrap();
        for r in ["r1", "r2"] {
            t.start_run(r, "j", "a", "native", false, &serde_json::json!({}))
                .await
                .unwrap();
        }
        t.claim_test_set("j", "r1").await.unwrap();
        let err = t.claim_test_set("j", "r2").await.unwrap_err();
        assert!(matches!(err, TrackError::TestAlreadyUsed { ref run, .. } if run == "r1"));
    }

    #[test]
    fn rfc3339_now_is_well_formed() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_000), (2024, 10, 4));
        let s = now();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z'));
    }
}

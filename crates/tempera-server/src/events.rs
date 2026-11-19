//! Live job events (Contracts.md §9): the orchestrator emits, the HTTP server
//! streams them over WebSocket. Every event is also kept so a client that
//! connects late (or reconnects) gets the full history first.
//!
//! The CLI never registers a job here, so `emit` is a cheap no-op there.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{broadcast, watch};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    Stage {
        stage: &'static str,
        message: String,
    },
    Audit {
        inference: Value,
        audit: Value,
    },
    Blocked {
        findings: Value,
    },
    RunStarted {
        run_id: String,
        algorithm: String,
        backend: String,
        index: usize,
        total: usize,
    },
    Metric {
        run_id: String,
        name: String,
        step: Option<i64>,
        value: f64,
    },
    Progress {
        run_id: String,
        step: u64,
        total_steps: Option<u64>,
        eta_s: Option<f64>,
        job_eta_s: Option<f64>,
    },
    Log {
        run_id: String,
        level: &'static str,
        message: String,
    },
    RunFinished {
        run_id: String,
        status: String,
        valid_score: Option<f64>,
        valid_metrics: Value,
        duration_s: f64,
        error: Option<String>,
    },
    Winner {
        run_id: String,
        algorithm: String,
        test: Value,
    },
    JobFinished {
        status: &'static str,
        report_url: Option<String>,
        error: Option<String>,
    },
}

impl JobEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::JobFinished { .. })
    }
}

#[derive(Debug)]
struct Channel {
    history: Vec<JobEvent>,
    tx: broadcast::Sender<JobEvent>,
    cancel: watch::Sender<bool>,
}

static JOBS: LazyLock<Mutex<HashMap<String, Arc<Mutex<Channel>>>>> = LazyLock::new(Default::default);

fn channel(job_id: &str) -> Option<Arc<Mutex<Channel>>> {
    JOBS.lock().ok()?.get(job_id).cloned()
}

/// Registers a job before it starts (server mode).
pub fn register(job_id: &str) {
    let (tx, _) = broadcast::channel(4096);
    let (cancel, _) = watch::channel(false);
    if let Ok(mut jobs) = JOBS.lock() {
        jobs.insert(
            job_id.to_owned(),
            Arc::new(Mutex::new(Channel {
                history: Vec::new(),
                tx,
                cancel,
            })),
        );
    }
}

pub fn emit(job_id: &str, event: JobEvent) {
    let Some(ch) = channel(job_id) else { return };
    let Ok(mut ch) = ch.lock() else { return };
    // Keep at most a bounded number of metric points in the replay buffer.
    if matches!(event, JobEvent::Metric { .. }) && ch.history.len() > 200_000 {
        return;
    }
    let _ = ch.tx.send(event.clone());
    ch.history.push(event);
}

/// History so far plus a receiver for what comes next (no gap, no duplicates).
pub fn subscribe(job_id: &str) -> Option<(Vec<JobEvent>, broadcast::Receiver<JobEvent>)> {
    let ch = channel(job_id)?;
    let ch = ch.lock().ok()?;
    Some((ch.history.clone(), ch.tx.subscribe()))
}

pub fn request_cancel(job_id: &str) -> bool {
    channel(job_id)
        .and_then(|ch| ch.lock().ok().map(|c| c.cancel.send_replace(true)))
        .is_some()
}

pub fn is_cancelled(job_id: &str) -> bool {
    channel(job_id)
        .and_then(|ch| ch.lock().ok().map(|c| *c.cancel.borrow()))
        .unwrap_or(false)
}

/// Resolves when the job is cancelled (never, for unregistered jobs).
pub async fn cancelled(job_id: &str) {
    let rx = channel(job_id).and_then(|ch| ch.lock().ok().map(|c| c.cancel.subscribe()));
    match rx {
        Some(mut rx) => {
            let _ = rx.wait_for(|c| *c).await;
        }
        None => std::future::pending::<()>().await,
    }
}

/// Remaining time of a run from its step timing: skips the first (warm-up)
/// step, then an exponential moving average of the seconds per step.
#[derive(Debug, Default)]
pub struct Eta {
    last: Option<(u64, std::time::Instant)>,
    ema: Option<f64>,
}

impl Eta {
    const ALPHA: f64 = 0.2;

    pub fn update(&mut self, step: u64, total: Option<u64>) -> Option<f64> {
        let now = std::time::Instant::now();
        if let Some((prev_step, prev_t)) = self.last
            && step > prev_step
        {
            let per_step = now.duration_since(prev_t).as_secs_f64() / (step - prev_step) as f64;
            self.ema = Some(
                self.ema
                    .map_or(per_step, |e| Self::ALPHA * per_step + (1.0 - Self::ALPHA) * e),
            );
        }
        self.last = Some((step, now));
        Some(self.ema? * total?.saturating_sub(step) as f64)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn replay_then_live() {
        register("j-test");
        emit(
            "j-test",
            JobEvent::Stage {
                stage: "ingest",
                message: "x".into(),
            },
        );
        let (history, mut rx) = subscribe("j-test").unwrap();
        assert_eq!(history.len(), 1);
        emit(
            "j-test",
            JobEvent::JobFinished {
                status: "succeeded",
                report_url: None,
                error: None,
            },
        );
        assert!(rx.try_recv().unwrap().is_terminal());
        assert!(!is_cancelled("j-test"));
        assert!(request_cancel("j-test"));
        assert!(is_cancelled("j-test"));
        assert!(subscribe("unknown").is_none());
        emit(
            "unknown",
            JobEvent::Stage {
                stage: "x",
                message: String::new(),
            },
        ); // no-op
    }

    #[test]
    fn eta_needs_two_points_and_a_total() {
        let mut e = Eta::default();
        assert_eq!(e.update(1, Some(10)), None);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let eta = e.update(2, Some(10)).unwrap();
        assert!(eta > 0.1 && eta < 2.0, "{eta}");
        assert_eq!(e.update(3, None), None);
    }

    #[test]
    fn events_serialize_with_type_tag() {
        let v = serde_json::to_value(JobEvent::Progress {
            run_id: "r".into(),
            step: 1,
            total_steps: Some(2),
            eta_s: Some(1.0),
            job_eta_s: None,
        })
        .unwrap();
        assert_eq!(v["type"], "progress");
    }
}

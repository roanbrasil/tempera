//! Cost guardrails (#57): a per-job budget meter, instance leases with
//! heartbeats, and the reaper that terminates any Têmpera instance without
//! a fresh lease (a crashed server, a killed CLI, a lost network).
//!
//! The lease is written *before* `launch` (keyed by instance name, which the
//! provider echoes back), so there is no window where an instance exists
//! without a lease. While a job runs, the server refreshes the heartbeat
//! every [`HEARTBEAT_S`]; anything silent for [`LEASE_TTL_S`] is reaped.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{InstanceState, Provider, Result};

pub const HEARTBEAT_S: u64 = 60;
/// A lease without a heartbeat for this long is dead. With the reaper running
/// every few minutes, a crash never leaves an instance up for more than ~15 min.
pub const LEASE_TTL_S: u64 = 10 * 60;

/// Spending state of one job.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum BudgetStatus {
    Ok {
        spent_usd: f64,
    },
    /// Crossed 80% of the cap (reported once).
    Alert {
        spent_usd: f64,
        limit_usd: f64,
    },
    /// Reached the cap: checkpoint and stop.
    Exceeded {
        spent_usd: f64,
        limit_usd: f64,
    },
}

#[derive(Debug, Clone)]
pub struct BudgetMeter {
    pub limit_usd: Option<f64>,
    pub usd_per_hour: f64,
    alerted: bool,
}

impl BudgetMeter {
    pub const ALERT_FRACTION: f64 = 0.8;

    pub fn new(limit_usd: Option<f64>, usd_per_hour: f64) -> Self {
        Self {
            limit_usd,
            usd_per_hour,
            alerted: false,
        }
    }

    pub fn spent(&self, elapsed_s: f64) -> f64 {
        self.usd_per_hour * elapsed_s.max(0.0) / 3600.0
    }

    /// Call periodically with the time the instance has been held.
    pub fn check(&mut self, elapsed_s: f64) -> BudgetStatus {
        let spent_usd = self.spent(elapsed_s);
        let Some(limit_usd) = self.limit_usd else {
            return BudgetStatus::Ok { spent_usd };
        };
        if spent_usd >= limit_usd {
            BudgetStatus::Exceeded { spent_usd, limit_usd }
        } else if spent_usd >= limit_usd * Self::ALERT_FRACTION && !self.alerted {
            self.alerted = true;
            BudgetStatus::Alert { spent_usd, limit_usd }
        } else {
            BudgetStatus::Ok { spent_usd }
        }
    }

    /// Seconds until the cap is hit (None without a cap or price).
    pub fn seconds_left(&self, elapsed_s: f64) -> Option<f64> {
        let limit = self.limit_usd?;
        (self.usd_per_hour > 0.0).then(|| (limit / self.usd_per_hour * 3600.0 - elapsed_s).max(0.0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lease {
    pub provider: String,
    /// Empty until `launch` returns.
    #[serde(default)]
    pub instance_id: String,
    pub instance_name: String,
    pub job_id: String,
    pub created_at: u64,
    pub heartbeat_at: u64,
}

impl Lease {
    pub fn is_fresh(&self, now: u64, ttl_s: u64) -> bool {
        now.saturating_sub(self.heartbeat_at) <= ttl_s
    }
}

/// `<home>/cloud/leases.json`.
#[derive(Debug, Clone)]
pub struct Leases {
    path: PathBuf,
}

impl Leases {
    pub fn new(home: &Path) -> Self {
        Self {
            path: home.join("cloud").join("leases.json"),
        }
    }

    pub fn load(&self) -> Vec<Lease> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    fn save(&self, leases: &[Lease]) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = self.path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(leases)?)?;
        std::fs::rename(tmp, &self.path)?;
        Ok(())
    }

    fn update(&self, f: impl FnOnce(&mut Vec<Lease>)) -> Result<()> {
        let mut l = self.load();
        f(&mut l);
        self.save(&l)
    }

    /// Records the intent to launch `instance_name` (before calling the provider).
    pub fn open(&self, provider: &str, instance_name: &str, job_id: &str, now: u64) -> Result<()> {
        self.update(|l| {
            l.retain(|x| !(x.provider == provider && x.instance_name == instance_name));
            l.push(Lease {
                provider: provider.to_owned(),
                instance_id: String::new(),
                instance_name: instance_name.to_owned(),
                job_id: job_id.to_owned(),
                created_at: now,
                heartbeat_at: now,
            });
        })
    }

    pub fn set_instance(
        &self,
        provider: &str,
        instance_name: &str,
        instance_id: &str,
        now: u64,
    ) -> Result<()> {
        self.update(|l| {
            for x in l
                .iter_mut()
                .filter(|x| x.provider == provider && x.instance_name == instance_name)
            {
                instance_id.clone_into(&mut x.instance_id);
                x.heartbeat_at = now;
            }
        })
    }

    pub fn heartbeat(&self, provider: &str, instance_name: &str, now: u64) -> Result<()> {
        self.update(|l| {
            for x in l
                .iter_mut()
                .filter(|x| x.provider == provider && x.instance_name == instance_name)
            {
                x.heartbeat_at = now;
            }
        })
    }

    pub fn close(&self, provider: &str, instance_name: &str) -> Result<()> {
        self.update(|l| l.retain(|x| !(x.provider == provider && x.instance_name == instance_name)))
    }
}

/// What the reaper did.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReapReport {
    /// `(id, name)` of terminated instances.
    pub terminated: Vec<(String, String)>,
    /// `(id, name, error)` of instances that could not be terminated.
    pub failed: Vec<(String, String, String)>,
    /// Tagged instances kept alive by a fresh lease.
    pub kept: usize,
    /// Dead leases removed.
    pub pruned_leases: usize,
}

/// Terminates every [`crate::TAG`]-named instance of `provider` without a
/// fresh lease, and drops leases that are stale or whose instance is gone.
pub fn reap(provider: &dyn Provider, leases: &Leases, now: u64, ttl_s: u64) -> Result<ReapReport> {
    let all = leases.load();
    let mine: Vec<&Lease> = all.iter().filter(|l| l.provider == provider.name()).collect();
    let mut report = ReapReport::default();
    let instances = provider.instances()?;
    for inst in instances
        .iter()
        .filter(|i| i.is_tempera() && i.state != InstanceState::Gone)
    {
        let alive = mine.iter().any(|l| {
            l.is_fresh(now, ttl_s)
                && (l.instance_name == inst.name || (!l.instance_id.is_empty() && l.instance_id == inst.id))
        });
        if alive {
            report.kept += 1;
            continue;
        }
        match provider.terminate(&inst.id) {
            Ok(()) => {
                tracing::warn!(provider = provider.name(), id = %inst.id, name = %inst.name, "reaped orphan instance");
                report.terminated.push((inst.id.clone(), inst.name.clone()));
            }
            Err(e) => report
                .failed
                .push((inst.id.clone(), inst.name.clone(), e.to_string())),
        }
    }
    let live_names: Vec<&str> = instances
        .iter()
        .filter(|i| i.state != InstanceState::Gone)
        .map(|i| i.name.as_str())
        .collect();
    let before = all.len();
    leases.update(|l| {
        l.retain(|x| {
            x.provider != provider.name()
                || (x.is_fresh(now, ttl_s)
                    && (x.instance_id.is_empty() || live_names.contains(&x.instance_name.as_str())))
        });
    })?;
    report.pruned_leases = before.saturating_sub(leases.load().len());
    Ok(report)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::mock::Mock;
    use crate::{BootSpec, ResourceRequest};

    #[test]
    fn budget_meter_alerts_once_then_stops() {
        let mut m = BudgetMeter::new(Some(1.0), 3.6); // $0.001/s
        assert!(matches!(m.check(100.0), BudgetStatus::Ok { .. }));
        assert!(matches!(m.check(850.0), BudgetStatus::Alert { .. }));
        assert!(
            matches!(m.check(900.0), BudgetStatus::Ok { .. }),
            "alert is reported once"
        );
        assert!(matches!(m.check(1000.0), BudgetStatus::Exceeded { .. }));
        assert!((m.seconds_left(400.0).unwrap() - 600.0).abs() < 1e-6);
        assert!(matches!(
            BudgetMeter::new(None, 9.0).check(1e9),
            BudgetStatus::Ok { .. }
        ));
    }

    #[test]
    fn reaper_kills_only_orphans() {
        let home = tempfile::tempdir().unwrap();
        let p = Mock::with_state(home.path().join("mock.json"));
        let leases = Leases::new(home.path());
        let offer = p.list_offers(&ResourceRequest::default()).unwrap().remove(0);

        // Job A: healthy lease. Job B: server crashed long ago. C: not ours.
        let a = BootSpec::for_job("a", "k");
        leases.open("mock", &a.name, "a", 1000).unwrap();
        let ia = p.launch(&offer, &a).unwrap();
        leases.set_instance("mock", &a.name, &ia, 1000).unwrap();
        let b = BootSpec::for_job("b", "k");
        leases.open("mock", &b.name, "b", 0).unwrap();
        p.launch(&offer, &b).unwrap();
        let mut c = BootSpec::for_job("c", "k");
        c.name = "someone-elses-pod".into();
        p.launch(&offer, &c).unwrap();
        // An instance with our tag and no lease at all (crash between launch and lease).
        p.launch(&offer, &BootSpec::for_job("d", "k")).unwrap();

        let r = reap(&p, &leases, 1000 + 60, LEASE_TTL_S).unwrap();
        let names: Vec<&str> = r.terminated.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["tempera-b", "tempera-d"]);
        assert_eq!(r.kept, 1);
        assert_eq!(r.pruned_leases, 1, "b's stale lease is dropped");
        let left: Vec<String> = p.instances().unwrap().into_iter().map(|i| i.name).collect();
        assert_eq!(left, ["tempera-a", "someone-elses-pod"]);

        // A's heartbeat stops: the next pass reaps it too.
        let r = reap(&p, &leases, 1000 + LEASE_TTL_S + 61, LEASE_TTL_S).unwrap();
        assert_eq!(r.terminated.len(), 1);
        assert!(leases.load().is_empty());
    }

    #[test]
    fn lease_lifecycle() {
        let home = tempfile::tempdir().unwrap();
        let l = Leases::new(home.path());
        l.open("mock", "tempera-x", "x", 10).unwrap();
        l.set_instance("mock", "tempera-x", "id-1", 20).unwrap();
        l.heartbeat("mock", "tempera-x", 30).unwrap();
        let v = l.load();
        assert_eq!((v[0].instance_id.as_str(), v[0].heartbeat_at), ("id-1", 30));
        l.close("mock", "tempera-x").unwrap();
        assert!(l.load().is_empty());
    }
}

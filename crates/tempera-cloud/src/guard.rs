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
/// A lease without a heartbeat for this long is dead. The reaper runs before every
/// rental, every few minutes under `tempera up`, and on `tempera cloud reap`; between
/// those, an orphan keeps billing (guarantees independent of a running Têmpera: #88).
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

/// The rental ledger: `<state>/rentals/<provider>__<instance>.json`, one file per rental.
///
/// It lives in the per-user state directory ([`crate::state_dir`]), not in a project's
/// workspace, so every project of the user sees every rental. One file per rental means
/// a heartbeat of one job can never overwrite the lease of another (#87); each write is
/// a rename of a uniquely named temp file, so readers never see a torn lease.
#[derive(Debug, Clone)]
pub struct Leases {
    dir: PathBuf,
}

static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Leases {
    pub fn new(state_dir: &Path) -> Self {
        Self {
            dir: state_dir.join("rentals"),
        }
    }

    fn file(&self, provider: &str, instance_name: &str) -> PathBuf {
        let clean = |s: &str| {
            s.chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect::<String>()
        };
        self.dir
            .join(format!("{}__{}.json", clean(provider), clean(instance_name)))
    }

    /// Every lease. An unreadable or corrupt lease is an **error**, never "no lease":
    /// the reaper must not terminate instances it cannot account for.
    pub fn load(&self) -> Result<Vec<Lease>> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e.into()),
        };
        let mut out = vec![];
        for entry in entries {
            let path = entry?.path();
            if path.extension().is_some_and(|x| x == "json") {
                let text = std::fs::read_to_string(&path)?;
                out.push(serde_json::from_str(&text).map_err(|e| {
                    crate::CloudError::Refused(format!("corrupt rental record {}: {e}", path.display()))
                })?);
            }
        }
        out.sort_by(|a: &Lease, b: &Lease| {
            a.created_at
                .cmp(&b.created_at)
                .then(a.instance_name.cmp(&b.instance_name))
        });
        Ok(out)
    }

    fn write(&self, lease: &Lease) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let path = self.file(&lease.provider, &lease.instance_name);
        let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = self.dir.join(format!(".{}.{seq}.tmp", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(lease)?)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    fn get(&self, provider: &str, instance_name: &str) -> Result<Option<Lease>> {
        match std::fs::read_to_string(self.file(provider, instance_name)) {
            Ok(t) => Ok(Some(serde_json::from_str(&t)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn modify(&self, provider: &str, instance_name: &str, f: impl FnOnce(&mut Lease)) -> Result<()> {
        if let Some(mut l) = self.get(provider, instance_name)? {
            f(&mut l);
            self.write(&l)?;
        }
        Ok(())
    }

    /// Records the intent to launch `instance_name` (before calling the provider).
    pub fn open(&self, provider: &str, instance_name: &str, job_id: &str, now: u64) -> Result<()> {
        self.write(&Lease {
            provider: provider.to_owned(),
            instance_id: String::new(),
            instance_name: instance_name.to_owned(),
            job_id: job_id.to_owned(),
            created_at: now,
            heartbeat_at: now,
        })
    }

    pub fn set_instance(
        &self,
        provider: &str,
        instance_name: &str,
        instance_id: &str,
        now: u64,
    ) -> Result<()> {
        self.modify(provider, instance_name, |x| {
            instance_id.clone_into(&mut x.instance_id);
            x.heartbeat_at = now;
        })
    }

    pub fn heartbeat(&self, provider: &str, instance_name: &str, now: u64) -> Result<()> {
        self.modify(provider, instance_name, |x| x.heartbeat_at = now)
    }

    pub fn close(&self, provider: &str, instance_name: &str) -> Result<()> {
        match std::fs::remove_file(self.file(provider, instance_name)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
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
    /// Têmpera-named instances of another owner (left alone).
    #[serde(default)]
    pub foreign: usize,
}

/// Terminates every instance **of this owner** (named `tempera-<owner>-…`) without a
/// fresh lease, and drops leases that are stale or whose instance is gone.
///
/// Fails closed: if the ledger cannot be read, nothing is terminated.
pub fn reap(
    provider: &dyn Provider,
    leases: &Leases,
    owner: &str,
    now: u64,
    ttl_s: u64,
) -> Result<ReapReport> {
    let all = leases.load()?;
    let mine: Vec<&Lease> = all.iter().filter(|l| l.provider == provider.name()).collect();
    let mut report = ReapReport::default();
    let instances = provider.instances()?;
    report.foreign = instances
        .iter()
        .filter(|i| i.is_tempera() && !i.is_owned_by(owner) && i.state != InstanceState::Gone)
        .count();
    for inst in instances
        .iter()
        .filter(|i| i.is_owned_by(owner) && i.state != InstanceState::Gone)
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
    for l in mine {
        let dead = !l.is_fresh(now, ttl_s)
            || (!l.instance_id.is_empty() && !live_names.contains(&l.instance_name.as_str()));
        if dead {
            leases.close(&l.provider, &l.instance_name)?;
            report.pruned_leases += 1;
        }
    }
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

    const ME: &str = "0000beef";

    #[test]
    fn reaper_kills_only_orphans() {
        let home = tempfile::tempdir().unwrap();
        let p = Mock::with_state(home.path().join("mock.json"));
        let leases = Leases::new(home.path());
        let offer = p.list_offers(&ResourceRequest::default()).unwrap().remove(0);

        // Job A: healthy lease. Job B: server crashed long ago. C: not ours.
        let a = BootSpec::for_job(ME, "a", "k");
        leases.open("mock", &a.name, "a", 1000).unwrap();
        let ia = p.launch(&offer, &a).unwrap();
        leases.set_instance("mock", &a.name, &ia, 1000).unwrap();
        let b = BootSpec::for_job(ME, "b", "k");
        leases.open("mock", &b.name, "b", 0).unwrap();
        p.launch(&offer, &b).unwrap();
        let mut c = BootSpec::for_job(ME, "c", "k");
        c.name = "someone-elses-pod".into();
        p.launch(&offer, &c).unwrap();
        // An instance with our name and no lease at all (crash between launch and lease).
        p.launch(&offer, &BootSpec::for_job(ME, "d", "k")).unwrap();

        let r = reap(&p, &leases, ME, 1000 + 60, LEASE_TTL_S).unwrap();
        let names: Vec<&str> = r.terminated.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["tempera-0000beef-b", "tempera-0000beef-d"]);
        assert_eq!(r.kept, 1);
        assert_eq!(r.pruned_leases, 1, "b's stale lease is dropped");
        let left: Vec<String> = p.instances().unwrap().into_iter().map(|i| i.name).collect();
        assert_eq!(left, ["tempera-0000beef-a", "someone-elses-pod"]);

        // A's heartbeat stops: the next pass reaps it too.
        let r = reap(&p, &leases, ME, 1000 + LEASE_TTL_S + 61, LEASE_TTL_S).unwrap();
        assert_eq!(r.terminated.len(), 1);
        assert!(leases.load().unwrap().is_empty());
    }

    /// #87: two projects (or a teammate) sharing one provider account never reap each other.
    #[test]
    fn reaper_leaves_other_owners_and_hand_made_instances_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let p = Mock::with_state(tmp.path().join("mock.json"));
        let offer = p.list_offers(&ResourceRequest::default()).unwrap().remove(0);
        let (mine, theirs) = (
            Leases::new(&tmp.path().join("me")),
            Leases::new(&tmp.path().join("them")),
        );

        let other = BootSpec::for_job("cafe0001", "training", "k");
        theirs.open("mock", &other.name, "training", 1000).unwrap();
        p.launch(&offer, &other).unwrap();
        for name in ["tempera-dev", "temperature-sweep", "tempera-legacy-job"] {
            let mut b = BootSpec::for_job(ME, "x", "k");
            b.name = name.into();
            p.launch(&offer, &b).unwrap();
        }

        let r = reap(&p, &mine, ME, 1000, LEASE_TTL_S).unwrap();
        assert!(r.terminated.is_empty(), "{r:?}");
        assert_eq!(r.foreign, 3, "tempera-* of other owners are counted, not touched");
        assert_eq!(p.instances().unwrap().len(), 4);
    }

    /// An unreadable ledger must stop the reaper, not look like "no leases".
    #[test]
    fn corrupt_ledger_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let p = Mock::with_state(tmp.path().join("mock.json"));
        let offer = p.list_offers(&ResourceRequest::default()).unwrap().remove(0);
        let leases = Leases::new(tmp.path());
        let a = BootSpec::for_job(ME, "a", "k");
        leases.open("mock", &a.name, "a", 1000).unwrap();
        p.launch(&offer, &a).unwrap();
        std::fs::write(
            tmp.path().join("rentals").join("mock__tempera-0000beef-zz.json"),
            "{torn",
        )
        .unwrap();

        assert!(reap(&p, &leases, ME, 1000, LEASE_TTL_S).is_err());
        assert_eq!(p.instances().unwrap().len(), 1, "nothing may be terminated");
    }

    /// Heartbeats of many jobs at once lose no lease (one file per rental).
    #[test]
    fn concurrent_writers_lose_no_lease() {
        let tmp = tempfile::tempdir().unwrap();
        let leases = Leases::new(tmp.path());
        std::thread::scope(|s| {
            for t in 0..16 {
                let l = leases.clone();
                s.spawn(move || {
                    let name = format!("tempera-{ME}-job{t}");
                    l.open("mock", &name, &format!("job{t}"), 1).unwrap();
                    for now in 2..30 {
                        l.heartbeat("mock", &name, now).unwrap();
                    }
                });
            }
        });
        let all = leases.load().unwrap();
        assert_eq!(all.len(), 16);
        assert!(all.iter().all(|l| l.heartbeat_at == 29));
    }

    #[test]
    fn owner_id_is_created_once_and_validated() {
        let tmp = tempfile::tempdir().unwrap();
        let a = crate::owner_id(tmp.path()).unwrap();
        assert_eq!(a.len(), 8);
        assert_eq!(crate::owner_id(tmp.path()).unwrap(), a);
        std::fs::write(tmp.path().join("owner"), "nope").unwrap();
        assert!(crate::owner_id(tmp.path()).is_err());
    }

    #[test]
    fn lease_lifecycle() {
        let home = tempfile::tempdir().unwrap();
        let l = Leases::new(home.path());
        l.open("mock", "tempera-x", "x", 10).unwrap();
        l.set_instance("mock", "tempera-x", "id-1", 20).unwrap();
        l.heartbeat("mock", "tempera-x", 30).unwrap();
        let v = l.load().unwrap();
        assert_eq!((v[0].instance_id.as_str(), v[0].heartbeat_at), ("id-1", 30));
        l.close("mock", "tempera-x").unwrap();
        assert!(l.load().unwrap().is_empty());
        l.close("mock", "tempera-x").unwrap();
    }
}

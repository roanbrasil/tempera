//! Running a job on a rented cloud machine (M6).
//!
//! 1. **Plan:** list the provider's offers (saved as a pricing snapshot),
//!    estimate ETA and cost on each (plus this machine when `local` is
//!    allowed), rank them by `compute.objective`, refuse when the best
//!    estimate exceeds `budget.max_cost`.
//! 2. **Launch:** write a lease *before* calling the provider, launch with a
//!    throwaway SSH key, wait until SSH answers. From there the job runs
//!    through the same SSH path as a LAN machine (`remote.rs`).
//! 3. **Watch:** a background task refreshes the lease heartbeat, meters the
//!    spend (alert at 80%, cancel at 100% of the budget) and notices a
//!    vanished (preempted) instance.
//! 4. **Always terminate:** `finish` on the happy path; `Drop` on every other
//!    path (errors, panics, cancel). The reaper (`tempera cloud reap`, and at
//!    `tempera up`) catches what a killed process could not clean up.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use tempera_cloud::estimate::{
    self, Estimate, MachineKind, MachineOption, Objective, Recommendation, Workload, fmt_duration, fmt_usd,
};
use tempera_cloud::guard::{BudgetMeter, BudgetStatus, HEARTBEAT_S, LEASE_TTL_S, Leases};
use tempera_cloud::{BootSpec, InstanceState, Offer, Provider, ResourceRequest, catalog, now_s};
use tempera_core::JobSpec;
use tempera_core::spec::ComputeObjective;
use tempera_exec::SshMachine;

use crate::candidates::Candidate;
use crate::events::{self, JobEvent};
use crate::ui;

/// How long a cloud instance may take to become reachable over SSH.
const BOOT_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// The cloud provider named in `compute.allow`, if any (SSH machines and
/// `local` take the other entries).
pub fn provider_in(spec: &JobSpec, is_registered_machine: impl Fn(&str) -> bool) -> Option<String> {
    spec.compute
        .allow
        .iter()
        .find(|n| tempera_cloud::is_provider_name(n) && !is_registered_machine(n))
        .cloned()
}

pub fn objective(spec: &JobSpec) -> Objective {
    match spec.compute.objective {
        ComputeObjective::Cheapest => Objective::Cheapest,
        ComputeObjective::Fastest => Objective::Fastest,
        ComputeObjective::CheapestUnder => Objective::CheapestUnder(
            spec.compute
                .deadline
                .map_or(2.0 * 3600.0, |d| d.as_duration().as_secs_f64()),
        ),
    }
}

/// Workload prior for the job's candidates.
pub fn workload(spec: &JobSpec, rows: u64, cols: u64, cands: &[Candidate]) -> Workload {
    let algos: Vec<&str> = cands.iter().map(|c| c.algorithm.as_str()).collect();
    let trials = spec.strategy.hpo.as_ref().map_or(0, |h| h.max_trials);
    estimate::tabular_workload(rows, cols, &algos, trials)
}

/// The budget cap in USD (cloud prices are in USD).
pub fn budget_usd(spec: &JobSpec) -> anyhow::Result<Option<f64>> {
    match &spec.budget.max_cost {
        None => Ok(None),
        Some(m) if m.currency.eq_ignore_ascii_case("USD") => Ok(Some(m.amount)),
        Some(m) => bail!(
            "budget.max_cost is in {}; cloud prices are in USD, so set the cap in USD for cloud jobs",
            m.currency
        ),
    }
}

/// This machine as a candidate (free, unless you count energy).
pub fn local_option() -> MachineOption {
    let hw = tempera_exec::system::hardware();
    let vram = hw.gpus.first().map(|g| g.vram_bytes as f64 / 1e9);
    MachineOption::own(
        "local",
        MachineKind::Local,
        hw.cpu_cores,
        hw.ram_bytes as f64 / 1e9,
        vram,
        None,
    )
}

/// The decision before anything is rented.
#[derive(Debug)]
pub struct Plan {
    pub provider: Arc<dyn Provider>,
    pub recommendation: Recommendation,
    pub workload: Workload,
    pub offers: Vec<Offer>,
    pub budget_usd: Option<f64>,
}

impl Plan {
    /// The cloud offer to rent, when a cloud machine won.
    pub fn chosen(&self) -> Option<(&Estimate, &Offer)> {
        let best = self.recommendation.best()?;
        best.machine.offer.as_ref().map(|o| (best, o))
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> anyhow::Result<T> {
    Ok(tokio::task::spawn_blocking(f).await?)
}

/// Lists offers, saves a pricing snapshot, ranks machines and prints the comparison.
pub async fn plan(
    home: &Path,
    spec: &JobSpec,
    provider_name: &str,
    workload: Workload,
    include_local: bool,
) -> anyhow::Result<Plan> {
    let name = provider_name.to_owned();
    let provider: Arc<dyn Provider> = Arc::from(blocking(move || tempera_cloud::provider(&name)).await??);
    let req = ResourceRequest {
        min_vram_gb: workload.needs_gpu.then_some(8.0),
        min_ram_gb: (workload.mem_gb * 1.2).max(4.0),
        min_disk_gb: 20.0,
        spot: spec.compute.spot,
        max_usd_per_hour: None,
        min_reliability: 0.98,
    };
    let p = provider.clone();
    let offers = blocking(move || p.list_offers(&req)).await??;
    let snap = catalog::Snapshot {
        provider: provider.name().to_owned(),
        fetched_at: now_s(),
        offers: offers.clone(),
    };
    if let Err(e) = catalog::save(home, &snap) {
        ui::detail(&format!("could not save the pricing snapshot: {e}"));
    }
    let mut machines: Vec<MachineOption> = offers.iter().map(MachineOption::from_offer).collect();
    if include_local {
        machines.push(local_option());
    }
    let recommendation = estimate::recommend(&workload, &machines, objective(spec));
    print_comparison(&recommendation);
    let budget_usd = budget_usd(spec)?;
    Ok(Plan {
        provider,
        recommendation,
        workload,
        offers,
        budget_usd,
    })
}

pub fn print_comparison(r: &Recommendation) {
    ui::step("Machines  estimated before renting anything");
    for e in r.ranked.iter().take(6) {
        let fit = match &e.reason {
            Some(why) => format!("  ✗ {why}"),
            None => String::new(),
        };
        ui::detail(&format!(
            "{:<38} {:>8}/h   ETA {:>7}   cost {:>7}{fit}",
            e.machine.name,
            fmt_usd(e.machine.usd_per_hour),
            fmt_duration(e.eta_s),
            fmt_usd(e.cost_usd),
        ));
    }
    ui::detail(&r.rationale);
}

/// Refuses to start when the estimate is over budget.
pub fn check_budget(plan: &Plan) -> anyhow::Result<()> {
    if let (Some(limit), Some((best, _))) = (plan.budget_usd, plan.chosen())
        && best.cost_usd > limit
    {
        bail!(
            "estimated cost {} on {} exceeds budget.max_cost {}; raise the budget, allow cheaper machines or reduce the search",
            fmt_usd(best.cost_usd),
            best.machine.name,
            fmt_usd(limit)
        );
    }
    Ok(())
}

/// A rented instance. Terminated by [`CloudInstance::finish`] or, failing that, on drop.
#[derive(Debug)]
pub struct CloudInstance {
    provider: Arc<dyn Provider>,
    leases: Leases,
    pub id: String,
    pub name: String,
    pub offer: Offer,
    pub estimated_usd: f64,
    launched: Instant,
    watcher: Option<tokio::task::JoinHandle<()>>,
    done: bool,
}

/// What the instance cost, for the report.
#[derive(Debug, Clone)]
pub struct CloudCost {
    pub machine: String,
    pub estimated_usd: f64,
    pub actual_usd: f64,
    pub savings: Option<estimate::Savings>,
}

fn throwaway_key(dir: &Path) -> anyhow::Result<(PathBuf, String)> {
    std::fs::create_dir_all(dir)?;
    let key = dir.join("id_ed25519");
    let _ = std::fs::remove_file(&key);
    let _ = std::fs::remove_file(key.with_extension("pub"));
    let out = std::process::Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-C", "tempera-cloud", "-f"])
        .arg(&key)
        .output()?;
    if !out.status.success() {
        bail!(
            "ssh-keygen failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let public = std::fs::read_to_string(key.with_extension("pub"))?;
    Ok((key, public.trim().to_owned()))
}

impl CloudInstance {
    /// Rents the planned offer and waits until it answers over SSH.
    pub async fn launch(home: &Path, job_id: &str, plan: &Plan) -> anyhow::Result<(Self, SshMachine)> {
        let (best, offer) = plan
            .chosen()
            .ok_or_else(|| anyhow!("no cloud offer was chosen"))?;
        let key_dir = home.join("cloud").join("keys").join(job_id);
        let (key, public) = throwaway_key(&key_dir)?;
        let boot = BootSpec::for_job(job_id, public);
        let leases = Leases::new(home);
        let provider_name = plan.provider.name().to_owned();

        // Clean up after earlier crashes before renting more.
        {
            let (p, l) = (plan.provider.clone(), leases.clone());
            match blocking(move || tempera_cloud::guard::reap(p.as_ref(), &l, now_s(), LEASE_TTL_S)).await? {
                Ok(r) if !r.terminated.is_empty() => {
                    ui::detail(&format!(
                        "reaped {} orphan instance(s) from earlier runs",
                        r.terminated.len()
                    ));
                }
                Ok(_) => {}
                Err(e) => ui::detail(&format!("reaper skipped: {e}")),
            }
        }

        leases.open(&provider_name, &boot.name, job_id, now_s())?;
        ui::step(&format!(
            "Renting   {} at {}/h (estimate {} for ~{})",
            offer.label(),
            fmt_usd(offer.usd_per_hour),
            fmt_usd(best.cost_usd),
            fmt_duration(best.eta_s)
        ));
        let (p, o, b) = (plan.provider.clone(), offer.clone(), boot.clone());
        let launched = Instant::now();
        let id = match blocking(move || p.launch(&o, &b)).await? {
            Ok(id) => id,
            Err(e) => {
                let _ = leases.close(&provider_name, &boot.name);
                return Err(e.into());
            }
        };
        leases.set_instance(&provider_name, &boot.name, &id, now_s())?;
        let mut inst = Self {
            provider: plan.provider.clone(),
            leases,
            id,
            name: boot.name.clone(),
            offer: offer.clone(),
            estimated_usd: best.cost_usd,
            launched,
            watcher: None,
            done: false,
        };
        events::emit(
            job_id,
            JobEvent::Cost {
                machine: offer.label(),
                usd_per_hour: offer.usd_per_hour,
                estimated_usd: best.cost_usd,
                spent_usd: 0.0,
                budget_usd: plan.budget_usd,
                status: "launched",
                savings: None,
            },
        );

        // Wait for an SSH endpoint, then for sshd itself.
        let deadline = Instant::now() + BOOT_TIMEOUT;
        let instance = loop {
            let (p, id) = (inst.provider.clone(), inst.id.clone());
            let st = blocking(move || p.status(&id)).await??;
            if st.ssh_ready() {
                break st;
            }
            if st.state == InstanceState::Gone {
                bail!("instance {} disappeared while booting", inst.id);
            }
            if Instant::now() > deadline || events::is_cancelled(job_id) {
                bail!(
                    "instance {} not reachable after {}",
                    inst.id,
                    fmt_duration(BOOT_TIMEOUT.as_secs_f64())
                );
            }
            let _ = inst.leases.heartbeat(inst.provider.name(), &inst.name, now_s());
            tokio::time::sleep(Duration::from_secs(5)).await;
        };
        let mut ssh = SshMachine::new(
            format!("{}-{}", inst.provider.name(), inst.id),
            instance.ssh_host.clone().unwrap_or_default(),
        );
        ssh.user = instance.ssh_user.clone().or(Some("root".into()));
        ssh.port = instance.ssh_port;
        ssh.identity = Some(key);
        ssh.ssh_options = vec![
            format!("UserKnownHostsFile={}", key_dir.join("known_hosts").display()),
            "StrictHostKeyChecking=accept-new".into(),
            "IdentityAgent=none".into(),
            "ConnectTimeout=10".into(),
        ];
        loop {
            match ssh.exec("true").await {
                Ok(_) => break,
                Err(e) if Instant::now() > deadline || events::is_cancelled(job_id) => {
                    bail!("SSH to {} never came up: {e}", inst.id)
                }
                Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
            }
        }
        ui::detail(&format!(
            "{} is up after {}",
            offer.label(),
            fmt_duration(inst.launched.elapsed().as_secs_f64())
        ));
        inst.watcher = Some(inst.spawn_watcher(job_id, plan.budget_usd));
        Ok((inst, ssh))
    }

    /// Heartbeat, budget meter and preemption detection while the job runs.
    fn spawn_watcher(&self, job_id: &str, budget_usd: Option<f64>) -> tokio::task::JoinHandle<()> {
        let (provider, leases, name, id) = (
            self.provider.clone(),
            self.leases.clone(),
            self.name.clone(),
            self.id.clone(),
        );
        let (job_id, launched, offer, estimated) = (
            job_id.to_owned(),
            self.launched,
            self.offer.clone(),
            self.estimated_usd,
        );
        tokio::spawn(async move {
            let mut meter = BudgetMeter::new(budget_usd, offer.usd_per_hour);
            let tick = Duration::from_secs(15);
            let mut since_heartbeat = Duration::ZERO;
            loop {
                tokio::time::sleep(tick).await;
                since_heartbeat += tick;
                let elapsed = launched.elapsed().as_secs_f64();
                let status = meter.check(elapsed);
                let label = match status {
                    BudgetStatus::Ok { .. } => "running",
                    BudgetStatus::Alert { spent_usd, limit_usd } => {
                        ui::detail(&format!(
                            "budget: {} of {} spent (80%)",
                            fmt_usd(spent_usd),
                            fmt_usd(limit_usd)
                        ));
                        "budget_alert"
                    }
                    BudgetStatus::Exceeded { spent_usd, limit_usd } => {
                        ui::detail(&format!(
                            "budget exhausted: {} of {}; stopping the job",
                            fmt_usd(spent_usd),
                            fmt_usd(limit_usd)
                        ));
                        events::request_cancel(&job_id);
                        "budget_exceeded"
                    }
                };
                events::emit(
                    &job_id,
                    JobEvent::Cost {
                        machine: offer.label(),
                        usd_per_hour: offer.usd_per_hour,
                        estimated_usd: estimated,
                        spent_usd: meter.spent(elapsed),
                        budget_usd,
                        status: label,
                        savings: None,
                    },
                );
                if since_heartbeat.as_secs() >= HEARTBEAT_S {
                    since_heartbeat = Duration::ZERO;
                    let _ = leases.heartbeat(provider.name(), &name, now_s());
                    let (p, i) = (provider.clone(), id.clone());
                    if let Ok(Ok(st)) = tokio::task::spawn_blocking(move || p.status(&i)).await
                        && st.state == InstanceState::Gone
                    {
                        ui::detail(&format!("instance {id} is gone (preempted?); stopping the job"));
                        events::request_cancel(&job_id);
                        break;
                    }
                }
            }
        })
    }

    fn spent(&self) -> f64 {
        self.offer.usd_per_hour * self.launched.elapsed().as_secs_f64() / 3600.0
    }

    /// Terminates the instance and returns what it cost.
    pub async fn finish(mut self, plan: &Plan, naive_multiplier: f64) -> CloudCost {
        if let Some(w) = self.watcher.take() {
            w.abort();
        }
        let actual = self.spent();
        let (p, id) = (self.provider.clone(), self.id.clone());
        match blocking(move || p.terminate(&id)).await {
            Ok(Ok(())) => {
                let _ = self.leases.close(self.provider.name(), &self.name);
                ui::detail(&format!(
                    "terminated {} ({} spent)",
                    self.offer.label(),
                    fmt_usd(actual)
                ));
            }
            // Keep the lease: it goes stale and the reaper retries.
            Ok(Err(e)) => ui::detail(&format!(
                "could not terminate {}: {e}; the reaper will retry",
                self.id
            )),
            Err(e) => ui::detail(&format!(
                "could not terminate {}: {e}; the reaper will retry",
                self.id
            )),
        }
        self.done = true;
        CloudCost {
            machine: self.offer.label(),
            estimated_usd: self.estimated_usd,
            actual_usd: actual,
            savings: estimate::savings(&plan.workload, &plan.offers, actual, naive_multiplier),
        }
    }
}

impl Drop for CloudInstance {
    fn drop(&mut self) {
        if let Some(w) = self.watcher.take() {
            w.abort();
        }
        if self.done {
            return;
        }
        // Error or cancel path: terminate synchronously on a plain thread
        // (no runtime needed), so the instance never outlives the job.
        let (p, l, id, name) = (
            self.provider.clone(),
            self.leases.clone(),
            self.id.clone(),
            self.name.clone(),
        );
        let joined = std::thread::spawn(move || {
            if p.terminate(&id).is_ok() {
                let _ = l.close(p.name(), &name);
                true
            } else {
                false
            }
        })
        .join();
        match joined {
            Ok(true) => ui::detail(&format!(
                "terminated {} after an interrupted job",
                self.offer.label()
            )),
            _ => ui::detail(&format!("could not terminate {}; the reaper will retry", self.id)),
        }
    }
}

/// Background reaper for `tempera up`: every few minutes, for every provider
/// whose API key is configured.
pub fn spawn_reaper(home: PathBuf) {
    tokio::spawn(async move {
        loop {
            for name in configured_providers() {
                let home = home.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if let Ok(p) = tempera_cloud::provider(name) {
                        match tempera_cloud::guard::reap(
                            p.as_ref(),
                            &Leases::new(&home),
                            now_s(),
                            LEASE_TTL_S,
                        ) {
                            Ok(r) if !r.terminated.is_empty() => {
                                tracing::warn!(
                                    provider = name,
                                    n = r.terminated.len(),
                                    "reaped orphan instances"
                                );
                            }
                            Ok(_) => {}
                            Err(e) => tracing::warn!(provider = name, "reaper: {e}"),
                        }
                    }
                })
                .await;
            }
            tokio::time::sleep(Duration::from_secs(4 * 60)).await;
        }
    });
}

/// Providers usable right now (API key present; mock when its state file is set).
pub fn configured_providers() -> Vec<&'static str> {
    let mut v = Vec::new();
    if std::env::var_os(tempera_cloud::runpod::KEY_ENV).is_some() {
        v.push("runpod");
    }
    if std::env::var_os(tempera_cloud::vast::KEY_ENV).is_some() {
        v.push("vast");
    }
    if std::env::var_os("TEMPERA_MOCK_CLOUD_STATE").is_some() {
        v.push("mock");
    }
    v
}

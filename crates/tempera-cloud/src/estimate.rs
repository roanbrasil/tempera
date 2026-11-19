//! Cost and ETA estimation + machine recommendation (#55), and the savings
//! line (#58). See docs/wiki/Cost-and-Estimation.md.
//!
//! The tabular workload prior is deliberately simple: seconds per million
//! cells per fit on one reference core, per algorithm, scaled to each machine
//! with Amdahl's law. It is a prior, not a promise; the live ETA (EMA of
//! real progress) takes over once the job runs, and calibrating the prior
//! from finished runs is tracked in the roadmap.

use serde::{Deserialize, Serialize};

use crate::Offer;

/// What a job will cost in compute, independent of the machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workload {
    /// Work on one reference CPU core, in seconds.
    pub core_seconds: f64,
    /// Fraction of the work that parallelizes across cores (Amdahl).
    pub parallel_fraction: f64,
    /// Speed-up a GPU gives this workload (1.0 = none).
    pub gpu_speedup: f64,
    pub needs_gpu: bool,
    /// Peak RAM.
    pub mem_gb: f64,
    /// Fixed per-run overhead (process start, imports), in seconds.
    pub overhead_s: f64,
}

/// Seconds per million cells (rows × columns) per fit on one reference core.
fn seconds_per_mcell(algorithm: &str) -> f64 {
    match algorithm {
        "linear_regression"
        | "ridge"
        | "lasso"
        | "elastic_net"
        | "logistic_regression"
        | "naive_bayes"
        | "majority"
        | "mean" => 0.5,
        "sgd" => 1.0,
        "decision_tree" => 2.0,
        "lightgbm" | "hist_gradient_boosting" | "kmeans" | "pca_kmeans" => 4.0,
        "xgboost" => 6.0,
        "isolation_forest" | "adaboost" => 5.0,
        "extra_trees" => 10.0,
        "catboost" => 12.0,
        "random_forest" => 15.0,
        "knn" => 3.0,
        "mlp" => 20.0,
        "svm" => 30.0,
        "dbscan" => 25.0,
        _ => 5.0,
    }
}

/// Native Rust algorithms start instantly; Python ones pay interpreter + import time.
fn per_fit_overhead_s(algorithm: &str) -> f64 {
    match algorithm {
        "linear_regression" | "ridge" | "logistic_regression" | "decision_tree" | "majority" | "mean" => 0.2,
        _ => 4.0,
    }
}

/// Prior for a tabular job: every candidate fitted once, plus `hpo_trials`
/// extra fits per tuned candidate at successive-halving cost (~36% of a full fit).
pub fn tabular_workload(rows: u64, cols: u64, algorithms: &[&str], hpo_trials: u32) -> Workload {
    let mcells = (rows as f64 * cols.max(1) as f64) / 1e6;
    let fits_per_algo = 1.0 + f64::from(hpo_trials) * 0.36;
    let mut core_seconds = 0.0;
    let mut overhead_s = 0.0;
    for a in algorithms {
        // SVM and DBSCAN are super-linear in rows.
        let rows_factor = if matches!(*a, "svm" | "dbscan") {
            (rows as f64 / 1e5).max(1.0)
        } else {
            1.0
        };
        core_seconds += seconds_per_mcell(a) * mcells * rows_factor * fits_per_algo;
        overhead_s += per_fit_overhead_s(a) * fits_per_algo;
    }
    // Profiling, audit and evaluation: a few passes over the data.
    core_seconds += 3.0 * mcells;
    Workload {
        core_seconds,
        parallel_fraction: 0.85,
        gpu_speedup: 1.0,
        needs_gpu: false,
        mem_gb: (rows as f64 * cols.max(1) as f64 * 8.0 * 4.0) / 1e9 + 1.0,
        overhead_s,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineKind {
    Local,
    Ssh,
    Cloud,
}

/// A place the job could run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MachineOption {
    pub name: String,
    pub kind: MachineKind,
    pub cpu_cores: u32,
    pub ram_gb: f64,
    /// VRAM of the GPU, if any.
    pub vram_gb: Option<f64>,
    /// Price while the machine is held (cloud) or energy cost (own machines).
    pub usd_per_hour: f64,
    /// Time before work can start (instance boot, agent install), seconds.
    pub boot_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offer: Option<Offer>,
}

/// Typical time for a cloud instance to boot, install the agent and sync data.
pub const CLOUD_BOOT_S: f64 = 180.0;
/// Agent install + data sync on an SSH machine.
pub const SSH_BOOT_S: f64 = 15.0;

impl MachineOption {
    pub fn from_offer(o: &Offer) -> Self {
        Self {
            name: o.label(),
            kind: MachineKind::Cloud,
            cpu_cores: o.cpu_cores.max(1),
            ram_gb: o.ram_gb,
            vram_gb: (o.gpu_count > 0 && o.vram_gb > 0.0).then_some(o.vram_gb),
            usd_per_hour: o.usd_per_hour,
            boot_s: CLOUD_BOOT_S,
            offer: Some(o.clone()),
        }
    }

    /// Your own machine: free unless you give its power draw and energy price.
    pub fn own(
        name: impl Into<String>,
        kind: MachineKind,
        cpu_cores: u32,
        ram_gb: f64,
        vram_gb: Option<f64>,
        energy: Option<(f64, f64)>,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            cpu_cores: cpu_cores.max(1),
            ram_gb,
            vram_gb,
            usd_per_hour: energy.map_or(0.0, |(watts, usd_per_kwh)| watts / 1000.0 * usd_per_kwh),
            boot_s: if kind == MachineKind::Ssh { SSH_BOOT_S } else { 0.0 },
            offer: None,
        }
    }
}

/// One machine, estimated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub machine: MachineOption,
    pub eta_s: f64,
    pub cost_usd: f64,
    pub fits: bool,
    /// Why it does not fit, when it does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

fn amdahl(cores: u32, p: f64) -> f64 {
    1.0 / ((1.0 - p) + p / f64::from(cores.max(1)))
}

pub fn estimate(w: &Workload, m: &MachineOption) -> Estimate {
    let mut reason = None;
    if m.ram_gb > 0.0 && m.ram_gb < w.mem_gb * 1.2 {
        reason = Some(format!(
            "needs ~{:.1} GB RAM, has {:.0} GB",
            w.mem_gb * 1.2,
            m.ram_gb
        ));
    }
    if w.needs_gpu && m.vram_gb.is_none() {
        reason = Some("needs a GPU".into());
    }
    let gpu = if m.vram_gb.is_some() {
        w.gpu_speedup.max(1.0)
    } else {
        1.0
    };
    let compute = w.core_seconds / amdahl(m.cpu_cores, w.parallel_fraction) / gpu;
    let eta_s = m.boot_s + w.overhead_s + compute;
    Estimate {
        cost_usd: m.usd_per_hour * eta_s / 3600.0,
        eta_s,
        fits: reason.is_none(),
        reason,
        machine: m.clone(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "objective", content = "deadline_s")]
pub enum Objective {
    Cheapest,
    Fastest,
    CheapestUnder(f64),
}

/// The ranking and the reason for the pick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    /// Fitting machines, best first, then the ones that do not fit.
    pub ranked: Vec<Estimate>,
    pub rationale: String,
}

impl Recommendation {
    pub fn best(&self) -> Option<&Estimate> {
        self.ranked.first().filter(|e| e.fits)
    }
}

pub fn fmt_duration(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
    }
}

pub fn fmt_usd(v: f64) -> String {
    if v == 0.0 {
        "$0".into()
    } else if v < 0.01 {
        "<$0.01".into()
    } else {
        format!("${v:.2}")
    }
}

pub fn recommend(w: &Workload, machines: &[MachineOption], objective: Objective) -> Recommendation {
    let all: Vec<Estimate> = machines.iter().map(|m| estimate(w, m)).collect();
    let (mut fit, unfit): (Vec<Estimate>, Vec<Estimate>) = all.into_iter().partition(|e| e.fits);
    let by_cost = |a: &Estimate, b: &Estimate| {
        a.cost_usd
            .total_cmp(&b.cost_usd)
            .then(a.eta_s.total_cmp(&b.eta_s))
    };
    let by_eta = |a: &Estimate, b: &Estimate| {
        a.eta_s
            .total_cmp(&b.eta_s)
            .then(a.cost_usd.total_cmp(&b.cost_usd))
    };
    let mut note = String::new();
    match objective {
        Objective::Cheapest => fit.sort_by(by_cost),
        Objective::Fastest => fit.sort_by(by_eta),
        Objective::CheapestUnder(deadline) => {
            let (mut ok, mut late): (Vec<Estimate>, Vec<Estimate>) =
                fit.into_iter().partition(|e| e.eta_s <= deadline);
            ok.sort_by(by_cost);
            late.sort_by(by_eta);
            if ok.is_empty() && !late.is_empty() {
                note = format!(
                    " No machine finishes within {}; picked the fastest.",
                    fmt_duration(deadline)
                );
            }
            ok.extend(late);
            fit = ok;
        }
    }
    let rationale = match (fit.first(), fit.get(1)) {
        (None, _) => "No machine fits this job.".to_owned(),
        (Some(b), None) => format!(
            "{}: {} in ~{} (only machine that fits).{note}",
            b.machine.name,
            fmt_usd(b.cost_usd),
            fmt_duration(b.eta_s)
        ),
        (Some(b), Some(r)) => {
            let speed = r.eta_s.max(1e-9) / b.eta_s.max(1e-9);
            let compare = if r.eta_s < b.eta_s * 0.95 {
                format!(
                    "{} is {:.1}x faster but costs {}",
                    r.machine.name,
                    b.eta_s / r.eta_s.max(1e-9),
                    fmt_usd(r.cost_usd)
                )
            } else if r.cost_usd < b.cost_usd && speed > 1.05 {
                format!(
                    "{} is cheaper ({}) but {:.1}x slower",
                    r.machine.name,
                    fmt_usd(r.cost_usd),
                    speed
                )
            } else {
                format!(
                    "next best: {} ({}, ~{})",
                    r.machine.name,
                    fmt_usd(r.cost_usd),
                    fmt_duration(r.eta_s)
                )
            };
            format!(
                "{}: {} in ~{}; {compare}.{note}",
                b.machine.name,
                fmt_usd(b.cost_usd),
                fmt_duration(b.eta_s)
            )
        }
    };
    fit.extend(unfit);
    Recommendation {
        ranked: fit,
        rationale,
    }
}

/// The "what you would have paid" baseline: the job on the largest on-demand
/// GPU among `offers`, running `naive_multiplier` times longer (no early
/// stopping, exhaustive search).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Savings {
    pub naive_machine: String,
    pub naive_cost_usd: f64,
    pub actual_cost_usd: f64,
    pub saved_usd: f64,
    pub saved_pct: f64,
}

impl Savings {
    pub fn line(&self) -> String {
        format!(
            "Savings vs. naive run ({}, on-demand, no early stopping): {} ({}%)",
            self.naive_machine,
            fmt_usd(self.saved_usd),
            // Never round a real cost up to "100% saved".
            if self.saved_pct > 99.0 && self.saved_pct < 100.0 {
                99
            } else {
                self.saved_pct.round() as i64
            }
        )
    }
}

pub fn savings(
    w: &Workload,
    offers: &[Offer],
    actual_cost_usd: f64,
    naive_multiplier: f64,
) -> Option<Savings> {
    let naive = offers
        .iter()
        .filter(|o| !o.spot && o.gpu_count > 0)
        .max_by(|a, b| {
            a.vram_gb
                .total_cmp(&b.vram_gb)
                .then(a.usd_per_hour.total_cmp(&b.usd_per_hour))
        })?;
    let e = estimate(w, &MachineOption::from_offer(naive));
    let naive_cost = naive.usd_per_hour * e.eta_s * naive_multiplier.max(1.0) / 3600.0;
    let saved = naive_cost - actual_cost_usd;
    Some(Savings {
        naive_machine: naive.label(),
        naive_cost_usd: naive_cost,
        actual_cost_usd,
        saved_usd: saved,
        saved_pct: if naive_cost > 0.0 {
            saved / naive_cost * 100.0
        } else {
            0.0
        },
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::mock::Mock;

    fn cloud() -> Vec<MachineOption> {
        Mock::catalog(0).iter().map(MachineOption::from_offer).collect()
    }

    #[test]
    fn workload_scales_with_data_and_algorithms() {
        let small = tabular_workload(10_000, 10, &["logistic_regression"], 0);
        let big = tabular_workload(1_000_000, 10, &["logistic_regression"], 0);
        let more = tabular_workload(10_000, 10, &["logistic_regression", "random_forest"], 0);
        let tuned = tabular_workload(10_000, 10, &["random_forest"], 20);
        assert!(big.core_seconds > 50.0 * small.core_seconds);
        assert!(more.core_seconds > small.core_seconds && more.overhead_s > small.overhead_s);
        assert!(tuned.core_seconds > tabular_workload(10_000, 10, &["random_forest"], 0).core_seconds * 5.0);
    }

    #[test]
    fn local_wins_cheapest_when_it_fits() {
        let w = tabular_workload(50_000, 20, &["lightgbm", "random_forest"], 10);
        let mut ms = cloud();
        ms.push(MachineOption::own(
            "local",
            MachineKind::Local,
            8,
            32.0,
            None,
            None,
        ));
        let r = recommend(&w, &ms, Objective::Cheapest);
        assert_eq!(r.best().unwrap().machine.name, "local");
        assert!(r.rationale.starts_with("local: $0"), "{}", r.rationale);
    }

    #[test]
    fn fastest_and_deadline_objectives() {
        let w = tabular_workload(2_000_000, 50, &["random_forest", "catboost"], 20);
        let mut ms = cloud();
        ms.push(MachineOption::own(
            "laptop",
            MachineKind::Local,
            2,
            16.0,
            None,
            None,
        ));
        let fastest = recommend(&w, &ms, Objective::Fastest);
        assert_ne!(fastest.best().unwrap().machine.name, "laptop");
        let laptop = fastest
            .ranked
            .iter()
            .find(|e| e.machine.name == "laptop")
            .unwrap();
        // A deadline shorter than the laptop's ETA forces a paid machine.
        let r = recommend(&w, &ms, Objective::CheapestUnder(laptop.eta_s / 2.0));
        let best = r.best().unwrap();
        assert!(
            best.eta_s <= laptop.eta_s / 2.0 && best.cost_usd > 0.0,
            "{}",
            r.rationale
        );
        // Impossible deadline → fastest, with a note.
        let r = recommend(&w, &ms, Objective::CheapestUnder(1.0));
        assert!(r.rationale.contains("No machine finishes"), "{}", r.rationale);
    }

    #[test]
    fn memory_and_gpu_filters() {
        let mut w = tabular_workload(10_000_000, 100, &["lightgbm"], 0); // ~33 GB
        let ms = cloud();
        let r = recommend(&w, &ms, Objective::Cheapest);
        assert!(
            r.ranked
                .iter()
                .any(|e| !e.fits && e.reason.as_deref().unwrap_or("").contains("RAM"))
        );
        w.needs_gpu = true;
        w.mem_gb = 1.0;
        let r = recommend(&w, &ms, Objective::Cheapest);
        assert!(r.best().unwrap().machine.vram_gb.is_some());
    }

    #[test]
    fn savings_against_largest_on_demand_gpu() {
        let w = tabular_workload(100_000, 20, &["lightgbm"], 10);
        let s = savings(&w, &Mock::catalog(0), 0.01, 2.8).unwrap();
        assert!(s.naive_machine.contains("A100"));
        assert!(s.saved_usd > 0.0 && s.saved_pct > 0.0 && s.saved_pct <= 100.0);
        assert!(s.line().starts_with("Savings vs. naive run"));
    }
}

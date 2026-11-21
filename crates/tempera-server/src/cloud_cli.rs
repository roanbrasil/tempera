//! `tempera cloud offers|sync|prices|estimate|instances|reap` (M6).
//!
//! API keys come only from the environment (`RUNPOD_API_KEY`, `VAST_API_KEY`)
//! and are never printed. Nothing here rents a machine: `tempera run` does,
//! after showing the same estimate `tempera cloud estimate` prints.

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::Subcommand;
use tempera_cloud::estimate::{fmt_duration, fmt_usd};
use tempera_cloud::guard::LEASE_TTL_S;
use tempera_cloud::{ResourceRequest, catalog, now_s};

use crate::candidates::candidates;
use crate::cloud;
use crate::orchestrator::load_spec;

#[derive(Debug, Subcommand)]
pub enum CloudCmd {
    /// List live offers (cheapest first).
    Offers {
        #[arg(long, default_value = "runpod")]
        provider: String,
        /// Substring of the GPU name, e.g. 4090.
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        max_price: Option<f64>,
        /// Minimum VRAM in GB (0 includes CPU-only offers).
        #[arg(long, default_value_t = 8.0)]
        min_vram: f64,
        /// Only on-demand (non-interruptible) offers.
        #[arg(long)]
        no_spot: bool,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Save a timestamped price snapshot (default: every provider with a key).
    Sync { providers: Vec<String> },
    /// Latest saved prices per provider, with their age.
    Prices { providers: Vec<String> },
    /// Estimate ETA and cost of a job on every machine, without renting anything.
    Estimate {
        #[arg(default_value = "tempera.yaml")]
        spec: PathBuf,
        /// Provider to price (default: the one in compute.allow).
        #[arg(long)]
        provider: Option<String>,
    },
    /// Instances in the provider account (Têmpera's are named `tempera-*`).
    Instances {
        #[arg(long, default_value = "runpod")]
        provider: String,
    },
    /// Terminate Têmpera instances without a live lease (orphans of crashes).
    Reap {
        /// Default: every provider with a key.
        providers: Vec<String>,
        /// A lease older than this (minutes) is dead.
        #[arg(long, default_value_t = LEASE_TTL_S / 60)]
        ttl_min: u64,
    },
}

fn or_configured(list: Vec<String>) -> anyhow::Result<Vec<String>> {
    if !list.is_empty() {
        return Ok(list);
    }
    let v: Vec<String> = cloud::configured_providers()
        .into_iter()
        .map(str::to_owned)
        .collect();
    if v.is_empty() {
        bail!("no provider configured: set RUNPOD_API_KEY or VAST_API_KEY (or name a provider)");
    }
    Ok(v)
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> anyhow::Result<T> {
    Ok(tokio::task::spawn_blocking(f).await?)
}

fn age(s: u64) -> String {
    fmt_duration(s as f64)
}

pub async fn run(home: &Path, cmd: CloudCmd) -> anyhow::Result<i32> {
    match cmd {
        CloudCmd::Offers {
            provider,
            gpu,
            max_price,
            min_vram,
            no_spot,
            limit,
        } => {
            let req = ResourceRequest {
                min_vram_gb: (min_vram > 0.0).then_some(min_vram),
                min_ram_gb: 0.0,
                min_disk_gb: 0.0,
                spot: !no_spot,
                max_usd_per_hour: max_price,
                min_reliability: 0.98,
            };
            let offers = blocking(move || -> anyhow::Result<_> {
                Ok(tempera_cloud::provider(&provider)?.list_offers(&req)?)
            })
            .await??;
            let gpu = gpu.map(|g| g.to_lowercase());
            let shown: Vec<_> = offers
                .iter()
                .filter(|o| gpu.as_ref().is_none_or(|g| o.gpu_name.to_lowercase().contains(g)))
                .take(limit)
                .collect();
            println!(
                "{:<34} {:>6} {:>6} {:>7} {:>9}  {:<6} id",
                "GPU", "VRAM", "vCPU", "RAM", "USD/h", "kind"
            );
            for o in &shown {
                println!(
                    "{:<34} {:>5.0}G {:>6} {:>6.0}G {:>9}  {:<6} {}",
                    format!(
                        "{}{}",
                        if o.gpu_count > 1 {
                            format!("{}x ", o.gpu_count)
                        } else {
                            String::new()
                        },
                        o.gpu_name
                    ),
                    o.vram_gb,
                    o.cpu_cores,
                    o.ram_gb,
                    format!("{:.3}", o.usd_per_hour),
                    if o.spot { "spot" } else { "on-dem" },
                    o.id
                );
            }
            println!("{} of {} offers", shown.len(), offers.len());
        }
        CloudCmd::Sync { providers } => {
            for name in or_configured(providers)? {
                let h = home.to_path_buf();
                let n = name.clone();
                let snap = blocking(move || -> anyhow::Result<_> {
                    let p = tempera_cloud::provider(&n)?;
                    Ok(catalog::sync(&h, p.as_ref())?)
                })
                .await?;
                match snap {
                    Ok(s) => println!(
                        "✓ {name}: {} offers saved (cheapest {})",
                        s.offers.len(),
                        s.cheapest().map_or("-".into(), |o| format!(
                            "{} {}",
                            o.label(),
                            fmt_usd(o.usd_per_hour)
                        ))
                    ),
                    Err(e) => println!("✗ {name}: {e}"),
                }
            }
        }
        CloudCmd::Prices { providers } => {
            let names = if providers.is_empty() {
                vec!["runpod".to_owned(), "vast".to_owned(), "mock".to_owned()]
            } else {
                providers
            };
            let now = now_s();
            for name in names {
                let hist = catalog::history(home, &name);
                let Some(s) = catalog::latest(home, &name) else {
                    if !hist.is_empty() {
                        println!("{name}: unreadable snapshot");
                    }
                    continue;
                };
                println!(
                    "{name}: {} offers, updated {} ago{} ({} snapshot(s) in history)",
                    s.offers.len(),
                    age(s.age_s(now)),
                    if s.is_stale(now) {
                        " — stale, run `tempera cloud sync`"
                    } else {
                        ""
                    },
                    hist.len()
                );
                let mut by_gpu: std::collections::BTreeMap<String, f64> = Default::default();
                for o in &s.offers {
                    let k = format!("{}{}", o.gpu_name, if o.spot { " (spot)" } else { "" });
                    let e = by_gpu.entry(k).or_insert(f64::INFINITY);
                    *e = e.min(o.usd_per_hour);
                }
                let mut rows: Vec<_> = by_gpu.into_iter().collect();
                rows.sort_by(|a, b| a.1.total_cmp(&b.1));
                for (gpu, price) in rows.into_iter().take(12) {
                    println!("  {gpu:<34} from {:.3} USD/h", price);
                }
            }
        }
        CloudCmd::Estimate { spec, provider } => {
            let (spec, _) = load_spec(&spec)?;
            let provider = provider
                .or_else(|| {
                    spec.compute
                        .allow
                        .iter()
                        .find(|n| tempera_cloud::is_provider_name(n))
                        .cloned()
                })
                .context("no provider: pass --provider or add one to compute.allow")?;
            let h = home.to_path_buf();
            let s2 = spec.clone();
            let (rows, cols, task) = blocking(move || -> anyhow::Result<_> {
                let ds = tempera_data::ingest(Path::new(&s2.data.source), &h)?;
                let profile = tempera_data::profile_dataset(&ds)?;
                let inference = tempera_data::infer_task(&s2, &profile)?;
                Ok((ds.rows, ds.columns.len() as u64, inference.task))
            })
            .await??;
            let cands = candidates(&spec, task).map_err(|e| anyhow::anyhow!("strategy.candidates: {e}"))?;
            let w = cloud::workload(&spec, rows, cols, &cands);
            println!(
                "{} candidates on {rows} rows × {cols} columns · ~{:.0} core-seconds · ~{:.1} GB RAM",
                cands.len(),
                w.core_seconds,
                w.mem_gb
            );
            let plan = cloud::plan(home, &spec, &provider, w, true).await?;
            if let Err(e) = cloud::check_budget(&plan) {
                println!("✗ {e}");
                return Ok(1);
            }
            let naive = if spec.strategy.hpo.is_some() {
                1.0 / 0.36
            } else {
                1.0
            };
            if let Some((best, _)) = plan.chosen()
                && let Some(s) =
                    tempera_cloud::estimate::savings(&plan.workload, &plan.offers, best.cost_usd, naive)
            {
                println!("{}", s.line());
            }
        }
        CloudCmd::Instances { provider } => {
            let list = blocking(move || -> anyhow::Result<_> {
                Ok(tempera_cloud::provider(&provider)?.instances()?)
            })
            .await??;
            let (ledger, owner) = crate::cloud::ledger()?;
            let leases = ledger.load()?;
            for i in &list {
                let leased = leases
                    .iter()
                    .any(|l| l.instance_name == i.name || l.instance_id == i.id);
                println!(
                    "{:<14} {:<28} {:<9} {:>8}/h  {}{}",
                    i.id,
                    i.name,
                    format!("{:?}", i.state).to_lowercase(),
                    i.usd_per_hour.map_or("-".into(), fmt_usd),
                    match (&i.ssh_host, i.ssh_port) {
                        (Some(h), Some(p)) => format!("ssh {h}:{p}"),
                        _ => String::new(),
                    },
                    match (i.is_owned_by(&owner), i.is_tempera(), leased) {
                        (true, _, true) => "  [tempera, leased]",
                        (true, _, false) => "  [tempera, ORPHAN]",
                        (false, true, _) => "  [tempera, another owner: not touched]",
                        _ => "",
                    }
                );
            }
            println!("{} instance(s)", list.len());
        }
        CloudCmd::Reap { providers, ttl_min } => {
            let mut code = 0;
            for name in or_configured(providers)? {
                let n = name.clone();
                let r = blocking(move || -> anyhow::Result<_> {
                    let p = tempera_cloud::provider(&n)?;
                    let (leases, owner) = crate::cloud::ledger()?;
                    Ok(tempera_cloud::guard::reap(
                        p.as_ref(),
                        &leases,
                        &owner,
                        now_s(),
                        ttl_min * 60,
                    )?)
                })
                .await?;
                match r {
                    Ok(r) => {
                        println!(
                            "{name}: terminated {}, kept {} with live leases, pruned {} dead lease(s)",
                            r.terminated.len(),
                            r.kept,
                            r.pruned_leases
                        );
                        for (id, n) in &r.terminated {
                            println!("  ✓ terminated {id} ({n})");
                        }
                        for (id, n, e) in &r.failed {
                            code = 1;
                            println!("  ✗ could not terminate {id} ({n}): {e}");
                        }
                    }
                    Err(e) => {
                        code = 1;
                        println!("✗ {name}: {e}");
                    }
                }
            }
            return Ok(code);
        }
    }
    Ok(0)
}

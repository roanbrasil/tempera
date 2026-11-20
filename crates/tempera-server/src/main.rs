//! `tempera`: command-line interface (M1). The HTTP server and UI come in M2.

mod builtin;
mod candidates;
mod cloud;
mod cloud_cli;
mod draft;
mod events;
mod hpo;
mod http;
mod machines_cli;
mod mcp;
mod orchestrator;
mod quality;
mod remote;
mod traces_cli;
mod ui;
mod workspace;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use tempera_core::JobSpec;
use workspace::Workspace;

#[derive(Debug, Parser)]
#[command(
    name = "tempera",
    version,
    about = "Train ML models the right way, on the cheapest machine that gets the job done"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create a tempera.yaml for a data file (asks for the target if not given).
    Init {
        /// CSV, Parquet or JSONL file.
        data: PathBuf,
        /// Column to predict.
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(short, long, default_value = "tempera.yaml")]
        output: PathBuf,
        /// Overwrite an existing spec.
        #[arg(long)]
        force: bool,
    },
    /// Ingest a data file and print its profile.
    Ingest { data: PathBuf },
    /// Run the data audit only (no training).
    Audit {
        #[arg(default_value = "tempera.yaml")]
        spec: PathBuf,
    },
    /// Audit, split, train every candidate, evaluate the winner once on the test set, write the report.
    Run {
        #[arg(default_value = "tempera.yaml")]
        spec: PathBuf,
    },
    /// List jobs, or show the runs of one job.
    Runs {
        /// Job id (or unique prefix).
        job: Option<String>,
        #[arg(long, default_value_t = 20)]
        limit: i64,
    },
    /// Print the path of a job's HTML report (the latest job by default).
    Report { job: Option<String> },
    /// Start the web UI and HTTP API (default http://127.0.0.1:7070).
    Up {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 7070)]
        port: u16,
    },
    /// Validate a tempera.yaml spec.
    Validate {
        #[arg(default_value = "tempera.yaml")]
        spec: PathBuf,
    },
    /// Print the JSON Schema of tempera.yaml.
    Schema,
    /// Manage worker machines reachable over SSH (<TEMPERA_HOME>/machines.yaml).
    Machines {
        #[command(subcommand)]
        action: machines_cli::MachinesCmd,
    },
    /// Cloud GPUs: offers, prices, estimates, instances and the orphan reaper.
    Cloud {
        #[command(subcommand)]
        action: cloud_cli::CloudCmd,
    },
    /// Agent traces (chat + tool calls) for LLM fine-tuning: profile and audit.
    Traces {
        #[command(subcommand)]
        action: traces_cli::TracesCmd,
    },
    /// MCP server on stdio, so agents (Claude Code, Hermes Agent, OpenClaw…) can use Têmpera.
    /// Jobs that rent paid cloud machines need TEMPERA_MCP_ALLOW_CLOUD=1.
    Mcp,
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let cli = Cli::parse();
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let code = match rt.block_on(dispatch(cli.command)) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("\nerror: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

async fn dispatch(cmd: Command) -> anyhow::Result<i32> {
    let ws = Workspace::discover();
    match cmd {
        Command::Validate { spec } => {
            let text =
                std::fs::read_to_string(&spec).with_context(|| format!("cannot read {}", spec.display()))?;
            match JobSpec::from_yaml(&text) {
                Ok(s) => println!("✓ {} is valid (job `{}`)", spec.display(), s.name),
                Err(e) => {
                    eprintln!("✗ {}\n{e}", spec.display());
                    return Ok(1);
                }
            }
        }
        Command::Schema => print!("{}", tempera_core::schema::job_spec_schema()),
        Command::Machines { action } => return machines_cli::run(&ws.root, action).await,
        Command::Cloud { action } => return cloud_cli::run(&ws.root, action).await,
        Command::Traces { action } => {
            return tokio::task::spawn_blocking(move || traces_cli::run(action)).await?;
        }
        Command::Mcp => tokio::task::spawn_blocking(|| mcp::serve(&mcp::SelfExe)).await??,
        Command::Up { host, port } => {
            let addr: std::net::SocketAddr = format!("{host}:{port}")
                .parse()
                .with_context(|| format!("invalid address {host}:{port}"))?;
            // Terminates cloud instances that crashed jobs left behind.
            cloud::spawn_reaper(ws.root.clone());
            http::serve(ws, addr).await?;
        }
        Command::Ingest { data } => {
            let ws2 = ws.clone();
            let (ds, profile) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
                let ds = tempera_data::ingest(&data, &ws2.root)?;
                let p = tempera_data::profile_dataset(&ds)?;
                Ok((ds, p))
            })
            .await??;
            ui::header(&format!("Dataset {}", &ds.id[..12]));
            ui::step(&format!(
                "{} rows × {} columns · {}",
                ui::count(ds.rows),
                ds.columns.len(),
                ds.dir.display()
            ));
            println!(
                "\n      {:<28} {:<10} {:<16} {:>7} {:>9}",
                "column", "type", "kind", "nulls", "distinct"
            );
            for c in &profile.columns {
                println!(
                    "      {:<28} {:<10} {:<16} {:>6.1}% {:>9}",
                    c.name,
                    c.dtype,
                    ui::snake(&c.kind),
                    c.null_fraction * 100.0,
                    ui::count(c.n_unique)
                );
            }
            println!();
        }
        Command::Init {
            data,
            target,
            name,
            output,
            force,
        } => init(&ws, &data, target, name, &output, force).await?,
        Command::Audit { spec } => {
            let (spec, _) = orchestrator::load_spec(&spec)?;
            ui::header(&format!("Audit {}", spec.name));
            let ws2 = ws.clone();
            let p = tokio::task::spawn_blocking(move || orchestrator::prepare(&ws2, spec)).await??;
            orchestrator::print_audit(&p.audit);
            if p.audit.blocked {
                ui::blocked(&p.audit);
                return Ok(2);
            }
            println!(
                "\n  ✓ Nothing blocks training. Details: {}\n",
                p.ds.dir.join("audit.json").display()
            );
        }
        Command::Run { spec } => {
            let out = orchestrator::run_job(&ws, &spec).await?;
            if out.blocked {
                return Ok(2);
            }
        }
        Command::Runs { job, limit } => runs(&ws, job, limit).await?,
        Command::Report { job } => {
            let t = tempera_track::Tracker::open(&ws.db_path()).await?;
            let row = match job {
                Some(j) => t.job(&j).await?,
                None => t
                    .jobs(1)
                    .await?
                    .into_iter()
                    .next()
                    .context("no jobs yet; run `tempera run`")?,
            };
            match row.report_path {
                Some(p) => println!("{p}"),
                None => bail!("job {} has no report (status: {})", row.id, row.status),
            }
        }
    }
    Ok(0)
}

async fn runs(ws: &Workspace, job: Option<String>, limit: i64) -> anyhow::Result<()> {
    let t = tempera_track::Tracker::open(&ws.db_path()).await?;
    match job {
        None => {
            println!(
                "{:<44} {:<10} {:<24} {:<10} created",
                "job", "status", "task", "metric"
            );
            for j in t.jobs(limit).await? {
                println!(
                    "{:<44} {:<10} {:<24} {:<10} {}",
                    j.id,
                    j.status,
                    j.task.unwrap_or_default(),
                    j.metric.unwrap_or_default(),
                    j.created_at
                );
            }
        }
        Some(id) => {
            let j = t.job(&id).await?;
            println!(
                "{} · {} · {} / {}",
                j.id,
                j.status,
                j.task.as_deref().unwrap_or("?"),
                j.metric.as_deref().unwrap_or("?")
            );
            if let Some(e) = &j.error {
                println!("error: {e}");
            }
            println!(
                "\n  {:<24} {:<8} {:<10} {:>10} {:>8}",
                "algorithm", "backend", "status", "valid", "time"
            );
            for r in t.runs(&j.id).await? {
                let mark = if j.winner_run_id.as_deref() == Some(r.id.as_str()) {
                    "★"
                } else {
                    " "
                };
                println!(
                    "{mark} {:<24} {:<8} {:<10} {:>10} {:>8}",
                    r.algorithm,
                    r.backend,
                    r.status,
                    r.valid_score.map_or("—".into(), |v| format!("{v:.4}")),
                    r.duration_s.map_or("—".into(), ui::secs)
                );
            }
            if let Some(p) = j.report_path {
                println!("\n  report: {p}");
            }
        }
    }
    Ok(())
}

async fn init(
    ws: &Workspace,
    data: &Path,
    target: Option<String>,
    name: Option<String>,
    output: &Path,
    force: bool,
) -> anyhow::Result<()> {
    if output.exists() && !force {
        bail!("{} already exists (use --force to overwrite)", output.display());
    }
    let (ws2, data2) = (ws.clone(), data.to_path_buf());
    let profile = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let ds = tempera_data::ingest(&data2, &ws2.root)?;
        Ok(tempera_data::profile_dataset(&ds)?)
    })
    .await??;
    let columns: Vec<&str> = profile.columns.iter().map(|c| c.name.as_str()).collect();
    let target = match target {
        Some(t) => t,
        None if std::io::stdin().is_terminal() => {
            println!("Columns: {}", columns.join(", "));
            print!("Which column do you want to predict? ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            std::io::stdin().lock().read_line(&mut line)?;
            line.trim().to_owned()
        }
        None => bail!("pass --target <column> (columns: {})", columns.join(", ")),
    };
    if !columns.contains(&target.as_str()) {
        bail!("`{target}` is not a column (columns: {})", columns.join(", "));
    }
    let name = name.unwrap_or_else(|| {
        data.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("job")
            .to_owned()
    });
    let base = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let yaml = draft::draft_yaml(&profile, &pathdiff(data, base), Some(&target), &name)?;
    std::fs::write(output, yaml)?;
    println!(
        "✓ wrote {} (target `{target}`). Next: tempera audit {0} && tempera run {0}",
        output.display()
    );
    Ok(())
}

/// `data` relative to `base` when possible, for portable specs.
fn pathdiff(data: &Path, base: &Path) -> String {
    let (Ok(d), Ok(b)) = (std::path::absolute(data), std::path::absolute(base)) else {
        return data.display().to_string();
    };
    match d.strip_prefix(&b) {
        Ok(rel) => format!("./{}", rel.display()),
        Err(_) => d.display().to_string(),
    }
}

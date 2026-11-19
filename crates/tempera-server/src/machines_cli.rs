//! `tempera machines add|list|check|remove`: the SSH machine registry in
//! `<TEMPERA_HOME>/machines.yaml` (docs/wiki/Execution-and-Machines.md, ADR-0007).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use clap::Subcommand;
use tempera_exec::machines::machines_path;
use tempera_exec::{Hub, MachineConfig, MachinesFile, StartOpts};

#[derive(Debug, Subcommand)]
pub enum MachinesCmd {
    /// Register (or, with --force, replace) a machine reachable over SSH.
    Add {
        name: String,
        /// Host name, IP or ~/.ssh/config alias.
        #[arg(long)]
        host: String,
        #[arg(long)]
        user: Option<String>,
        #[arg(long)]
        port: Option<u16>,
        /// Private key; without it ssh-agent and ~/.ssh/config are used.
        #[arg(long)]
        identity: Option<PathBuf>,
        /// Remote state directory (default ~/.tempera-agent).
        #[arg(long)]
        remote_dir: Option<String>,
        /// Free-form tag (repeatable), e.g. --tag gpu.
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// Replace an existing machine with the same name.
        #[arg(long)]
        force: bool,
    },
    /// List registered machines.
    List,
    /// Health check: installs the agent, runs it through an SSH tunnel and prints its hardware.
    Check {
        name: String,
        /// Also sync this Python worker project and verify uv.
        #[arg(long)]
        worker_dir: Option<PathBuf>,
        /// Install uv user-locally when missing (with --worker-dir).
        #[arg(long)]
        install_uv: bool,
    },
    /// Remove a machine from the registry (does not touch the remote side).
    Remove { name: String },
}

pub async fn run(home: &Path, cmd: MachinesCmd) -> anyhow::Result<i32> {
    let path = machines_path(home);
    let mut file = MachinesFile::load(&path)?;
    match cmd {
        MachinesCmd::Add {
            name,
            host,
            user,
            port,
            identity,
            remote_dir,
            tags,
            force,
        } => {
            if file.get(&name).is_some() && !force {
                bail!("machine `{name}` already exists (use --force to replace it)");
            }
            let mut m = MachineConfig::new(&name, host);
            m.user = user;
            m.port = port;
            m.identity = identity;
            m.remote_dir = remote_dir;
            m.tags = tags;
            m.to_ssh()?; // validate before saving
            let replaced = file.upsert(m);
            file.save(&path)?;
            println!(
                "✓ {} machine `{name}` in {}",
                if replaced { "replaced" } else { "added" },
                path.display()
            );
            println!("  next: tempera machines check {name}");
        }
        MachinesCmd::List => {
            if file.machines.is_empty() {
                println!("no machines yet; add one with `tempera machines add <name> --host H`");
                return Ok(0);
            }
            println!("{:<20} {:<32} {:<6} tags", "name", "destination", "port");
            for m in &file.machines {
                let dest = match &m.user {
                    Some(u) => format!("{u}@{}", m.host),
                    None => m.host.clone(),
                };
                println!(
                    "{:<20} {:<32} {:<6} {}",
                    m.name,
                    dest,
                    m.port.map_or_else(|| "-".to_owned(), |p| p.to_string()),
                    m.tags.join(",")
                );
            }
        }
        MachinesCmd::Remove { name } => {
            if file.remove(&name).is_none() {
                bail!("no machine named `{name}`");
            }
            file.save(&path)?;
            println!("✓ removed `{name}`");
        }
        MachinesCmd::Check {
            name,
            worker_dir,
            install_uv,
        } => return check(&file, &name, worker_dir, install_uv).await,
    }
    Ok(0)
}

async fn check(
    file: &MachinesFile,
    name: &str,
    worker_dir: Option<PathBuf>,
    install_uv: bool,
) -> anyhow::Result<i32> {
    let cfg = file
        .get(name)
        .with_context(|| format!("no machine named `{name}` (see `tempera machines list`)"))?;
    let ssh = cfg.to_ssh()?;
    println!("Checking {} ({})", name, ssh.destination());
    let info = ssh.check().await?;
    println!(
        "  system   {}/{} · {}",
        info.os,
        info.arch,
        info.libc.as_deref().unwrap_or("libc unknown")
    );
    println!(
        "  disk     {:.1} GiB free in {}",
        gib(info.disk_free_bytes),
        info.home
    );
    println!("  uv       {}", info.uv.as_deref().unwrap_or("not installed"));
    println!(
        "  agent    {}",
        info.agent_version.as_deref().unwrap_or("not installed yet")
    );

    let hub = Hub::new();
    let (addr, server) = hub.serve("127.0.0.1:0".parse()?).await?;
    let opts = StartOpts {
        worker_dir,
        install_uv,
        timeout: Duration::from_secs(60),
        ..Default::default()
    };
    let session = ssh.start(&hub, addr, &opts).await;
    let session = match session {
        Ok(s) => s,
        Err(e) => {
            server.abort();
            return Err(e.into());
        }
    };
    let hw = &session.hardware;
    println!(
        "  ✓ agent connected through the SSH tunnel ({})",
        session.agent_path
    );
    println!("  cpu      {} cores", hw.cpu_cores);
    println!("  ram      {:.1} GiB", gib(hw.ram_bytes));
    if hw.gpus.is_empty() {
        println!("  gpu      none");
    }
    for g in &hw.gpus {
        println!(
            "  gpu      {} · {:.1} GiB · driver {}",
            g.name,
            gib(g.vram_bytes),
            g.driver_version
        );
    }
    println!(
        "  python   {}",
        if hw.python_available {
            "available"
        } else {
            "not found"
        }
    );
    if let Some(w) = &session.remote_worker_dir {
        println!("  worker   {w}");
    }
    let warnings = session.info.warnings();
    session.stop(&hub).await;
    server.abort();
    for w in &warnings {
        println!("  ! {w}");
    }
    Ok(0)
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

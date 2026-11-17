//! `tempera-agent`: runs on a worker machine, dials the server and executes
//! jobs as backend subprocesses. See docs/wiki/Execution-and-Machines.md.

use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, about = "Têmpera worker agent")]
struct Args {
    /// Server gRPC endpoint, e.g. http://127.0.0.1:7071
    #[arg(long, env = "TEMPERA_SERVER")]
    server: Option<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args = Args::parse();
    tracing::info!(
        protocol = tempera_proto::PROTOCOL_VERSION,
        server = ?args.server,
        "tempera-agent starting (not implemented yet; see M1)"
    );
    Ok(())
}

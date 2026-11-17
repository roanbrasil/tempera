//! `tempera`: the server and CLI. For now only spec validation is wired up.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tempera_core::JobSpec;

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
    /// Validate a tempera.yaml spec.
    Validate {
        #[arg(default_value = "tempera.yaml")]
        spec: PathBuf,
    },
    /// Print the JSON Schema of tempera.yaml.
    Schema,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    match Cli::parse().command {
        Command::Validate { spec } => {
            let text = std::fs::read_to_string(&spec)
                .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", spec.display()))?;
            match JobSpec::from_yaml(&text) {
                Ok(s) => println!("✓ {} is valid (job `{}`)", spec.display(), s.name),
                Err(e) => {
                    eprintln!("✗ {}\n{e}", spec.display());
                    std::process::exit(1);
                }
            }
        }
        Command::Schema => print!("{}", tempera_core::schema::job_spec_schema()),
    }
    Ok(())
}

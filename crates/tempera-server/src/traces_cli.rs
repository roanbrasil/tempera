//! `tempera traces`: profile and audit agent traces before fine-tuning (#66, #67).

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::Subcommand;
use tempera_audit::Severity;
use tempera_audit::traces::{Benchmark, TraceAudit, audit_traces};
use tempera_data::traces::{read_traces, split_by_group};

use crate::ui;

#[derive(Debug, Subcommand)]
pub enum TracesCmd {
    /// Audit a JSONL file of conversations (OpenAI or ShareGPT/Hermes format).
    Audit {
        data: PathBuf,
        /// Benchmark you will report on, as NAME=prompts.jsonl (repeatable). Each line is a
        /// string, or an object with `prompt`, `question` or `text`, or a conversation.
        #[arg(long = "benchmark", value_name = "NAME=FILE")]
        benchmarks: Vec<String>,
        #[arg(long, default_value_t = 0.1)]
        valid: f64,
        #[arg(long, default_value_t = 0.2)]
        test: f64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Finding ids accepted by the user (repeatable), e.g. T6.
        #[arg(long = "acknowledge", value_name = "ID")]
        acknowledge: Vec<String>,
        /// Print the audit as JSON.
        #[arg(long)]
        json: bool,
    },
}

pub fn run(cmd: TracesCmd) -> anyhow::Result<i32> {
    match cmd {
        TracesCmd::Audit {
            data,
            benchmarks,
            valid,
            test,
            seed,
            acknowledge,
            json,
        } => {
            if !(0.0..1.0).contains(&valid) || !(0.0..1.0).contains(&test) || valid + test >= 1.0 {
                bail!("--valid and --test must be fractions that sum to less than 1");
            }
            let set = read_traces(&data)?;
            let bench = benchmarks
                .iter()
                .map(|b| load_benchmark(b))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let split = split_by_group(&set.conversations, valid, test, seed);
            let audit = audit_traces(&set, &split, &bench, &acknowledge);
            if json {
                println!("{}", serde_json::to_string_pretty(&audit)?);
            } else {
                print(&data, &audit);
            }
            Ok(if audit.blocked { 2 } else { 0 })
        }
    }
}

fn load_benchmark(arg: &str) -> anyhow::Result<Benchmark> {
    let (name, path) = arg
        .split_once('=')
        .with_context(|| format!("--benchmark expects NAME=FILE, got `{arg}`"))?;
    let text = std::fs::read_to_string(Path::new(path)).with_context(|| format!("cannot read {path}"))?;
    let mut prompts = vec![];
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or(serde_json::Value::String(line.into()));
        let p = match &v {
            serde_json::Value::String(s) => Some(s.clone()),
            _ => ["prompt", "question", "text", "query"]
                .iter()
                .find_map(|k| v.get(k).and_then(|x| x.as_str()).map(str::to_owned))
                .or_else(|| {
                    tempera_data::traces::parse_line(line, 0)
                        .ok()
                        .and_then(|c| c.first_user().map(str::to_owned))
                }),
        };
        prompts.extend(p);
    }
    if prompts.is_empty() {
        bail!("benchmark `{name}`: no prompts found in {path}");
    }
    Ok(Benchmark {
        name: name.to_owned(),
        prompts,
    })
}

fn print(data: &Path, a: &TraceAudit) {
    let p = &a.profile;
    ui::header(&format!("Traces {}", data.display()));
    ui::step(&format!(
        "{} conversations · {} groups · {:.1} turns avg · {} tool calls in {} conversations",
        ui::count(p.conversations as u64),
        ui::count(p.groups as u64),
        p.mean_turns,
        ui::count(p.tool_calls as u64),
        ui::count(p.with_tool_calls as u64),
    ));
    if !p.tools_used.is_empty() {
        let top: Vec<String> = p
            .tools_used
            .iter()
            .take(6)
            .map(|(n, c)| format!("{n} ({c})"))
            .collect();
        ui::detail(&format!("tools: {}", top.join(", ")));
    }
    let get = |k: &str| a.split.get(k).copied().unwrap_or(0);
    ui::detail(&format!(
        "split by group: train {} · valid {} · test {}",
        get("train"),
        get("valid"),
        get("test")
    ));
    let reds = a.findings.iter().filter(|f| f.severity == Severity::Red).count();
    let yellows = a
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Yellow)
        .count();
    ui::step(&format!("Audit     {reds} red · {yellows} yellow"));
    for f in &a.findings {
        ui::finding(f);
    }
    if a.blocked {
        println!(
            "\n  ✗ Fine-tuning blocked by the trace audit. Fix the data, or accept a finding with \
             --acknowledge <ID>.\n"
        );
    } else {
        println!("\n  ✓ Nothing blocks fine-tuning.\n");
    }
}

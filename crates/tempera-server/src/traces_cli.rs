//! `tempera traces`: profile and audit agent traces (#66, #67), then fine-tune a small
//! chat LLM on them with LoRA/QLoRA (#47). Fine-tuning always runs the audit first and
//! refuses to start while a red finding is unresolved, exactly like `tempera run`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, bail};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::{Value, json};
use tempera_audit::Severity;
use tempera_audit::traces::{Benchmark, TraceAudit, audit_traces, schema_violation};
use tempera_data::split::Partition;
use tempera_data::traces::{
    Conversation, TraceSet, read_traces, split_by_group, training_record, untrainable,
};

use crate::ui;
use crate::workspace::{Workspace, python_available, worker_dir};

#[derive(Debug, Args)]
pub struct AuditArgs {
    /// JSONL file of conversations (OpenAI or ShareGPT/Hermes format).
    pub data: PathBuf,
    /// Benchmark you will report on, as NAME=prompts.jsonl (repeatable). Each line is a
    /// string, or an object with `prompt`, `question` or `text`, or a conversation.
    #[arg(long = "benchmark", value_name = "NAME=FILE")]
    pub benchmarks: Vec<String>,
    #[arg(long, default_value_t = 0.1)]
    pub valid: f64,
    #[arg(long, default_value_t = 0.2)]
    pub test: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Finding ids accepted by the user (repeatable), e.g. T6.
    #[arg(long = "acknowledge", value_name = "ID")]
    pub acknowledge: Vec<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Quantization {
    /// 4-bit NF4 when a CUDA GPU and bitsandbytes are present, else plain LoRA.
    Auto,
    /// Always 4-bit NF4 (QLoRA); fails without a CUDA GPU.
    Nf4,
    /// Plain LoRA on 16-bit (GPU) or 32-bit (CPU) weights.
    None,
}

#[derive(Debug, Subcommand)]
pub enum TracesCmd {
    /// Audit a JSONL file of conversations (OpenAI or ShareGPT/Hermes format).
    Audit {
        #[command(flatten)]
        audit: AuditArgs,
        /// Print the audit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Audit, split by group, then fine-tune a chat LLM on the traces with LoRA/QLoRA.
    /// The untuned base model is measured on the same held-out conversations first.
    Finetune {
        #[command(flatten)]
        audit: AuditArgs,
        /// Hugging Face model id or local directory of an instruct/chat model.
        #[arg(long)]
        base: String,
        #[arg(long, value_enum, default_value = "auto")]
        quantization: Quantization,
        /// LoRA rank (literature default 16; tune learning rate and epochs before rank).
        #[arg(long, default_value_t = 16)]
        r: u32,
        /// LoRA alpha (default 2 × r).
        #[arg(long)]
        alpha: Option<u32>,
        #[arg(long, default_value_t = 1.0)]
        epochs: f64,
        /// Stop after this many optimizer steps (0 = no cap).
        #[arg(long, default_value_t = 0)]
        max_steps: u32,
        #[arg(long, default_value_t = 2e-4)]
        learning_rate: f64,
        /// Longest conversation in tokens; longer ones are dropped, never truncated.
        #[arg(long, default_value_t = 2048)]
        max_len: u32,
        /// Extra worker parameter as KEY=VALUE (JSON value), e.g. --param grad_accum=16.
        #[arg(long = "param", value_name = "KEY=VALUE")]
        params: Vec<String>,
        /// Prepare the data and the manifest, print the worker command, do not train.
        #[arg(long)]
        dry_run: bool,
    },
}

pub fn run(cmd: TracesCmd) -> anyhow::Result<i32> {
    match cmd {
        TracesCmd::Audit { audit: a, json } => {
            let (_, _, audit) = audited(&a)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&audit)?);
            } else {
                print(&a.data, &audit);
            }
            Ok(if audit.blocked { 2 } else { 0 })
        }
        TracesCmd::Finetune {
            audit: a,
            base,
            quantization,
            r,
            alpha,
            epochs,
            max_steps,
            learning_rate,
            max_len,
            params,
            dry_run,
        } => {
            let mut p = serde_json::Map::new();
            p.insert("r".into(), json!(r));
            p.insert("alpha".into(), json!(alpha.unwrap_or(2 * r)));
            p.insert("epochs".into(), json!(epochs));
            p.insert("max_steps".into(), json!(max_steps));
            p.insert("learning_rate".into(), json!(learning_rate));
            p.insert("max_len".into(), json!(max_len));
            p.insert(
                "quantization".into(),
                json!(format!("{quantization:?}").to_lowercase()),
            );
            for kv in &params {
                let (k, v) = kv
                    .split_once('=')
                    .with_context(|| format!("--param expects KEY=VALUE, got `{kv}`"))?;
                let v = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_owned()));
                p.insert(k.to_owned(), v);
            }
            finetune(&a, &base, Value::Object(p), dry_run)
        }
    }
}

fn audited(a: &AuditArgs) -> anyhow::Result<(TraceSet, Vec<Partition>, TraceAudit)> {
    let ok = |x: f64| (0.0..1.0).contains(&x);
    if !ok(a.valid) || !ok(a.test) || a.valid + a.test >= 1.0 {
        bail!("--valid and --test must be fractions that sum to less than 1");
    }
    let set = read_traces(&a.data)?;
    let bench = a
        .benchmarks
        .iter()
        .map(|b| load_benchmark(b))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let split = split_by_group(&set.conversations, a.valid, a.test, a.seed);
    let audit = audit_traces(&set, &split, &bench, &a.acknowledge);
    Ok((set, split, audit))
}

/// Why a conversation is left out of fine-tuning: unparseable calls, or calls that break
/// the JSON Schema the conversation itself declares (the T5 check of the audit).
fn exclusion(c: &Conversation) -> Option<String> {
    if let Some(why) = untrainable(c) {
        return Some(why.to_owned());
    }
    for t in c.tool_calls() {
        let Some(def) = c.tools.iter().find(|d| d.name == t.name) else {
            if !c.tools.is_empty() {
                return Some(format!("calls undeclared tool `{}`", t.name));
            }
            continue;
        };
        if let Some(args) = &t.arguments
            && let Some(v) = schema_violation(&def.parameters, args)
        {
            return Some(format!("`{}`: {v}", t.name));
        }
    }
    None
}

/// The files of a fine-tuning job, written under `$TEMPERA_HOME/finetune/<id>`.
struct Prepared {
    dir: PathBuf,
    manifest: PathBuf,
    counts: [usize; 3],
    excluded: Vec<(usize, String)>,
}

fn prepare(
    set: &TraceSet,
    split: &[Partition],
    base: &str,
    params: Value,
    seed: u64,
) -> anyhow::Result<Prepared> {
    let id = crate::orchestrator::new_job_id("ft");
    let dir = std::path::absolute(Workspace::discover().root.join("finetune").join(&id))?;
    let data = dir.join("data");
    std::fs::create_dir_all(&data)?;
    let names = ["train", "valid", "test"];
    let mut files = names
        .iter()
        .map(|n| std::fs::File::create(data.join(format!("{n}.jsonl"))).map(std::io::BufWriter::new))
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut counts = [0usize; 3];
    let mut excluded = vec![];
    for (c, part) in set.conversations.iter().zip(split) {
        if let Some(why) = exclusion(c) {
            excluded.push((c.line, why));
            continue;
        }
        let k = match part {
            Partition::Train => 0,
            Partition::Valid => 1,
            Partition::Test => 2,
        };
        writeln!(files[k], "{}", training_record(c))?;
        counts[k] += 1;
    }
    for f in &mut files {
        f.flush()?;
    }
    if counts[0] == 0 {
        bail!("no trainable conversations in the train split");
    }
    let path = |i: usize| (counts[i] > 0).then(|| data.join(format!("{}.jsonl", names[i])));
    let manifest = json!({
        "base_model": base,
        "train": path(0),
        "valid": path(1),
        "test": path(2),
        "work_dir": dir.join("work"),
        "model_dir": dir.join("model"),
        "seed": seed,
        "params": params,
    });
    let mpath = dir.join("manifest.json");
    std::fs::write(&mpath, serde_json::to_string_pretty(&manifest)?)?;
    Ok(Prepared {
        dir,
        manifest: mpath,
        counts,
        excluded,
    })
}

/// The worker command: `$TEMPERA_PYTHON -m tempera_worker`, else `uv run --extra llm`.
fn worker_command(manifest: &Path) -> anyhow::Result<Command> {
    let worker = worker_dir();
    python_available(worker.as_deref()).map_err(anyhow::Error::msg)?;
    let mut c = if let Ok(py) = std::env::var("TEMPERA_PYTHON") {
        Command::new(py)
    } else {
        let mut c = Command::new("uv");
        c.arg("run")
            .arg("--project")
            .arg(worker.context("worker project not found")?)
            .args(["--extra", "llm", "--quiet", "python"]);
        c
    };
    c.args(["-m", "tempera_worker", "finetune", "--manifest"])
        .arg(manifest);
    Ok(c)
}

fn finetune(a: &AuditArgs, base: &str, params: Value, dry_run: bool) -> anyhow::Result<i32> {
    let (set, split, audit) = audited(a)?;
    print(&a.data, &audit);
    if audit.blocked {
        return Ok(2);
    }
    let prep = prepare(&set, &split, base, params, a.seed)?;
    let [tr, va, te] = prep.counts;
    ui::step(&format!(
        "Data      train {tr} · valid {va} · test {te} conversations"
    ));
    if !prep.excluded.is_empty() {
        let shown: Vec<String> = prep
            .excluded
            .iter()
            .take(3)
            .map(|(l, w)| format!("line {l}: {w}"))
            .collect();
        ui::detail(&format!(
            "left out {} conversations that cannot teach tool use ({}{})",
            prep.excluded.len(),
            shown.join("; "),
            if prep.excluded.len() > 3 { "; …" } else { "" }
        ));
    }
    ui::detail(&format!("job dir {}", prep.dir.display()));
    let mut cmd = worker_command(&prep.manifest)?;
    if dry_run {
        println!(
            "\n  Dry run: nothing trained. Start it with:\n    {}\n",
            shell_line(&cmd)
        );
        return Ok(0);
    }
    ui::step(&format!("Training  {base} (LoRA, base model measured first)"));
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("cannot start the Python worker")?;
    let stdout = child.stdout.take().context("worker stdout")?;
    let mut log = std::fs::File::create(prep.dir.join("events.jsonl"))?;
    let mut summary = None;
    let mut error = None;
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        writeln!(log, "{line}")?;
        let Ok(ev) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match ev["t"].as_str() {
            Some("log") => ui::detail(ev["msg"].as_str().unwrap_or("")),
            Some("metric") => {
                let name = ev["name"].as_str().unwrap_or("");
                let v = ev["value"].as_f64().unwrap_or(f64::NAN);
                let step = ev["step"].as_u64();
                // Every 10th training step is enough for a terminal.
                if name != "train/lr" && (name != "train/loss" || step.is_some_and(|s| s % 10 == 0)) {
                    let at = step.map(|s| format!(" @ step {s}")).unwrap_or_default();
                    ui::detail(&format!("{name} = {v:.4}{at}"));
                }
            }
            Some("artifact") => ui::detail(&format!(
                "{} → {}",
                ev["kind"].as_str().unwrap_or(""),
                ev["path"].as_str().unwrap_or("")
            )),
            Some("done") => summary = Some(ev["summary"].clone()),
            Some("error") => error = ev["msg"].as_str().map(str::to_owned),
            _ => {}
        }
    }
    let status = child.wait()?;
    if let Some(s) = summary.filter(|_| status.success()) {
        std::fs::write(prep.dir.join("summary.json"), serde_json::to_string_pretty(&s)?)?;
        print_comparison(&s);
        println!(
            "\n  ✓ Adapter saved in {}\n",
            prep.dir.join("model/adapter").display()
        );
        return Ok(0);
    }
    let code = status.code().unwrap_or(1);
    println!(
        "\n  ✗ Fine-tuning failed (worker exit {code}): {}\n",
        error.as_deref().unwrap_or("see the log above")
    );
    Ok(if code == 0 { 1 } else { code })
}

/// Base model vs adapter on the same held-out conversations.
fn print_comparison(s: &Value) {
    let rows = [
        ("perplexity", "perplexity (lower is better)"),
        ("tool_call_rate", "emits a tool call"),
        ("valid_call_rate", "well-formed call"),
        ("tool_name_accuracy", "right tool"),
        ("exact_match", "right tool and arguments"),
    ];
    ui::step("Result    base → tuned, same held-out conversations");
    for (k, label) in rows {
        if let (Some(b), Some(t)) = (s[format!("base_{k}")].as_f64(), s[k].as_f64()) {
            ui::detail(&format!("{label:<30} {b:>8.3} → {t:.3}"));
        }
    }
}

fn shell_line(c: &Command) -> String {
    std::iter::once(c.get_program())
        .chain(c.get_args())
        .map(|a| {
            let a = a.to_string_lossy();
            if a.contains(' ') {
                format!("'{a}'")
            } else {
                a.into_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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

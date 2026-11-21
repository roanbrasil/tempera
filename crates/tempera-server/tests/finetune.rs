//! Integration: `tempera traces finetune` (#47). The audit gates training, the split and
//! the manifest reach the worker, and the worker's events become the base → tuned table.
//! A fake worker (`TEMPERA_PYTHON`) stands in for torch so this runs in CI; the real
//! QLoRA run is covered by the Python suite (`test_qlora_on_gpu_beats_the_base_model`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn data(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/data")
        .join(name)
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}

/// A stand-in for `python -m tempera_worker finetune --manifest M`: copies the manifest
/// next to it, checks the data files, and prints the events a real run prints.
fn fake_worker(dir: &Path, exit: i32) -> PathBuf {
    let script = dir.join("fake-python");
    let body = format!(
        r#"#!/bin/sh
manifest="$5"
cp "$manifest" "{seen}"
for f in train valid test; do test -s "$(dirname "$manifest")/data/$f.jsonl" || exit 9; done
echo '{{"t":"log","level":"info","msg":"base on cuda: 4-bit NF4 (QLoRA)"}}'
echo 'native library noise that is not JSON'
echo '{{"t":"metric","name":"eval/base_perplexity","value":1.8,"ts":1}}'
echo '{{"t":"metric","name":"train/loss","value":0.5,"step":10,"ts":1}}'
if [ {exit} -ne 0 ]; then echo '{{"t":"error","msg":"CUDA out of memory","retryable":false}}'; exit {exit}; fi
echo '{{"t":"artifact","kind":"model","path":"/x/adapter","format":"safetensors"}}'
echo '{{"t":"done","summary":{{"base_perplexity":1.8,"perplexity":1.01,"base_exact_match":0.125,"exact_match":0.875}}}}'
"#,
        seen = dir.join("seen.json").display(),
    );
    std::fs::write(&script, body).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    script
}

fn tempera(home: &Path, python: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tempera"))
        .args(args)
        .env("TEMPERA_HOME", home.join(".tempera"))
        .env("TEMPERA_PYTHON", python)
        .env("NO_COLOR", "1")
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn jobs(home: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(home.join(".tempera/finetune"))
        .map(|d| d.map(|e| e.unwrap().path()).collect())
        .unwrap_or_default()
}

fn lines(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn clean_traces_train_and_print_base_vs_tuned() {
    let tmp = tempfile::tempdir().unwrap();
    let py = fake_worker(tmp.path(), 0);
    let f = data("agent_traces_clean.jsonl");
    let out = tempera(
        tmp.path(),
        &py,
        &[
            "traces",
            "finetune",
            &f,
            "--base",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--max-steps",
            "20",
            "--param",
            "grad_accum=4",
        ],
    );
    let t = text(&out);
    assert_eq!(out.status.code(), Some(0), "{t}");
    assert!(t.contains("Nothing blocks fine-tuning"), "{t}");
    assert!(
        t.contains("right tool and arguments") && t.contains("0.125 → 0.875"),
        "{t}"
    );
    assert!(t.contains("Adapter saved"), "{t}");

    let job = &jobs(tmp.path())[0];
    // The manifest the worker received: paths into the job dir, literature defaults, overrides.
    let m: Value =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join("seen.json")).unwrap()).unwrap();
    assert_eq!(m["base_model"], "Qwen/Qwen2.5-0.5B-Instruct");
    assert_eq!(m["params"]["r"], 16);
    assert_eq!(m["params"]["alpha"], 32);
    assert_eq!(m["params"]["max_steps"], 20);
    assert_eq!(m["params"]["grad_accum"], 4);
    assert_eq!(m["params"]["quantization"], "auto");
    assert_eq!(m["seed"], 42);
    assert!(m["train"].as_str().unwrap().starts_with(job.to_str().unwrap()));

    // Group split: 800 conversations, no group in two partitions, Hermes markup removed.
    let parts: Vec<Vec<Value>> = ["train", "valid", "test"]
        .iter()
        .map(|p| lines(&job.join(format!("data/{p}.jsonl"))))
        .collect();
    assert_eq!(parts.iter().map(Vec::len).sum::<usize>(), 800);
    assert!(parts[0].len() > parts[2].len() && parts[2].len() > parts[1].len() / 2);
    let groups = |v: &[Value]| {
        v.iter()
            .map(|c| c["group"].as_str().unwrap().to_owned())
            .collect::<std::collections::HashSet<_>>()
    };
    assert!(groups(&parts[0]).is_disjoint(&groups(&parts[2])));
    assert!(groups(&parts[0]).is_disjoint(&groups(&parts[1])));
    let all = parts.concat();
    assert!(
        all.iter()
            .all(|c| !c.to_string().contains("<tool_call>") && !c.to_string().contains("<tools>"))
    );
    let call = all
        .iter()
        .flat_map(|c| c["messages"].as_array().unwrap().clone())
        .find(|m| m.get("tool_calls").is_some())
        .unwrap();
    assert!(call["tool_calls"][0]["arguments"].is_object(), "{call}");

    // Every event is kept, the summary is stored.
    let ev = std::fs::read_to_string(job.join("events.jsonl")).unwrap();
    assert!(ev.contains("\"t\":\"done\"") && ev.contains("not JSON"));
    let s: Value = serde_json::from_str(&std::fs::read_to_string(job.join("summary.json")).unwrap()).unwrap();
    assert_eq!(s["exact_match"], 0.875);
}

#[test]
fn red_audit_blocks_before_any_file_or_worker() {
    let tmp = tempfile::tempdir().unwrap();
    let py = fake_worker(tmp.path(), 0);
    let f = data("agent_traces.jsonl");
    let out = tempera(tmp.path(), &py, &["traces", "finetune", &f, "--base", "m"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(2), "{t}");
    assert!(t.contains("Fine-tuning blocked"), "{t}");
    assert!(jobs(tmp.path()).is_empty(), "no job dir may be created");
    assert!(
        !tmp.path().join("seen.json").exists(),
        "the worker must not start"
    );
}

#[test]
fn acknowledged_findings_train_without_the_broken_calls() {
    let tmp = tempfile::tempdir().unwrap();
    let py = fake_worker(tmp.path(), 0);
    let f = data("agent_traces.jsonl");
    let out = tempera(
        tmp.path(),
        &py,
        &[
            "traces",
            "finetune",
            &f,
            "--base",
            "m",
            "--dry-run",
            "--acknowledge",
            "T1",
            "--acknowledge",
            "T2",
            "--acknowledge",
            "T4",
            "--acknowledge",
            "T5",
        ],
    );
    let t = text(&out);
    assert_eq!(out.status.code(), Some(0), "{t}");
    assert!(t.contains("Dry run") && t.contains("finetune --manifest"), "{t}");
    // Exactly the 18 calls the T5 check reports as invalid are left out of training.
    assert!(t.contains("18 of 795 tool calls (2.3%) are invalid"), "{t}");
    assert!(
        t.contains("left out 18 conversations that cannot teach tool use"),
        "{t}"
    );
    assert!(
        !tmp.path().join("seen.json").exists(),
        "dry run must not start the worker"
    );
    let job = &jobs(tmp.path())[0];
    assert!(job.join("manifest.json").is_file());
    let kept: usize = ["train", "valid", "test"]
        .iter()
        .map(|p| lines(&job.join(format!("data/{p}.jsonl"))).len())
        .sum();
    assert_eq!(kept, 848 - 18, "every malformed call is left out, nothing else");
}

#[test]
fn worker_failure_is_reported_with_its_exit_code() {
    let tmp = tempfile::tempdir().unwrap();
    let py = fake_worker(tmp.path(), 1);
    let f = data("agent_traces_clean.jsonl");
    let out = tempera(tmp.path(), &py, &["traces", "finetune", &f, "--base", "m"]);
    let t = text(&out);
    assert_eq!(out.status.code(), Some(1), "{t}");
    assert!(
        t.contains("Fine-tuning failed (worker exit 1): CUDA out of memory"),
        "{t}"
    );
    assert!(!jobs(tmp.path())[0].join("summary.json").exists());

    let bad = tempera(
        tmp.path(),
        &py,
        &["traces", "finetune", &f, "--base", "m", "--param", "oops"],
    );
    assert!(text(&bad).contains("--param expects KEY=VALUE"), "{}", text(&bad));
}

//! Integration: `tempera traces audit` on the example agent traces.
//! Every planted problem is found; the clean twin has no findings.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn data(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/data")
        .join(name)
        .canonicalize()
        .unwrap()
        .display()
        .to_string()
}

fn tempera(args: &[&str]) -> Output {
    let home: PathBuf = std::env::temp_dir().join("tempera-traces-test");
    Command::new(env!("CARGO_BIN_EXE_tempera"))
        .args(args)
        .env("TEMPERA_HOME", home)
        .env("NO_COLOR", "1")
        .output()
        .unwrap()
}

fn bench() -> String {
    format!("support_bench={}", data("agent_benchmark.jsonl"))
}

fn audit_json(file: &str, extra: &[&str]) -> (Option<i32>, serde_json::Value) {
    let b = bench();
    let f = data(file);
    let mut args = vec!["traces", "audit", f.as_str(), "--benchmark", b.as_str(), "--json"];
    args.extend_from_slice(extra);
    let out = tempera(&args);
    let v = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    (out.status.code(), v)
}

fn ids(v: &serde_json::Value) -> Vec<String> {
    v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["id"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn planted_problems_are_all_found_and_block() {
    let (code, v) = audit_json("agent_traces.jsonl", &[]);
    assert_eq!(code, Some(2));
    assert_eq!(v["blocked"], true);
    let got = ids(&v);
    for want in ["T1", "T2", "T3:support_bench", "T4", "T5"] {
        assert!(got.contains(&want.to_owned()), "{want} missing in {got:?}");
    }
    assert!(v["profile"]["formats"]["sharegpt"].as_u64().unwrap() > 0);
    assert!(v["profile"]["formats"]["openai"].as_u64().unwrap() > 0);
}

#[test]
fn clean_traces_have_no_findings() {
    let (code, v) = audit_json("agent_traces_clean.jsonl", &[]);
    assert_eq!(code, Some(0), "{v:#}");
    assert!(ids(&v).is_empty(), "{:?}", ids(&v));
    let split = &v["split"];
    let total: u64 = ["train", "valid", "test"]
        .iter()
        .map(|k| split[k].as_u64().unwrap())
        .sum();
    assert_eq!(total, 800);
}

#[test]
fn acknowledging_every_red_unblocks() {
    let ack = [
        "--acknowledge",
        "T1",
        "--acknowledge",
        "T3:support_bench",
        "--acknowledge",
        "T4",
        "--acknowledge",
        "T5",
    ];
    let (code, v) = audit_json("agent_traces.jsonl", &ack);
    assert_eq!(code, Some(0), "{v:#}");
    assert_eq!(v["blocked"], false);
}

#[test]
fn human_output_and_errors() {
    let f = data("agent_traces.jsonl");
    let out = tempera(&["traces", "audit", f.as_str()]);
    let s = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        s.contains("Fine-tuning blocked") && s.contains("RED    T4"),
        "{s}"
    );
    assert!(
        !s.contains("@example.com"),
        "audit output must not echo personal data"
    );

    let out = tempera(&["traces", "audit", "/nonexistent.jsonl"]);
    assert_eq!(out.status.code(), Some(1));
    let out = tempera(&["traces", "audit", f.as_str(), "--benchmark", "nofile"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("NAME=FILE"));
    let out = tempera(&["traces", "audit", f.as_str(), "--valid", "0.6", "--test", "0.5"]);
    assert_eq!(out.status.code(), Some(1));
}

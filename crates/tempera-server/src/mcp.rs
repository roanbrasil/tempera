//! `tempera mcp`: Têmpera as an MCP server (stdio, JSON-RPC 2.0, one message per line).
//!
//! Agents (Claude Code, Hermes Agent, OpenClaw, any MCP client) get the same
//! guardrails as a person at the terminal:
//! - every tool runs the real `tempera` CLI in a subprocess, so the audit still
//!   blocks on red findings and the test set is still read only once (SQLite);
//! - no tool accepts an audit finding: that decision stays with a human;
//! - a job that would rent cloud machines is refused unless a human started the
//!   server with `TEMPERA_MCP_ALLOW_CLOUD=1`.
//!
//! Stdout carries only protocol messages; logs go to stderr.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tempera_core::JobSpec;

/// Protocol revisions this server speaks, newest first.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// Env var a human sets to let agents run jobs on paid cloud providers.
pub const ALLOW_CLOUD_ENV: &str = "TEMPERA_MCP_ALLOW_CLOUD";

const INSTRUCTIONS: &str = "Têmpera trains tabular ML models with guardrails. Typical flow: \
init_spec (or write a tempera.yaml) → audit → run → show_job / report_path. \
When audit or run reports BLOCKED, show the findings to the user and let them decide \
(drop the column or accept it in tempera.yaml); do not decide for them. \
The test set is evaluated once per job; retraining to improve the test score is not possible by design.";

/// Exit status and captured output of one CLI invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `tempera <args>`; a trait so the protocol logic can be tested without processes.
pub trait Cli {
    fn run(&self, args: &[String]) -> CliOutput;
}

/// The real thing: re-executes the current binary.
#[derive(Debug)]
pub struct SelfExe;

impl Cli for SelfExe {
    fn run(&self, args: &[String]) -> CliOutput {
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("tempera"));
        match std::process::Command::new(exe)
            .args(args)
            .env("NO_COLOR", "1")
            .stdin(std::process::Stdio::null())
            .output()
        {
            Ok(o) => CliOutput {
                code: o.status.code().unwrap_or(1),
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
            },
            Err(e) => CliOutput {
                code: 1,
                stdout: String::new(),
                stderr: format!("cannot start tempera: {e}"),
            },
        }
    }
}

/// Serves MCP over stdin/stdout until stdin closes.
pub fn serve(cli: &dyn Cli) -> anyhow::Result<()> {
    let allow_cloud = std::env::var(ALLOW_CLOUD_ENV).is_ok_and(|v| v == "1");
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(&msg, cli, allow_cloud),
            Err(e) => Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(reply) = reply {
            writeln!(stdout, "{reply}")?;
            stdout.flush()?;
        }
    }
    Ok(())
}

/// Handles one JSON-RPC message; `None` for notifications.
pub fn handle(msg: &Value, cli: &dyn Cli, allow_cloud: bool) -> Option<Value> {
    let Some(method) = msg.get("method").and_then(Value::as_str) else {
        // A response from the client (we never send requests) or garbage.
        return msg
            .get("id")
            .filter(|_| msg.get("result").is_none() && msg.get("error").is_none())
            .map(|id| error(id.clone(), -32600, "invalid request: missing method"));
    };
    let id = msg.get("id").cloned()?; // notifications get no reply
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    Some(match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str);
            let version = asked
                .filter(|v| PROTOCOL_VERSIONS.contains(v))
                .unwrap_or(PROTOCOL_VERSIONS[0]);
            ok(
                id,
                json!({
                    "protocolVersion": version,
                    "capabilities": { "tools": { "listChanged": false } },
                    "serverInfo": { "name": "tempera", "version": env!("CARGO_PKG_VERSION") },
                    "instructions": INSTRUCTIONS,
                }),
            )
        }
        "ping" => ok(id, json!({})),
        "tools/list" => ok(id, json!({ "tools": tools() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            match call(name, &args, cli, allow_cloud) {
                Ok(result) => ok(id, result),
                Err(ToolError::Unknown) => error(id, -32602, &format!("unknown tool `{name}`")),
                Err(ToolError::Invalid(m)) => ok(id, text_result(&m, true)),
            }
        }
        other => error(id, -32601, &format!("method not found: {other}")),
    })
}

#[derive(Debug)]
enum ToolError {
    Unknown,
    /// Bad arguments or a refused action: reported to the model as a tool error.
    Invalid(String),
}

fn tools() -> Value {
    let spec = json!({ "type": "object", "properties": {
        "spec": { "type": "string", "description": "Path to a tempera.yaml" } }, "required": ["spec"] });
    let job = json!({ "type": "object", "properties": {
        "job": { "type": "string", "description": "Job id or unique prefix" } }, "required": ["job"] });
    json!([
        { "name": "validate_spec", "description": "Check a tempera.yaml against the schema.",
          "inputSchema": spec, "annotations": { "readOnlyHint": true } },
        { "name": "spec_schema", "description": "JSON Schema of tempera.yaml, to write specs correctly.",
          "inputSchema": { "type": "object", "properties": {} }, "annotations": { "readOnlyHint": true } },
        { "name": "profile_data", "description": "Ingest a CSV/Parquet/JSONL file and return rows, columns, types, null fractions and cardinality.",
          "inputSchema": { "type": "object", "properties": {
              "data": { "type": "string", "description": "Path to the data file" } }, "required": ["data"] },
          "annotations": { "readOnlyHint": true } },
        { "name": "init_spec", "description": "Draft a tempera.yaml for a data file and a target column. Never overwrites an existing file.",
          "inputSchema": { "type": "object", "properties": {
              "data": { "type": "string" },
              "target": { "type": "string", "description": "Column to predict" },
              "name": { "type": "string" },
              "output": { "type": "string", "description": "Where to write the spec (default tempera.yaml)" } },
              "required": ["data", "target"] } },
        { "name": "audit", "description": "Run the 13-check data audit (leakage, duplicates, drift, PII, fairness…) without training. BLOCKED means a human must decide on red findings.",
          "inputSchema": spec },
        { "name": "run", "description": "Audit, split, train every candidate, pick the winner on validation and evaluate it once on the held-out test set. Can take minutes. Refused when the spec rents cloud machines unless the user enabled it.",
          "inputSchema": spec },
        { "name": "estimate", "description": "Estimate time and cost of a job on each allowed machine without renting anything.",
          "inputSchema": spec, "annotations": { "readOnlyHint": true } },
        { "name": "audit_traces", "description": "Audit agent traces (JSONL chat conversations with tool calls, OpenAI or ShareGPT/Hermes format) before fine-tuning an LLM: same prompt in train and test, duplicates, benchmark contamination, personal data, malformed tool calls, size and balance. BLOCKED means a human must decide.",
          "inputSchema": { "type": "object", "properties": {
              "data": { "type": "string", "description": "Path to the JSONL file" },
              "benchmarks": { "type": "array", "items": { "type": "string" },
                              "description": "Benchmarks you will report on, as NAME=prompts.jsonl" } },
              "required": ["data"] },
          "annotations": { "readOnlyHint": true } },
        { "name": "list_jobs", "description": "Recent jobs with status, task and metric.",
          "inputSchema": { "type": "object", "properties": {
              "limit": { "type": "integer", "minimum": 1, "maximum": 200 } } },
          "annotations": { "readOnlyHint": true } },
        { "name": "show_job", "description": "Runs of one job: algorithm, backend, status, validation score, time, winner and report path.",
          "inputSchema": job, "annotations": { "readOnlyHint": true } },
        { "name": "report_path", "description": "Path of a job's HTML report (latest job when omitted).",
          "inputSchema": { "type": "object", "properties": { "job": { "type": "string" } } },
          "annotations": { "readOnlyHint": true } },
    ])
}

fn str_arg(args: &Value, key: &str) -> Result<String, ToolError> {
    match args.get(key).and_then(Value::as_str) {
        Some(s) if !s.trim().is_empty() => Ok(s.to_owned()),
        _ => Err(ToolError::Invalid(format!("missing string argument `{key}`"))),
    }
}

fn opt_str(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
}

fn call(name: &str, args: &Value, cli: &dyn Cli, allow_cloud: bool) -> Result<Value, ToolError> {
    let argv: Vec<String> = match name {
        "validate_spec" => vec!["validate".into(), str_arg(args, "spec")?],
        "spec_schema" => vec!["schema".into()],
        "profile_data" => vec!["ingest".into(), str_arg(args, "data")?],
        "init_spec" => {
            let mut v = vec![
                "init".into(),
                str_arg(args, "data")?,
                "--target".into(),
                str_arg(args, "target")?,
            ];
            if let Some(n) = opt_str(args, "name") {
                v.extend(["--name".into(), n]);
            }
            if let Some(o) = opt_str(args, "output") {
                v.extend(["--output".into(), o]);
            }
            v
        }
        "audit" => vec!["audit".into(), str_arg(args, "spec")?],
        "run" => {
            let spec = str_arg(args, "spec")?;
            if let Some(providers) = cloud_providers(Path::new(&spec))?
                && !allow_cloud
            {
                return Err(ToolError::Invalid(format!(
                    "refused: this spec may rent paid cloud machines ({}). Ask the user to review the \
                     estimate (tool `estimate`) and restart the MCP server with {ALLOW_CLOUD_ENV}=1, \
                     or remove the providers from compute.allow.",
                    providers.join(", ")
                )));
            }
            vec!["run".into(), spec]
        }
        "estimate" => vec!["cloud".into(), "estimate".into(), str_arg(args, "spec")?],
        "audit_traces" => {
            let mut v = vec!["traces".into(), "audit".into(), str_arg(args, "data")?];
            for b in args
                .get("benchmarks")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let b = b
                    .as_str()
                    .filter(|b| b.contains('='))
                    .ok_or_else(|| ToolError::Invalid("benchmarks must be NAME=FILE strings".into()))?;
                v.extend(["--benchmark".into(), b.to_owned()]);
            }
            v
        }
        "list_jobs" => {
            let limit = args
                .get("limit")
                .and_then(Value::as_i64)
                .unwrap_or(20)
                .clamp(1, 200);
            vec!["runs".into(), "--limit".into(), limit.to_string()]
        }
        "show_job" => vec!["runs".into(), str_arg(args, "job")?],
        "report_path" => {
            let mut v = vec!["report".to_owned()];
            v.extend(opt_str(args, "job"));
            v
        }
        _ => return Err(ToolError::Unknown),
    };
    Ok(cli_result(&cli.run(&argv)))
}

/// Paid cloud providers in `compute.allow` (the free `mock` provider does not count).
fn cloud_providers(spec: &Path) -> Result<Option<Vec<String>>, ToolError> {
    let text = std::fs::read_to_string(spec)
        .map_err(|e| ToolError::Invalid(format!("cannot read {}: {e}", spec.display())))?;
    let spec = JobSpec::from_yaml(&text).map_err(|e| ToolError::Invalid(format!("invalid spec: {e}")))?;
    let paid: Vec<String> = spec
        .compute
        .allow
        .iter()
        .filter(|a| tempera_cloud::is_provider_name(a) && a.as_str() != "mock")
        .cloned()
        .collect();
    Ok((!paid.is_empty()).then_some(paid))
}

fn cli_result(o: &CliOutput) -> Value {
    let mut text = String::new();
    match o.code {
        0 => {}
        2 => text.push_str(
            "BLOCKED by the data audit. Show the red findings to the user; they decide whether to \
             drop each column or accept it in tempera.yaml.\n\n",
        ),
        c => text.push_str(&format!("FAILED (exit code {c}).\n\n")),
    }
    text.push_str(o.stdout.trim_end());
    let err = o.stderr.trim();
    if !err.is_empty() {
        text.push_str("\n\nstderr:\n");
        text.push_str(err);
    }
    text_result(text.trim(), o.code != 0)
}

fn text_result(text: &str, is_error: bool) -> Value {
    json!({ "content": [{ "type": "text", "text": text }], "isError": is_error })
}

fn ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::Mutex;

    use super::*;

    /// Records calls and answers with a canned output.
    #[derive(Debug)]
    struct Fake {
        out: CliOutput,
        calls: Mutex<Vec<Vec<String>>>,
    }

    impl Fake {
        fn new(code: i32, stdout: &str) -> Self {
            Self {
                out: CliOutput {
                    code,
                    stdout: stdout.into(),
                    stderr: String::new(),
                },
                calls: Mutex::new(vec![]),
            }
        }
        fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Cli for Fake {
        fn run(&self, args: &[String]) -> CliOutput {
            self.calls.lock().unwrap().push(args.to_vec());
            self.out.clone()
        }
    }

    fn req(method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": 7, "method": method, "params": params })
    }

    fn call_tool(fake: &Fake, name: &str, args: Value, allow_cloud: bool) -> Value {
        handle(
            &req("tools/call", json!({ "name": name, "arguments": args })),
            fake,
            allow_cloud,
        )
        .unwrap()
    }

    fn text(reply: &Value) -> &str {
        reply["result"]["content"][0]["text"].as_str().unwrap()
    }

    #[test]
    fn initialize_negotiates_version() {
        let f = Fake::new(0, "");
        let r = handle(
            &req("initialize", json!({ "protocolVersion": "2024-11-05" })),
            &f,
            false,
        )
        .unwrap();
        assert_eq!(r["id"], 7);
        assert_eq!(r["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(r["result"]["serverInfo"]["name"], "tempera");
        assert!(r["result"]["capabilities"]["tools"].is_object());
        let r = handle(
            &req("initialize", json!({ "protocolVersion": "1999-01-01" })),
            &f,
            false,
        )
        .unwrap();
        assert_eq!(r["result"]["protocolVersion"], PROTOCOL_VERSIONS[0]);
    }

    #[test]
    fn notifications_get_no_reply() {
        let f = Fake::new(0, "");
        let n = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(handle(&n, &f, false).is_none());
        // A client response to a request we never sent is ignored too.
        assert!(handle(&json!({ "jsonrpc": "2.0", "id": 1, "result": {} }), &f, false).is_none());
    }

    #[test]
    fn unknown_method_and_tool_are_errors() {
        let f = Fake::new(0, "");
        let r = handle(&req("resources/list", json!({})), &f, false).unwrap();
        assert_eq!(r["error"]["code"], -32601);
        let r = call_tool(&f, "rm_rf", json!({}), false);
        assert_eq!(r["error"]["code"], -32602);
        assert!(f.calls().is_empty());
    }

    #[test]
    fn tools_list_is_well_formed() {
        let f = Fake::new(0, "");
        let r = handle(&req("tools/list", json!({})), &f, false).unwrap();
        let tools = r["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 11);
        for t in tools {
            assert!(t["name"].is_string() && t["description"].is_string());
            assert_eq!(t["inputSchema"]["type"], "object", "{t}");
        }
        // No tool can accept an audit finding on the user's behalf.
        assert!(
            !tools
                .iter()
                .any(|t| t["name"].as_str().unwrap().contains("accept"))
        );
    }

    #[test]
    fn tools_map_to_cli_arguments() {
        let f = Fake::new(0, "ok");
        call_tool(&f, "validate_spec", json!({ "spec": "a.yaml" }), false);
        call_tool(&f, "profile_data", json!({ "data": "d.csv" }), false);
        call_tool(
            &f,
            "init_spec",
            json!({ "data": "d.csv", "target": "y", "output": "o.yaml" }),
            false,
        );
        call_tool(&f, "list_jobs", json!({ "limit": 5000 }), false);
        call_tool(&f, "show_job", json!({ "job": "abc" }), false);
        call_tool(&f, "report_path", json!({}), false);
        call_tool(&f, "estimate", json!({ "spec": "a.yaml" }), false);
        call_tool(
            &f,
            "audit_traces",
            json!({ "data": "t.jsonl", "benchmarks": ["bfcl=b.jsonl"] }),
            false,
        );
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        assert_eq!(
            f.calls(),
            vec![
                s(&["validate", "a.yaml"]),
                s(&["ingest", "d.csv"]),
                s(&["init", "d.csv", "--target", "y", "--output", "o.yaml"]),
                s(&["runs", "--limit", "200"]),
                s(&["runs", "abc"]),
                s(&["report"]),
                s(&["cloud", "estimate", "a.yaml"]),
                s(&["traces", "audit", "t.jsonl", "--benchmark", "bfcl=b.jsonl"]),
            ]
        );
    }

    #[test]
    fn missing_argument_is_a_tool_error_without_running() {
        let f = Fake::new(0, "");
        let r = call_tool(&f, "audit", json!({}), false);
        assert_eq!(r["result"]["isError"], true);
        assert!(text(&r).contains("`spec`"));
        assert!(f.calls().is_empty());
    }

    #[test]
    fn blocked_and_failed_exit_codes_are_explained() {
        let f = Fake::new(2, "RED A1:refund_amount");
        let r = call_tool(&f, "audit", json!({ "spec": "x.yaml" }), false);
        assert_eq!(r["result"]["isError"], true);
        assert!(text(&r).starts_with("BLOCKED"));
        assert!(text(&r).contains("A1:refund_amount"));
        let f = Fake::new(1, "");
        let r = call_tool(&f, "validate_spec", json!({ "spec": "x.yaml" }), false);
        assert!(text(&r).starts_with("FAILED (exit code 1)"));
    }

    fn spec_file(dir: &Path, allow: &str) -> String {
        let p = dir.join("s.yaml");
        std::fs::write(
            &p,
            format!(
                "version: 1\nname: t\ndata: {{ source: d.csv, target: y }}\ncompute: {{ allow: [{allow}] }}\n"
            ),
        )
        .unwrap();
        p.display().to_string()
    }

    #[test]
    fn run_on_paid_cloud_needs_human_opt_in() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec_file(tmp.path(), "local, runpod");
        let f = Fake::new(0, "trained");
        let r = call_tool(&f, "run", json!({ "spec": spec }), false);
        assert_eq!(r["result"]["isError"], true);
        assert!(text(&r).contains("runpod") && text(&r).contains(ALLOW_CLOUD_ENV));
        assert!(f.calls().is_empty(), "nothing may run before the opt-in");

        let r = call_tool(&f, "run", json!({ "spec": spec }), true);
        assert_eq!(r["result"]["isError"], false);
        assert_eq!(f.calls().len(), 1);
    }

    #[test]
    fn run_locally_or_on_mock_needs_no_opt_in() {
        let tmp = tempfile::tempdir().unwrap();
        let f = Fake::new(0, "trained");
        for allow in ["local", "mock"] {
            let spec = spec_file(tmp.path(), allow);
            let r = call_tool(&f, "run", json!({ "spec": spec }), false);
            assert_eq!(r["result"]["isError"], false, "{allow}: {r}");
        }
        assert_eq!(f.calls().len(), 2);
    }

    #[test]
    fn audit_traces_rejects_malformed_benchmarks() {
        let f = Fake::new(0, "");
        let r = call_tool(
            &f,
            "audit_traces",
            json!({ "data": "t.jsonl", "benchmarks": ["no-equals"] }),
            false,
        );
        assert_eq!(r["result"]["isError"], true);
        assert!(f.calls().is_empty());
    }

    #[test]
    fn run_with_unreadable_spec_is_refused() {
        let f = Fake::new(0, "");
        let r = call_tool(&f, "run", json!({ "spec": "/nonexistent/x.yaml" }), false);
        assert_eq!(r["result"]["isError"], true);
        assert!(f.calls().is_empty());
    }
}

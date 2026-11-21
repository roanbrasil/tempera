//! Integration: an MCP client talking to the real `tempera mcp` over stdio,
//! driving the churn example from audit (blocked) to a trained, reported job.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

struct Client {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl Client {
    fn start(home: &Path, env: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_tempera"));
        cmd.arg("mcp")
            .current_dir(home)
            .env("TEMPERA_HOME", home.join(".tempera"))
            .env("TEMPERA_STATE_DIR", home.join(".state"))
            .env("TEMPERA_NUM_THREADS", "2")
            .env("TEMPERA_ADVISOR", "off")
            .env_remove("TEMPERA_MCP_ALLOW_CLOUD")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
        assert_eq!(reply["id"], id, "{reply}");
        reply
    }

    fn tool(&mut self, name: &str, args: Value) -> (bool, String) {
        let r = self.request("tools/call", json!({ "name": name, "arguments": args }));
        let res = &r["result"];
        (
            res["isError"].as_bool().unwrap(),
            res["content"][0]["text"].as_str().unwrap().to_owned(),
        )
    }

    fn handshake(&mut self) -> Value {
        let r = self.request(
            "initialize",
            json!({ "protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": { "name": "test", "version": "0" } }),
        );
        self.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        r
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let data = repo().join("examples/data");
    let p = dir.join(name);
    std::fs::write(&p, body.replace("$DATA", &data.display().to_string())).unwrap();
    p.display().to_string()
}

#[test]
fn agent_drives_churn_from_blocked_audit_to_report() {
    let tmp = tempfile::tempdir().unwrap();
    let mut c = Client::start(tmp.path(), &[]);

    let init = c.handshake();
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "tempera");

    let tools = c.request("tools/list", json!({}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for n in ["audit", "run", "profile_data", "report_path"] {
        assert!(names.contains(&n), "{names:?}");
    }

    let csv = repo().join("examples/data/churn.csv");
    let (err, text) = c.tool("profile_data", json!({ "data": csv }));
    assert!(!err, "{text}");
    assert!(
        text.contains("refund_amount") && text.contains("churned"),
        "{text}"
    );
    assert!(!text.contains('\u{1b}'), "no ANSI colors over MCP");

    // The leak blocks the agent exactly as it blocks a person.
    let leaky = write(
        tmp.path(),
        "leaky.yaml",
        "version: 1\nname: churn\ndata: { source: $DATA/churn.csv, target: churned }\n",
    );
    let (err, text) = c.tool("audit", json!({ "spec": leaky }));
    assert!(err, "{text}");
    assert!(
        text.starts_with("BLOCKED") && text.contains("A1:refund_amount"),
        "{text}"
    );
    let (err, text) = c.tool("run", json!({ "spec": leaky }));
    assert!(err && text.starts_with("BLOCKED"), "{text}");

    // After the human's decision (drop), the agent can train.
    let fixed = write(
        tmp.path(),
        "fixed.yaml",
        "version: 1\nname: churn\ndata: { source: $DATA/churn.csv, target: churned, time_column: signup_date, drop: [refund_amount] }\nstrategy: { candidates: [logistic_regression, decision_tree] }\n",
    );
    let (err, text) = c.tool("validate_spec", json!({ "spec": fixed }));
    assert!(!err, "{text}");
    let (err, text) = c.tool("run", json!({ "spec": fixed }));
    assert!(!err, "{text}");
    assert!(text.contains("evaluated once"), "{text}");

    let (err, text) = c.tool("list_jobs", json!({ "limit": 5 }));
    assert!(
        !err && text.contains("succeeded") && text.contains("blocked"),
        "{text}"
    );

    // Agent traces go through the same gate.
    let traces = repo().join("examples/data/agent_traces.jsonl");
    let (err, text) = c.tool("audit_traces", json!({ "data": traces }));
    assert!(
        err && text.starts_with("BLOCKED") && text.contains("T5"),
        "{text}"
    );

    let (err, path) = c.tool("report_path", json!({}));
    assert!(!err, "{path}");
    let html = std::fs::read_to_string(path.trim()).unwrap();
    assert!(html.contains("logistic_regression"));
}

#[test]
fn paid_cloud_job_is_refused_without_opt_in() {
    let tmp = tempfile::tempdir().unwrap();
    let mut c = Client::start(tmp.path(), &[]);
    c.handshake();
    let spec = write(
        tmp.path(),
        "cloud.yaml",
        "version: 1\nname: houses\ndata: { source: $DATA/houses.csv, target: price }\ncompute: { allow: [vast] }\n",
    );
    let (err, text) = c.tool("run", json!({ "spec": spec }));
    assert!(err && text.contains("TEMPERA_MCP_ALLOW_CLOUD"), "{text}");
    let (_, jobs) = c.tool("list_jobs", json!({}));
    assert!(!jobs.contains("houses"), "no job may start: {jobs}");
}

#[test]
fn protocol_errors_do_not_kill_the_server() {
    let tmp = tempfile::tempdir().unwrap();
    let mut c = Client::start(tmp.path(), &[]);
    c.handshake();
    // Garbage line → parse error reply, server keeps going.
    writeln!(c.stdin, "{{not json").unwrap();
    c.stdin.flush().unwrap();
    let mut line = String::new();
    c.stdout.read_line(&mut line).unwrap();
    let r: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(r["error"]["code"], -32700);

    let r = c.request("nope/method", json!({}));
    assert_eq!(r["error"]["code"], -32601);
    let r = c.request("ping", json!({}));
    assert!(r["result"].is_object());
}

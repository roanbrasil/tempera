//! Agent traces (#66): chat conversations with tool calls, for fine-tuning LLMs
//! on tool use. Two input formats, one normalized model:
//!
//! - **OpenAI chat**: `{"messages": [{"role", "content", "tool_calls": [{"function":
//!   {"name", "arguments"}}]}], "tools": [{"type": "function", "function": {...}}]}`.
//! - **ShareGPT / Hermes function calling**: `{"conversations": [{"from", "value"}]}` where
//!   assistant turns carry `<tool_call>{"name", "arguments"}</tool_call>` and tools are
//!   declared in a `tools` field or inside `<tools>…</tools>` in the system prompt.
//!
//! Splits are by **group** (`task_id`/`group` field, else the conversation id, else
//! the first user message), never by turn: turns of one conversation, or two
//! conversations of the same task, must not land in train and test.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::split::Partition;
use crate::{DataError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFormat {
    OpenAi,
    ShareGpt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Parsed arguments; `None` when they are not valid JSON.
    pub arguments: Option<Value>,
    /// Arguments exactly as written.
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Tool-call blocks that are not even `{"name", "arguments"}` JSON.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub broken_calls: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    /// JSON Schema of the arguments (`{}` when not declared).
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conversation {
    /// 1-based line in the source file.
    pub line: usize,
    pub id: String,
    /// Split unit: conversations sharing a group always share a partition.
    pub group: String,
    pub format: TraceFormat,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
}

impl Conversation {
    pub fn first_user(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
    }

    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.messages.iter().flat_map(|m| &m.tool_calls)
    }
}

/// Every conversation of a file, plus the lines that could not be read.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceSet {
    pub conversations: Vec<Conversation>,
    pub invalid_lines: Vec<(usize, String)>,
}

/// Reads a JSONL file of conversations.
pub fn read_traces(path: &Path) -> Result<TraceSet> {
    if !path.is_file() {
        return Err(DataError::NotFound(path.to_path_buf()));
    }
    let file = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut set = TraceSet::default();
    for (i, line) in file.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match parse_line(&line, i + 1) {
            Ok(c) => set.conversations.push(c),
            Err(e) => set.invalid_lines.push((i + 1, e)),
        }
    }
    if set.conversations.is_empty() {
        return Err(DataError::Invalid(format!(
            "{}: no readable conversations ({} invalid lines)",
            path.display(),
            set.invalid_lines.len()
        )));
    }
    Ok(set)
}

/// Parses one JSONL record.
pub fn parse_line(line: &str, line_no: usize) -> std::result::Result<Conversation, String> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let (format, messages) = if let Some(msgs) = v.get("messages").and_then(Value::as_array) {
        (
            TraceFormat::OpenAi,
            msgs.iter().map(openai_message).collect::<Result<Vec<_>, _>>()?,
        )
    } else if let Some(msgs) = v.get("conversations").and_then(Value::as_array) {
        (
            TraceFormat::ShareGpt,
            msgs.iter().map(sharegpt_message).collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        return Err("expected a `messages` (OpenAI) or `conversations` (ShareGPT/Hermes) array".into());
    };
    if messages.is_empty() {
        return Err("empty conversation".into());
    }
    let mut tools = v.get("tools").map(tool_defs).unwrap_or_default();
    if tools.is_empty()
        && let Some(sys) = messages.iter().find(|m| m.role == Role::System)
    {
        tools = tools_in_text(&sys.content);
    }
    let field = |k: &str| {
        v.get(k).and_then(|x| match x {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        })
    };
    let first_user = messages
        .iter()
        .find(|m| m.role == Role::User)
        .map(|m| normalize(&m.content));
    let id = field("id").unwrap_or_else(|| format!("line-{line_no}"));
    let group = field("task_id")
        .or_else(|| field("group"))
        .or_else(|| field("id"))
        .or_else(|| first_user.map(|u| format!("prompt:{}", short_hash(&u))))
        .unwrap_or_else(|| id.clone());
    Ok(Conversation {
        line: line_no,
        id,
        group,
        format,
        messages,
        tools,
    })
}

fn content_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        // Multimodal content parts: keep the text ones.
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn role(s: &str) -> std::result::Result<Role, String> {
    Ok(match s {
        "system" | "developer" => Role::System,
        "user" | "human" => Role::User,
        "assistant" | "gpt" | "model" => Role::Assistant,
        "tool" | "function" | "ipython" | "observation" => Role::Tool,
        other => return Err(format!("unknown role `{other}`")),
    })
}

fn call_from(name: Option<&str>, args: Option<&Value>) -> ToolCall {
    let (arguments, raw) = match args {
        // OpenAI encodes arguments as a JSON string.
        Some(Value::String(s)) => (serde_json::from_str(s).ok(), s.clone()),
        Some(v) => (Some(v.clone()), v.to_string()),
        None => (Some(Value::Object(Default::default())), "{}".into()),
    };
    ToolCall {
        name: name.unwrap_or("").to_owned(),
        arguments,
        raw,
    }
}

fn openai_message(m: &Value) -> std::result::Result<Message, String> {
    let r = role(m.get("role").and_then(Value::as_str).unwrap_or(""))?;
    let mut tool_calls: Vec<ToolCall> = m
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|c| {
                    let f = c.get("function").unwrap_or(c);
                    call_from(f.get("name").and_then(Value::as_str), f.get("arguments"))
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(f) = m.get("function_call") {
        tool_calls.push(call_from(
            f.get("name").and_then(Value::as_str),
            f.get("arguments"),
        ));
    }
    let content = content_text(m.get("content"));
    // Some OpenAI-format datasets still embed Hermes-style calls in the text.
    let (embedded, broken) = if r == Role::Assistant {
        tool_calls_in_text(&content)
    } else {
        (vec![], vec![])
    };
    tool_calls.extend(embedded);
    Ok(Message {
        role: r,
        content,
        tool_calls,
        broken_calls: broken,
    })
}

fn sharegpt_message(m: &Value) -> std::result::Result<Message, String> {
    let r = role(
        m.get("from")
            .or_else(|| m.get("role"))
            .and_then(Value::as_str)
            .unwrap_or(""),
    )?;
    let content = content_text(m.get("value").or_else(|| m.get("content")));
    let (tool_calls, broken_calls) = if r == Role::Assistant {
        tool_calls_in_text(&content)
    } else {
        (vec![], vec![])
    };
    Ok(Message {
        role: r,
        content,
        tool_calls,
        broken_calls,
    })
}

/// Blocks between `<open>` and `<close>`.
fn tagged<'a>(text: &'a str, open: &str, close: &str) -> Vec<&'a str> {
    let mut out = vec![];
    let mut rest = text;
    while let Some(i) = rest.find(open) {
        let after = &rest[i + open.len()..];
        match after.find(close) {
            Some(j) => {
                out.push(after[..j].trim());
                rest = &after[j + close.len()..];
            }
            None => {
                out.push(after.trim()); // unterminated: still a (broken) call
                break;
            }
        }
    }
    out
}

/// Hermes `<tool_call>{"name": …, "arguments": …}</tool_call>` blocks.
pub fn tool_calls_in_text(text: &str) -> (Vec<ToolCall>, Vec<String>) {
    let mut calls = vec![];
    let mut broken = vec![];
    for block in tagged(text, "<tool_call>", "</tool_call>") {
        match serde_json::from_str::<Value>(block) {
            Ok(v) if v.get("name").and_then(Value::as_str).is_some() => {
                calls.push(call_from(
                    v.get("name").and_then(Value::as_str),
                    v.get("arguments"),
                ));
            }
            _ => broken.push(block.to_owned()),
        }
    }
    (calls, broken)
}

fn tool_def(v: &Value) -> Option<ToolDef> {
    let f = v.get("function").unwrap_or(v);
    Some(ToolDef {
        name: f.get("name")?.as_str()?.to_owned(),
        parameters: f
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default())),
    })
}

fn tool_defs(v: &Value) -> Vec<ToolDef> {
    match v {
        Value::Array(a) => a.iter().filter_map(tool_def).collect(),
        // Some datasets store the tool list as a JSON string.
        Value::String(s) => serde_json::from_str::<Value>(s)
            .map(|v| tool_defs(&v))
            .unwrap_or_default(),
        Value::Object(_) => tool_def(v).into_iter().collect(),
        _ => vec![],
    }
}

/// Tools declared in a system prompt: `<tools>[…]</tools>` or a run of JSON objects.
pub fn tools_in_text(text: &str) -> Vec<ToolDef> {
    let mut out = vec![];
    for block in tagged(text, "<tools>", "</tools>") {
        for v in serde_json::Deserializer::from_str(block).into_iter::<Value>() {
            match v {
                Ok(v) => out.extend(tool_defs(&v)),
                Err(_) => break,
            }
        }
    }
    out
}

/// Lowercase, collapsed whitespace: the key for exact-duplicate detection.
pub fn normalize(s: &str) -> String {
    s.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn short_hash(s: &str) -> String {
    hex::encode(&Sha256::digest(s.as_bytes())[..8])
}

/// Deterministic group split: each group goes whole to one partition.
pub fn split_by_group(
    convs: &[Conversation],
    valid_fraction: f64,
    test_fraction: f64,
    seed: u64,
) -> Vec<Partition> {
    let part = |g: &str| {
        let h = Sha256::digest(format!("{seed}:{g}").as_bytes());
        let mut b = [0u8; 8];
        b.copy_from_slice(&h[..8]);
        let u = u64::from_le_bytes(b) as f64 / u64::MAX as f64;
        if u < test_fraction {
            Partition::Test
        } else if u < test_fraction + valid_fraction {
            Partition::Valid
        } else {
            Partition::Train
        }
    };
    convs.iter().map(|c| part(&c.group)).collect()
}

/// Summary shown to the user and stored next to the audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceProfile {
    pub conversations: usize,
    pub invalid_lines: usize,
    pub groups: usize,
    pub messages: usize,
    pub mean_turns: f64,
    pub with_tool_calls: usize,
    pub tool_calls: usize,
    /// Calls per tool name, most used first.
    pub tools_used: Vec<(String, usize)>,
    pub declared_tools: usize,
    pub formats: BTreeMap<String, usize>,
}

pub fn profile_traces(set: &TraceSet) -> TraceProfile {
    let convs = &set.conversations;
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    let mut declared = std::collections::BTreeSet::new();
    let mut formats = BTreeMap::new();
    let mut groups = std::collections::HashSet::new();
    for c in convs {
        for t in c.tool_calls() {
            *used.entry(t.name.clone()).or_default() += 1;
        }
        for d in &c.tools {
            declared.insert(d.name.clone());
        }
        let f = match c.format {
            TraceFormat::OpenAi => "openai",
            TraceFormat::ShareGpt => "sharegpt",
        };
        *formats.entry(f.to_owned()).or_default() += 1;
        groups.insert(c.group.as_str());
    }
    let messages: usize = convs.iter().map(|c| c.messages.len()).sum();
    let mut tools_used: Vec<_> = used.into_iter().collect();
    tools_used.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    TraceProfile {
        conversations: convs.len(),
        invalid_lines: set.invalid_lines.len(),
        groups: groups.len(),
        messages,
        mean_turns: messages as f64 / convs.len().max(1) as f64,
        with_tool_calls: convs.iter().filter(|c| c.tool_calls().next().is_some()).count(),
        tool_calls: tools_used.iter().map(|t| t.1).sum(),
        tools_used,
        declared_tools: declared.len(),
        formats,
    }
}

/// Removes every `open … close` block (an unterminated block runs to the end).
fn strip_blocks(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(open) {
        out.push_str(&rest[..i]);
        match rest[i + open.len()..].find(close) {
            Some(j) => rest = &rest[i + open.len() + j + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// Why a conversation cannot be used to train tool use, if it cannot (#47): a tool call
/// that does not parse, or nothing for the model to learn.
pub fn untrainable(c: &Conversation) -> Option<&'static str> {
    if c.messages.iter().any(|m| !m.broken_calls.is_empty()) {
        return Some("malformed tool call");
    }
    if c.tool_calls().any(|t| t.arguments.is_none()) {
        return Some("tool-call arguments are not JSON");
    }
    if !c.messages.iter().any(|m| m.role == Role::Assistant) {
        return Some("no assistant turn");
    }
    None
}

/// One conversation as a training record for `tempera-worker finetune` (#47): OpenAI-style
/// roles, tool calls as `{name, arguments}` objects and tools as `{name, parameters}`. The
/// `<tool_call>` / `<tools>` markup of ShareGPT/Hermes text is removed, because the chat
/// template of the base model renders calls and tools in that model's own format.
pub fn training_record(c: &Conversation) -> Value {
    let role = |r: Role| match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let messages: Vec<Value> = c
        .messages
        .iter()
        .map(|m| {
            let content = match (c.format, m.role) {
                (TraceFormat::ShareGpt, Role::Assistant) => {
                    strip_blocks(&m.content, "<tool_call>", "</tool_call>")
                }
                (TraceFormat::ShareGpt, Role::System) => strip_blocks(&m.content, "<tools>", "</tools>"),
                _ => m.content.clone(),
            };
            let mut v = serde_json::json!({ "role": role(m.role), "content": content });
            if !m.tool_calls.is_empty() {
                v["tool_calls"] = m
                    .tool_calls
                    .iter()
                    .map(|t| serde_json::json!({ "name": t.name, "arguments": t.arguments.clone().unwrap_or(Value::Null) }))
                    .collect();
            }
            v
        })
        .collect();
    serde_json::json!({
        "id": c.id,
        "group": c.group,
        "messages": messages,
        "tools": c.tools.iter().map(|t| serde_json::json!({ "name": t.name, "parameters": t.parameters })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const OPENAI: &str = r#"{"id":"c1","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Weather in Recife?"},{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Recife\"}"}}]},{"role":"tool","tool_call_id":"x","content":"31C"},{"role":"assistant","content":"It is 31C."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}"#;

    const HERMES: &str = r#"{"conversations":[{"from":"system","value":"You are a function calling AI. <tools> [{\"type\":\"function\",\"function\":{\"name\":\"get_stock\",\"parameters\":{\"type\":\"object\",\"properties\":{\"symbol\":{\"type\":\"string\"}},\"required\":[\"symbol\"]}}}] </tools>"},{"from":"human","value":"Price of PETR4?"},{"from":"gpt","value":"<tool_call>\n{\"name\": \"get_stock\", \"arguments\": {\"symbol\": \"PETR4\"}}\n</tool_call>"},{"from":"tool","value":"<tool_response>38.2</tool_response>"},{"from":"gpt","value":"38.2 BRL."}]}"#;

    #[test]
    fn parses_openai_format() {
        let c = parse_line(OPENAI, 1).unwrap();
        assert_eq!(c.format, TraceFormat::OpenAi);
        assert_eq!(c.id, "c1");
        assert_eq!(c.group, "c1");
        assert_eq!(c.messages.len(), 5);
        let calls: Vec<_> = c.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments.as_ref().unwrap()["city"], "Recife");
        assert_eq!(c.tools[0].name, "get_weather");
        assert_eq!(c.messages[3].role, Role::Tool);
    }

    #[test]
    fn parses_hermes_format_with_tools_in_system_prompt() {
        let c = parse_line(HERMES, 3).unwrap();
        assert_eq!(c.format, TraceFormat::ShareGpt);
        assert_eq!(c.id, "line-3");
        assert!(c.group.starts_with("prompt:"), "{}", c.group);
        let calls: Vec<_> = c.tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_stock");
        assert_eq!(c.tools.len(), 1);
        assert_eq!(c.tools[0].parameters["required"][0], "symbol");
    }

    #[test]
    fn broken_and_unparseable_calls_are_kept_for_the_audit() {
        let (calls, broken) = tool_calls_in_text(
            "<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call> <tool_call>{oops</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(broken, vec!["{oops".to_owned()]);
        // OpenAI arguments that are not JSON: call kept, arguments None.
        let line = r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{bad"}}]}]}"#;
        let c = parse_line(line, 1).unwrap();
        let call = c.tool_calls().next().unwrap();
        assert!(call.arguments.is_none());
        assert_eq!(call.raw, "{bad");
    }

    #[test]
    fn rejects_unknown_shapes() {
        assert!(parse_line("{\"text\":\"hi\"}", 1).is_err());
        assert!(parse_line("not json", 1).is_err());
        assert!(parse_line(r#"{"messages":[{"role":"robot","content":"x"}]}"#, 1).is_err());
        assert!(parse_line(r#"{"messages":[]}"#, 1).is_err());
    }

    #[test]
    fn task_id_wins_over_id_for_grouping() {
        let c = parse_line(
            r#"{"id":"a","task_id":"t9","messages":[{"role":"user","content":"x"}]}"#,
            1,
        )
        .unwrap();
        assert_eq!(c.group, "t9");
    }

    #[test]
    fn same_prompt_without_ids_shares_a_group() {
        let a = parse_line(r#"{"messages":[{"role":"user","content":"Book  a flight"}]}"#, 1).unwrap();
        let b = parse_line(r#"{"messages":[{"role":"user","content":"book a FLIGHT"}]}"#, 2).unwrap();
        assert_eq!(a.group, b.group);
    }

    #[test]
    fn split_keeps_groups_together_and_is_deterministic() {
        let convs: Vec<_> = (0..400)
            .map(|i| {
                parse_line(
                    &format!(
                        r#"{{"task_id":"t{}","messages":[{{"role":"user","content":"q{i}"}}]}}"#,
                        i / 4
                    ),
                    i + 1,
                )
                .unwrap()
            })
            .collect();
        let parts = split_by_group(&convs, 0.1, 0.2, 42);
        assert_eq!(parts, split_by_group(&convs, 0.1, 0.2, 42));
        let mut by_group: BTreeMap<&str, Partition> = BTreeMap::new();
        for (c, p) in convs.iter().zip(&parts) {
            assert_eq!(
                *by_group.entry(&c.group).or_insert(*p),
                *p,
                "group {} split",
                c.group
            );
        }
        let test = parts.iter().filter(|p| **p == Partition::Test).count();
        assert!((40..=130).contains(&test), "test={test}");
    }

    #[test]
    fn reads_file_and_profiles() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("t.jsonl");
        std::fs::write(&p, format!("{OPENAI}\n\n{HERMES}\nnot json\n")).unwrap();
        let set = read_traces(&p).unwrap();
        assert_eq!(set.conversations.len(), 2);
        assert_eq!(set.invalid_lines.len(), 1);
        assert_eq!(set.invalid_lines[0].0, 4);
        let prof = profile_traces(&set);
        assert_eq!(prof.tool_calls, 2);
        assert_eq!(prof.with_tool_calls, 2);
        assert_eq!(prof.declared_tools, 2);
        assert_eq!(prof.formats["openai"], 1);
        assert!(read_traces(&tmp.path().join("missing.jsonl")).is_err());
    }
    #[test]
    fn training_records_strip_hermes_markup_and_keep_structured_calls() {
        let line = r#"{"id":"h1","conversations":[{"from":"system","value":"Be brief. <tools>[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]</tools>"},{"from":"human","value":"Recife?"},{"from":"gpt","value":"Checking. <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Recife\"}}</tool_call>"},{"from":"tool","value":"31C"},{"from":"gpt","value":"31C."}]}"#;
        let c = parse_line(line, 1).unwrap();
        assert_eq!(untrainable(&c), None);
        let r = training_record(&c);
        assert_eq!(r["messages"][0]["content"], "Be brief.");
        assert_eq!(r["messages"][2]["content"], "Checking.");
        assert_eq!(r["messages"][2]["tool_calls"][0]["name"], "get_weather");
        assert_eq!(r["messages"][2]["tool_calls"][0]["arguments"]["city"], "Recife");
        assert_eq!(r["messages"][3]["role"], "tool");
        assert_eq!(r["tools"][0]["name"], "get_weather");
        assert!(r["messages"][4].get("tool_calls").is_none());

        // OpenAI text is kept verbatim.
        let o = training_record(&parse_line(OPENAI, 1).unwrap());
        assert_eq!(o["messages"][2]["tool_calls"][0]["arguments"]["city"], "Recife");
        assert_eq!(o["messages"][4]["content"], "It is 31C.");
    }

    #[test]
    fn untrainable_conversations_are_named() {
        let broken = r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"<tool_call>{nope</tool_call>"}]}"#;
        assert_eq!(
            untrainable(&parse_line(broken, 1).unwrap()),
            Some("malformed tool call")
        );
        let bad_args = r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{bad"}}]}]}"#;
        assert_eq!(
            untrainable(&parse_line(bad_args, 1).unwrap()),
            Some("tool-call arguments are not JSON")
        );
        let no_answer = r#"{"messages":[{"role":"user","content":"x"}]}"#;
        assert_eq!(
            untrainable(&parse_line(no_answer, 1).unwrap()),
            Some("no assistant turn")
        );
        assert_eq!(strip_blocks("a <t>x</t> b <t>y", "<t>", "</t>"), "a  b");
    }
}

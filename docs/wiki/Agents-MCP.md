# Agents (MCP)

`tempera mcp` exposes Têmpera as a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio (JSON-RPC 2.0, one message per line). Any MCP client works: Claude Code,
Claude Desktop, Hermes Agent, OpenClaw, Cursor, or your own agent through an MCP SDK.

## Setup

```bash
# Claude Code
claude mcp add tempera -- tempera mcp

# Any client that takes a JSON config
{ "mcpServers": { "tempera": { "command": "tempera", "args": ["mcp"],
                               "env": { "TEMPERA_HOME": "/path/to/.tempera" } } } }
```

Relative paths in tool arguments resolve against the server's working directory.

## Tools

| Tool | What it does | Writes? |
|---|---|---|
| `profile_data` | Rows, columns, types, nulls, cardinality of a CSV/Parquet/JSONL file | dataset cache |
| `init_spec` | Drafts a `tempera.yaml` for a file and a target (never overwrites) | the spec |
| `validate_spec` | Checks a spec against the schema | no |
| `spec_schema` | JSON Schema of `tempera.yaml` | no |
| `audit` | The 13-check data audit, no training | audit result |
| `run` | Audit → split → train candidates → winner → test set once → report | job, models, report |
| `estimate` | ETA and cost on each allowed machine; rents nothing | no |
| `audit_traces` | Audit agent traces (chat + tool calls) before fine-tuning | no |
| `finetune_traces` | Audit, then LoRA/QLoRA fine-tune a chat model on the local GPU; base → tuned ([Fine-Tuning](Fine-Tuning.md)) | job dir, adapter |
| `list_jobs` | Recent jobs | no |
| `show_job` | Runs of a job with validation scores and the winner | no |
| `report_path` | Path of a job's HTML report | no |

## Guardrails

The point of Têmpera is that rules are enforced by the system, not by discipline. An agent is
one more user who should not be able to skip them.

- **Same code path as a person.** Each tool runs the real `tempera` CLI in a subprocess. There is
  no separate "agent API" with weaker checks.
- **Red findings block.** `audit` and `run` return `BLOCKED` with the findings. The server's
  instructions ask the agent to show them to the user. No tool records a decision; dropping or
  accepting a column is an edit to `tempera.yaml` that the user should make or approve.
- **Test set once.** Enforced in SQLite, so an agent cannot loop "retrain until the test score
  improves".
- **No surprise spending.** `run` refuses a spec whose `compute.allow` lists a paid cloud
  provider (`runpod`, `vast`) unless the server was started with `TEMPERA_MCP_ALLOW_CLOUD=1`.
  `estimate` is always allowed. The budget cap (`budget.max_cost`) still applies when enabled.
- **Clean stdout.** Only protocol messages go to stdout; logs go to stderr; tool output has no
  ANSI colors.

## Limitations

- `run` is synchronous: the call returns when the job ends (minutes for large data). Clients
  with short tool timeouts should raise them.
- No progress notifications yet; live progress is in the web UI (`tempera up`).

## Tests

- Unit: `crates/tempera-server/src/mcp.rs` (protocol, argument mapping, cloud refusal).
- Integration: `crates/tempera-server/tests/mcp.rs` drives the churn example over stdio, from a
  blocked audit to a trained job and its report.
- Checked against the official MCP Python SDK client (`initialize`, `tools/list`, `tools/call`).

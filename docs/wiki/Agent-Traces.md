# Agent traces

Fine-tuning a small LLM to call tools starts from conversations with tool calls. Têmpera audits
them before any GPU time is spent, with the same rule as tabular data: red findings block until a
human decides (`--acknowledge <ID>`).

```bash
tempera traces audit data/agent_traces.jsonl --benchmark support_bench=data/agent_benchmark.jsonl
```

## Formats

One conversation per JSONL line, either:

- **OpenAI chat**: `messages` with `tool_calls` (arguments as a JSON string), `tools` with
  JSON-Schema `parameters`;
- **ShareGPT / Hermes function calling**: `conversations` with `from`/`value`, tool calls as
  `<tool_call>{"name", "arguments"}</tool_call>` and tools in a `tools` field or in
  `<tools>…</tools>` inside the system prompt.

Splits are **by group**, never by turn: the `task_id` (or `group`) field, else the conversation
`id`, else the normalized first user message. Related conversations should share a `task_id`.

## Checks

| Id | Check | Red when |
|---|---|---|
| T1 | Same or near-duplicate prompt in train and test (MinHash/LSH on word 3-shingles, Jaccard ≥ 0.8, verified exactly) | ≥ 1% of test |
| T2 | Exact duplicate conversations in train | ≥ 30% of train |
| T3 | Contamination with a benchmark you will report on (8-grams of user turns, item contaminated when > 50% of its 8-grams occur in training; Tülu 3) | > 2% of the benchmark |
| T4 | Personal data in messages, tool arguments or results (e-mail, CPF/CNPJ, cards, SSN, phone, IP); the evidence never echoes the values | ≥ 1% of conversations |
| T5 | Malformed tool calls: block not JSON, arguments not JSON, undeclared tool, missing required argument, wrong type, value outside `enum`, unknown argument with `additionalProperties: false` | ≥ 2% of calls |
| T6 | Few conversations | < 50 (yellow < 300) |
| T7 | One tool ≥ 80% of the calls (with ≥ 3 tools) | yellow only |
| T8 | No conversation answered without a tool (the model never learns not to call) | yellow only |
| T9 | Unreadable lines | > 5% |

Benchmark files: one prompt per line, as a JSON string, an object with `prompt`/`question`/`text`/
`query`, or a conversation.

## Example

`examples/data/agent_traces.jsonl` (from `examples/data/generate.py`) has planted problems: e-mail
addresses, broken and mistyped tool calls, prompts re-logged under new ids, and 8 of the 40
`agent_benchmark.jsonl` items. The audit finds all five (T1–T5) and blocks.
`agent_traces_clean.jsonl` is the same data without them: zero findings.

From an agent, the MCP tool `audit_traces` runs the same command.

Once the audit passes, `tempera traces finetune` trains on the same split: see
[LLM Fine-Tuning](Fine-Tuning.md).

## References

See [the literature review](../research/agents-and-m5-literature.md), topics 1–4.

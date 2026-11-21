# LLM Fine-Tuning (LoRA / QLoRA)

`tempera traces finetune` teaches a small chat model to call your tools, using conversations
you already have. It runs on the GPU you own: a 1–4B model fits in 10 GB with QLoRA.

```bash
uv sync --project workers/python --extra llm          # torch, transformers, peft, bitsandbytes
tempera traces finetune examples/data/agent_traces_clean.jsonl \
  --base Qwen/Qwen2.5-0.5B-Instruct --max-steps 40
```

## What happens, in order

1. **Audit.** The [trace audit](Agent-Traces.md) (T1–T9) runs first. Any unresolved red finding
   stops the command with exit code 2 before a file is written or a GPU is touched.
   `--acknowledge ID` records a human decision, as in `tempera audit`.
2. **Group split.** Conversations of the same task always share a partition (train, valid, test;
   default 70/10/20, `--valid`, `--test`, `--seed`).
3. **Clean records.** Conversations whose tool calls do not parse, or break the JSON Schema the
   conversation declares (what T5 reports), are left out and listed. ShareGPT/Hermes
   `<tool_call>` and `<tools>` markup is removed: the base model's chat template renders calls
   and tools in the format that model was trained on.
4. **Baseline.** The untouched base model is scored on the held-out conversations first.
5. **Train.** LoRA on every linear layer; 4-bit NF4 with double quantization when a CUDA GPU and
   bitsandbytes are present (QLoRA), plain LoRA otherwise. The loss covers assistant tokens
   only: prompts, tool outputs and tool specs are context, not targets. Conversations longer
   than `--max-len` are dropped, never truncated, because truncation cuts tool calls.
6. **Evaluate.** The adapter is scored on the same held-out conversations, and the terminal
   prints base → tuned.

```
  ● Result    base → tuned, same held-out conversations
      perplexity (lower is better)      2.237 → 1.001
      emits a tool call                 0.203 → 1.000
      well-formed call                  0.203 → 1.000
      right tool                        0.156 → 1.000
      right tool and arguments          0.062 → 1.000
```

Qwen2.5-0.5B-Instruct, 40 QLoRA steps (6 minutes) on an RTX 3080 10 GB, 562 train and 166 test
conversations of `examples/data/agent_traces_clean.jsonl`. The example traces come from a few
templates, so a perfect score there proves the pipeline, not the model: on real traces expect
smaller gains, and read `samples.jsonl` before trusting any number.

## Metrics

| Metric | Meaning |
|---|---|
| `perplexity` | exp of the mean loss on assistant tokens of held-out conversations |
| `tool_call_rate` | at a turn where the reference calls a tool, the model tries to call one |
| `valid_call_rate` | ... and the call parses as `{"name", "arguments"}` |
| `tool_name_accuracy` | ... and names the right tool |
| `exact_match` | ... and the arguments equal the reference (BFCL-style AST match) |

Each is generated greedily at the first tool-calling turn of up to 64 held-out conversations
(`--param eval_samples=N`). Exact match is strict on purpose: `"amount": 49.9` vs `49.90` is equal
JSON, but `"reason": "late"` vs `"delayed"` is not.

## Defaults and knobs

From [the literature review](../research/agents-and-m5-literature.md) (QLoRA, Dettmers et al. 2023;
LoRA Learns Less and Forgets Less, Biderman et al. 2024): tune learning rate and epochs before rank.

| Flag | Default | |
|---|---|---|
| `--r` / `--alpha` | 16 / 2×r | LoRA rank and scale |
| `--learning-rate` | 2e-4 | AdamW, 3% warm-up, cosine decay |
| `--epochs` / `--max-steps` | 1 / no cap | |
| `--quantization` | `auto` | `nf4` forces QLoRA, `none` forces 16/32-bit LoRA |
| `--max-len` | 2048 | tokens per conversation |
| `--param KEY=VALUE` | | `batch_size` (1), `grad_accum` (8), `dropout` (0.05), `eval_every` (50), `eval_samples` (64), `max_new_tokens` (160), `gradient_checkpointing` (true), `target_modules` (`all-linear`) |
| `--dry-run` | | write split and manifest, print the worker command, do not train |

## Output

Everything goes to `$TEMPERA_HOME/finetune/<job-id>/`:

```
data/{train,valid,test}.jsonl   the records the model saw (and did not see)
manifest.json                   worker input (docs/wiki/Contracts.md §11)
events.jsonl                    every event of the run
summary.json                    base_* and tuned metrics
model/adapter/                  adapter_model.safetensors, adapter_config.json, tokenizer
model/model.json                base model id + hub revision (or config hash), params, metrics
model/samples.jsonl             expected vs generated call for each evaluated turn
```

The adapter is small (about 35 MB for r=16 on a 0.5B model) and loads with
`PeftModel.from_pretrained(base, "model/adapter")`. `model.json` pins the base revision because
an adapter is only valid on the exact weights it was trained on.

## From an agent

The MCP tool `finetune_traces` runs the same command ([Agents (MCP)](Agents-MCP.md)). It has no
`acknowledge` argument: an agent can stop at BLOCKED and show the findings, but only the user can
accept one. It never rents a cloud machine.

## Not yet

- Merging the adapter and exporting GGUF for llama.cpp / Ollama (#70).
- Multi-turn agent evaluation in a sandbox (#68) and GRPO with verifiable rewards (#69).
- Fine-tuning jobs in the job list, the web UI and the HTML report (#66).

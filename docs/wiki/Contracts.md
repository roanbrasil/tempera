# Contracts and Specifications

## 1. `tempera.yaml`: job specification

The single source of truth for an experiment. The UI generates this file, the CLI consumes it, and it goes into git.

```yaml
version: 1
name: customer-churn

data:
  source: ./data/customers.parquet          # or s3://, postgres://, folder
  target: churned
  time_column: reference_date               # optional → temporal split
  group_column: customer_id                 # optional → group split
  sensitive: [gender, age_group]            # optional → fairness report
  drop: [chargeback_amount]                 # audit decisions

task:
  type: binary_classification               # inferred if omitted
  metric: pr_auc

audit:
  policy: block_on_red                      # block_on_red | warn_only
  advisor:
    model: qwen3-4b-q4                      # local SLM
    remote: false

split:
  strategy: auto                            # auto | random | stratified | group | temporal
  test_size: 0.2
  folds: 5
  seed: 42

strategy:
  init: auto                                # auto | scratch | pretrained
  candidates: auto                          # auto | list of algorithms
  baseline: true
  hpo:
    method: asha
    max_trials: 30

budget:
  max_cost: { amount: 10, currency: USD }
  max_time: 2h

compute:
  objective: cheapest_under                 # cheapest | fastest | cheapest_under
  deadline: 2h
  allow: [local, living-room-pc, runpod]    # allowed machines/providers
  spot: true
  max_restarts: 3

output:
  formats: [onnx, safetensors]
  report: true
```

The schema lives in JSON Schema (`schemas/tempera.schema.json`) and is validated in the CLI and on the server.

## 2. gRPC: agent ↔ server

```protobuf
syntax = "proto3";
package tempera.agent.v1;

service AgentControl {
  // The agent opens a bidirectional stream and keeps the connection alive.
  rpc Session(stream AgentMessage) returns (stream ServerMessage);
}

message AgentMessage {
  oneof msg {
    Register     register   = 1;  // hardware inventory, version
    Heartbeat    heartbeat  = 2;
    MetricBatch  metrics    = 3;
    LogBatch     logs       = 4;
    CheckpointReady checkpoint = 5;
    JobFinished  finished   = 6;
    JobFailed    failed     = 7;
    PreemptionNotice preemption = 8;
  }
}

message ServerMessage {
  oneof msg {
    StartJob   start  = 1;
    CancelJob  cancel = 2;
    RequestCheckpoint ckpt = 3;
    Shutdown   shutdown = 4;
  }
}
```

- Versioning: the `v1` package is forward-compatible; new fields are always optional.
- The agent reports `protocol_version` in `Register`. If it is incompatible, the server sends a `Shutdown` with update instructions.

## 3. Backend → agent events (stdout JSONL)

The backend (Python or Rust) **does not speak gRPC**. It writes one JSON line per event to stdout, which keeps adapters trivial:

```json
{"t":"metric","step":120,"epoch":2,"name":"val/pr_auc","value":0.731,"ts":1763251200.1}
{"t":"log","level":"info","msg":"epoch 2 done"}
{"t":"progress","step":120,"total_steps":2000}
{"t":"checkpoint","path":"/work/ckpt/step-120","step":120}
{"t":"artifact","kind":"model","format":"onnx","path":"/work/out/model.onnx"}
{"t":"done","summary":{"val/pr_auc":0.74}}
```

stderr is treated as a raw log.

## 4. Data: Arrow/Parquet layout

```
datasets/<sha256>/
├── data.parquet          # normalized data (inferred types)
├── profile.json          # statistical profile
├── audit.json            # audit findings + user decisions
└── splits/<split_hash>/
    ├── train.parquet
    ├── valid.parquet     # or folds/k=*.parquet
    ├── test.parquet
    └── split.json        # strategy, seed, counts
```

- Images and text: Parquet stores **paths** and metadata, and the blobs live in `blobs/`, addressed by hash.
- Schema conventions: the target goes in `__target__`, the optional weight in `__weight__`, the group in `__group__`, and the time in `__time__`.

## 5. Algorithm protocol (Python)

```python
class Algorithm(Protocol):
    name: str
    tasks: set[TaskType]

    def fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None: ...
    def predict(self, data: pa.Table) -> pa.Table: ...
    def save(self, model_dir: Path) -> None: ...
    def load(self, model_dir: Path) -> None: ...
    def save_checkpoint(self, path: Path) -> None: ...
    def resume_from(self, path: Path) -> None: ...
    def export(self, path: Path, fmt: Literal["onnx", "safetensors", "native"]) -> Path: ...
```

`ctx` carries the manifest data (`params`, `task`, `features`, `classes`, `seed`) and exposes
`ctx.log_metric()`, `ctx.should_stop()` (cancellation/budget/preemption), and `ctx.checkpoint_due()`.

## 6. Algorithm trait (Rust)

```rust
pub trait Algorithm: Send {
    fn name(&self) -> &'static str;
    fn tasks(&self) -> &'static [TaskType];
    fn fit(&mut self, train: &RecordBatch, valid: Option<&RecordBatch>, params: &Params, ctx: &mut RunContext) -> Result<()>;
    fn predict(&self, data: &RecordBatch) -> Result<RecordBatch>;
    fn save_checkpoint(&self, path: &Path) -> Result<()>;
    fn resume_from(&mut self, path: &Path) -> Result<()>;
    fn export(&self, path: &Path, fmt: ExportFormat) -> Result<PathBuf>;
}
```

## 7. Backend run manifest (M1)

Both backends (Python `tempera_worker`, native `tempera-agent native`) are subprocesses with the same
command-line contract. All inputs come in a JSON **manifest**; progress comes back as JSONL events
(section 3). Metrics are **not** computed by backends: they write predictions, and Rust computes every
metric in one place (`tempera_core::metrics`) so all algorithms are scored identically.

```bash
python -m tempera_worker fit     --manifest run.json   # train, then predict every entry in `predict`
python -m tempera_worker predict --manifest pred.json  # load a saved model, predict
tempera-agent native fit|predict --manifest ...        # same contract, Rust algorithms
```

### Model-ready Parquet (what backends read)

Produced by `tempera-data` after the audit. Feature columns keep their original names and are already
reduced to three kinds; special columns use reserved names and are **never** features:

| Column | Meaning |
|---|---|
| `<feature>` | `numeric` (Float64/Int64), `categorical` (Utf8, nulls allowed), `boolean` |
| `__target__` | Target. Utf8 for classification (labels as strings), Float64 for regression |
| `__group__`, `__time__` | Kept for traceability only, not features |

An empty string is treated as a missing categorical value (some ONNX converters need a string
sentinel), so `""` must not be used as a real category.

Datetime features are expanded in Rust (`<col>__year`, `__month`, `__dayofweek`, `__hour`). Free text and
ID-like columns are dropped by the audit in M1.

### `fit` manifest

```json
{
  "algorithm": "lightgbm",
  "task": "binary_classification",
  "classes": ["no", "yes"],
  "positive_class": "yes",
  "features": [
    {"name": "tenure_months", "kind": "numeric"},
    {"name": "plan", "kind": "categorical"},
    {"name": "has_discount", "kind": "boolean"}
  ],
  "params": {"num_leaves": 31},
  "seed": 42,
  "train": "/abs/splits/ab12/train.parquet",
  "valid": "/abs/splits/ab12/valid.parquet",
  "predict": [{"input": "/abs/splits/ab12/valid.parquet", "output": "/abs/run/valid_pred.parquet"}],
  "work_dir": "/abs/run",
  "export": ["onnx", "native"]
}
```

- `classes` is the sorted label list (classification only); `positive_class` is set for binary tasks.
- `valid` may be used for early stopping. The **test set never appears in a `fit` manifest.**
- Optional `"resume_from": "/abs/run/ckpt"` continues training from a checkpoint (a backend stopped by
  SIGTERM saves `<work_dir>/ckpt/`, emits a `checkpoint` event and exits with code 3).
- The model is saved under `<work_dir>/model/` (a `model.json` descriptor plus algorithm files).
  The `artifact` event of kind `model` / format `native` points at that **directory** (what a predict
  manifest's `model_dir` takes); other formats (e.g. `onnx`) point at the file. Each predictions file is
  announced with an `artifact` event of kind `predictions`.
- Backends that cannot export a requested format emit a `log` warning and continue; they never fail
  the run for it.
  Each exported file is announced with an `artifact` event.

### `predict` manifest

```json
{"algorithm": "lightgbm", "task": "binary_classification", "model_dir": "/abs/run/model",
 "input": "/abs/splits/ab12/test.parquet", "output": "/abs/run/test_pred.parquet"}
```

### Predictions Parquet

Same row order as the input.

| Task | Columns |
|---|---|
| Classification | `prediction` (Utf8 label) + `proba__<label>` (Float64) for **every** class |
| Regression | `prediction` (Float64) |

### Exit codes

`0` success (a `done` event was emitted) · `1` unexpected error · `2` bad manifest / unknown algorithm ·
`3` stopped by signal (checkpoint saved when supported).

## 8. Unsupervised tasks (clustering, anomaly detection)

Same manifests and commands as §7, with these differences.

**Model-ready Parquet.** `__target__` is absent for clustering. For anomaly detection it is present only
when the user has labels; it is used for **evaluation only** and backends must never read it during `fit`.

**Fit manifest.** `task` is `clustering` or `anomaly_detection`; `classes`/`positive_class` are omitted.
For anomaly detection with labels, the orchestrator passes `"positive_class"` (the anomaly label) so the
backend can name its prediction column consistently, but still must not train on it.

**Predictions Parquet** (same row order as the input):

| Task | Columns |
|---|---|
| Clustering | `cluster` (Int64; `-1` = noise, e.g. DBSCAN) |
| Anomaly detection | `anomaly_score` (Float64, **higher = more anomalous**) + `prediction` (Utf8: `anomaly` / `normal`) |

**Metrics.** Cluster quality depends on the model's own feature space (scaling, one-hot, PCA), so for
clustering the **backend** computes it and emits, for every `predict` entry and for the `predict`
command, metric events without a step:

| Event name | Meaning |
|---|---|
| `eval/silhouette` | silhouette on up to 10,000 rows, noise excluded (NaN → omitted when < 2 clusters) |
| `eval/davies_bouldin` | lower is better |
| `eval/n_clusters` | clusters found (noise excluded) |
| `eval/noise_fraction` | share of rows labeled `-1` |

For `fit`, these refer to the **first** `predict` entry (the validation set). Anomaly detection with labels
is scored in Rust from `anomaly_score` (PR-AUC, ROC-AUC); without labels there is nothing to rank and only
one candidate runs.

**Algorithms** (registry keys): clustering → `kmeans` (picks k in 2..10 by silhouette unless
`n_clusters` is given), `dbscan` (`eps` estimated from k-distance when not given), `pca_kmeans` (PCA keeping
95% of the variance, then k-means; emits `pca/explained_variance` and `pca/components`);
anomaly detection → `isolation_forest`.

## 9. HTTP API and live events (M2)

Served by `tempera up` (default `http://127.0.0.1:7070`); the web UI is a static build embedded in the
binary and served from `/`. JSON everywhere; errors are `{"error": "<message>", "issues": [...]?}` with a
4xx/5xx status. All paths are under `/api/v1`.

| Method | Path | Body → Response |
|---|---|---|
| GET | `/health` | → `{"ok": true, "version": "0.1.0"}` |
| POST | `/datasets` | multipart field `file` (csv/tsv/parquet/jsonl) → `{"dataset": Dataset, "profile": Profile}` |
| GET | `/datasets/{id}` | → `{"dataset": Dataset, "profile": Profile}` |
| POST | `/specs/validate` | `{"yaml": "..."}` → `{"valid": bool, "issues": [{"path","message"}]}` |
| POST | `/specs/draft` | `{"dataset_id", "target"?, "name"?}` → `{"yaml": "..."}` (same output as `tempera init`) |
| POST | `/audit` | `{"yaml": "..."}` → `{"inference": TaskInference, "audit": AuditReport}` |
| GET | `/algorithms` | → `[{"key","backend","tasks":[...]}]` |
| GET | `/machines` | → `[{"id":"local","kind":"local","hardware":{os,arch,cpu_cores,ram_bytes,gpus:[{name,vram_bytes}]},"status":"ready"}]` |
| POST | `/jobs` | `{"yaml": "..."}` → `202 {"job_id"}` (runs in the background) |
| GET | `/jobs` | → `[JobRow]` newest first |
| GET | `/jobs/{id}` | → `{"job": JobRow, "runs": [RunRow]}` |
| POST | `/jobs/{id}/cancel` | → `202 {}` |
| GET | `/jobs/{id}/report` | → the HTML report (`text/html`) |
| GET | `/runs/{id}/metrics?name=` | → `[MetricPoint]` |
| WS | `/jobs/{id}/events` | live event stream (below); on connect the server first replays past events |

`Dataset`, `Profile`, `TaskInference`, `AuditReport`, `JobRow`, `RunRow`, `MetricPoint` are the
serde shapes of the Rust types (`tempera-data`, `tempera-audit`, `tempera-track`).

**Live events** (one JSON object per WebSocket text message, field `type`):

```json
{"type":"stage","stage":"ingest|profile|audit|split|train|evaluate|report","message":"..."}
{"type":"audit","inference":{...},"audit":{...}}
{"type":"blocked","findings":[...]}
{"type":"run_started","run_id":"...","algorithm":"lightgbm","backend":"python","index":3,"total":7}
{"type":"metric","run_id":"...","name":"valid/binary_logloss","step":12,"value":0.41}
{"type":"progress","run_id":"...","step":120,"total_steps":500,"eta_s":31.5,"job_eta_s":95.0}
{"type":"log","run_id":"...","level":"warn","message":"..."}
{"type":"run_finished","run_id":"...","status":"succeeded","valid_score":0.87,"valid_metrics":{...},"duration_s":4.2,"error":null}
{"type":"winner","run_id":"...","algorithm":"lightgbm","test":{...Evaluation}}
{"type":"job_finished","status":"succeeded|failed|blocked|cancelled","report_url":"/api/v1/jobs/<id>/report","error":null}
```

`eta_s` is the remaining time of the current run (warm-up + EMA of the step time); `job_eta_s` adds an
estimate for the runs not started yet (median duration of finished runs).

## 10. Data-quality commands (A7, A8)

Run on the Python worker (`quality` extra: cleanlab, fairlearn) through the agent with
`COMMAND_QUALITY` / `COMMAND_FAIRNESS`. Same exit codes as §7; both announce their JSON output with an
`artifact` event of kind `report`. Results become audit findings; they never block or fail a job.

**`quality`** (label issues, classification only):
`{"task","classes","features","data","output","folds":5,"seed","max_rows":50000,"top_n":50}` →
`{"rows_checked","rows_total","subsampled","issues_found","estimated_noise_rate","per_class":{label:{"issues","rows"}},"top":[{"row","given","suggested","quality"}]}`.
`row` is the index in the input Parquet; rows are never removed. `done` summary: `issues_found`,
`estimated_noise_rate`.

**`fairness`** (winner's test predictions):
`{"task","classes","positive_class","truth","predictions","sensitive":{attr: parquet},"output"}` →
`{"task","rows","small_group_threshold":30,"attributes":{attr:{"groups":[{"group","count","metrics","small"}],"summary":{...}}}}`.
Binary metrics: `selection_rate`, `accuracy`, `tpr`, `fpr`; summary: demographic parity and equalized
odds (difference and ratio), equal opportunity and accuracy differences. Multiclass: `accuracy`.
Regression: `mae`, `rmse`, `mean_prediction`, `mean_truth`. Null group values form `"(missing)"`; groups
under 30 rows are flagged `small`. Model-ready splits carry sensitive attributes as `__sens__<name>`
(never features).

## 11. LLM fine-tuning (`finetune`, #47)

`python -m tempera_worker finetune --manifest M` (`llm` extra: torch, transformers, peft,
bitsandbytes). Written by `tempera traces finetune`; same exit codes as §7 (3 = stopped, adapter
checkpoint saved).

```json
{"base_model": "Qwen/Qwen2.5-0.5B-Instruct",
 "train": ".../data/train.jsonl", "valid": ".../data/valid.jsonl", "test": ".../data/test.jsonl",
 "work_dir": ".../work", "model_dir": ".../model", "seed": 42, "resume_from": null,
 "params": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear",
            "learning_rate": 2e-4, "weight_decay": 0.0, "warmup_ratio": 0.03, "epochs": 1.0,
            "max_steps": 0, "batch_size": 1, "grad_accum": 8, "max_len": 2048,
            "quantization": "auto", "gradient_checkpointing": true, "eval_every": 50,
            "eval_samples": 64, "max_new_tokens": 160}}
```

Unknown `params` keys are a manifest error (exit 2). `valid` and `test` are optional; the held-out
set for the baseline and the final score is `test`, else `valid`.

**Data lines**: `{"id", "messages": [{"role": "system|user|assistant|tool", "content",
"tool_calls": [{"name", "arguments": {...}}]}], "tools": [{"name", "parameters"}]}`. OpenAI chat
(`tool_calls[].function` with string `arguments`, `tools[].function`) is accepted too.

**Events**: `eval/base_perplexity` and `eval/base_{tool_call_rate, valid_call_rate,
tool_name_accuracy, exact_match}` before training; `train/loss` and `train/lr` per optimizer step;
`valid/loss`, `valid/perplexity` every `eval_every` steps and at the end; `eval/perplexity` and
`eval/<metric>` after training. Artifacts: `report` (`samples.jsonl`) and `model` with format
`safetensors` (the adapter directory). The `done` summary holds `base_<metric>` and `<metric>`.

**`model.json`**: `{"algorithm": "llm_lora", "task": "llm_fine_tune", "base_model": {"name",
"revision" | "config_sha256"}, "mode", "steps", "params", "seed", "metrics", "adapter": "adapter"}`.

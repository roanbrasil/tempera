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

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
  rpc Connect(stream AgentMessage) returns (stream ServerMessage);
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

    def fit(self, train: pa.Table, valid: pa.Table | None, params: dict, ctx: RunContext) -> None: ...
    def predict(self, data: pa.Table) -> pa.Table: ...
    def save_checkpoint(self, path: Path) -> None: ...
    def resume_from(self, path: Path) -> None: ...
    def export(self, path: Path, fmt: Literal["onnx", "safetensors", "native"]) -> Path: ...
```

`ctx` exposes `ctx.log_metric()`, `ctx.should_stop()` (cancellation/budget), and `ctx.checkpoint_due()`.

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

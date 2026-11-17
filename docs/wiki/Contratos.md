# Contratos e Especificações

## 1. `tempera.yaml`: especificação de job

Fonte única de verdade de um experimento. A UI gera esse arquivo, a CLI consome e ele vai para o git.

```yaml
version: 1
name: churn-clientes

data:
  source: ./dados/clientes.parquet          # ou s3://, postgres://, pasta
  target: cancelou
  time_column: data_referencia              # opcional → split temporal
  group_column: cliente_id                  # opcional → split por grupo
  sensitive: [genero, faixa_etaria]         # opcional → relatório de fairness
  drop: [valor_estornado]                   # decisões da auditoria

task:
  type: binary_classification               # inferido se omitido
  metric: pr_auc

audit:
  policy: block_on_red                      # block_on_red | warn_only
  advisor:
    model: qwen3-4b-q4                      # SLM local
    remote: false

split:
  strategy: auto                            # auto | random | stratified | group | temporal
  test_size: 0.2
  folds: 5
  seed: 42

strategy:
  init: auto                                # auto | scratch | pretrained
  candidates: auto                          # auto | lista de algoritmos
  baseline: true
  hpo:
    method: asha
    max_trials: 30

budget:
  max_cost: { amount: 10, currency: BRL }
  max_time: 2h

compute:
  objective: cheapest_under                 # cheapest | fastest | cheapest_under
  deadline: 2h
  allow: [local, pc-sala, runpod]           # máquinas/provedores permitidos
  spot: true
  max_restarts: 3

output:
  formats: [onnx, safetensors]
  report: true
```

O schema fica em JSON Schema (`schemas/tempera.schema.json`) e é validado na CLI e no server.

## 2. gRPC: agente ↔ server

```protobuf
syntax = "proto3";
package tempera.agent.v1;

service AgentControl {
  // O agente abre um stream bidirecional e mantém a conexão.
  rpc Connect(stream AgentMessage) returns (stream ServerMessage);
}

message AgentMessage {
  oneof msg {
    Register     register   = 1;  // inventário de hardware, versão
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

- Versionamento: o pacote `v1` é compatível para frente; campos novos são sempre opcionais.
- O agente informa `protocol_version` no `Register`. Se for incompatível, o server manda um `Shutdown` com instrução de atualização.

## 3. Eventos backend → agente (stdout JSONL)

O backend (Python ou Rust) **não fala gRPC**. Ele escreve uma linha JSON por evento no stdout, o que mantém os adaptadores triviais:

```json
{"t":"metric","step":120,"epoch":2,"name":"val/pr_auc","value":0.731,"ts":1763251200.1}
{"t":"log","level":"info","msg":"epoch 2 done"}
{"t":"progress","step":120,"total_steps":2000}
{"t":"checkpoint","path":"/work/ckpt/step-120","step":120}
{"t":"artifact","kind":"model","format":"onnx","path":"/work/out/model.onnx"}
{"t":"done","summary":{"val/pr_auc":0.74}}
```

O stderr é tratado como log bruto.

## 4. Dados: layout Arrow/Parquet

```
datasets/<sha256>/
├── data.parquet          # dados normalizados (tipos inferidos)
├── profile.json          # perfil estatístico
├── audit.json            # achados da auditoria + decisões do usuário
└── splits/<split_hash>/
    ├── train.parquet
    ├── valid.parquet     # ou folds/k=*.parquet
    ├── test.parquet
    └── split.json        # estratégia, seed, contagens
```

- Imagens e texto: o Parquet guarda **caminhos** e metadados, e os blobs ficam em `blobs/` endereçados por hash.
- Convenções de schema: o alvo vai em `__target__`, o peso opcional em `__weight__`, o grupo em `__group__` e o tempo em `__time__`.

## 5. Protocolo de algoritmo (Python)

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

`ctx` expõe `ctx.log_metric()`, `ctx.should_stop()` (cancelamento/orçamento) e `ctx.checkpoint_due()`.

## 6. Trait de algoritmo (Rust)

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

# Stack Tecnológica

> As versões exatas ficam nos lockfiles. Esta página registra **o quê** e **por quê**.

## Rust

| Área | Crate | Por quê |
|---|---|---|
| Async runtime | `tokio` | Padrão de fato |
| HTTP/API | `axum`, `tower-http` | Ergonômico, ecossistema tokio |
| gRPC | `tonic`, `prost` | Streaming bidirecional |
| Dados | `polars`, `arrow`, `parquet` | Desempenho e interoperabilidade com Python |
| ML clássico | `linfa`, `smartcore` | Nativo, sem Python |
| GBM nativo | `perpetual` | Fallback de boosting sem Python |
| Deep nativo | `burn` | Treino nativo multi-backend (CUDA/ROCm/Metal/wgpu) |
| Inferência | `ort` (ONNX Runtime), `candle` | Serving sem Python |
| SLM local | `mistral.rs` | Inferência local quantizada com saída estruturada (grammar/JSON) |
| SSH | `russh` | SSH puro Rust, sem depender do binário `ssh` |
| GPU info | `nvml-wrapper` | Inventário e telemetria NVIDIA |
| Persistência | `sqlx` + SQLite (→ Postgres) | Single-user simples, caminho para escala |
| Armazenamento | `object_store` | FS local, S3, R2, GCS com uma API só |
| CLI | `clap` | Padrão |
| UI embutida | `rust-embed` | Binário único |
| Segredos | `keyring`, `age` | Keyring do SO + fallback criptografado |
| Observabilidade | `tracing`, `tracing-subscriber` | Logs estruturados |
| Dedup | `gaoya` ou implementação própria de MinHash | Near-duplicates |
| Erros | `thiserror` (libs), `anyhow` (bins) | Convenção |

## Python (`workers/python`)

| Área | Biblioteca |
|---|---|
| Gerenciador | `uv` (ambientes reprodutíveis, rápido) |
| Dados | `pyarrow`, `polars` |
| Clássico | `scikit-learn`, `lightgbm`, `xgboost`, `catboost` |
| Deep | `torch`, `timm`, `transformers`, `peft`, `accelerate` |
| LLM fine-tune | `unsloth` (opcional, mais rápido e com menos memória) |
| Qualidade/fairness | `cleanlab`, `fairlearn` |
| Export | `onnx`, `skl2onnx`, `onnxmltools`, `safetensors` |
| Comunicação | stdout JSONL (sem gRPC no worker) |

Extras do Python como grupos opcionais (`tabular`, `vision`, `nlp`, `llm`): o agente instala só o que o job precisa, com ambientes menores e boot mais rápido.

## UI

| Área | Escolha | Por quê |
|---|---|---|
| Framework | SvelteKit (adapter-static) | Leve, rápido de desenvolver, build estático embutível |
| Gráficos | uPlot (curvas ao vivo) + ECharts (relatórios) | uPlot aguenta streaming de muitos pontos |
| Tempo real | WebSocket | Métricas/logs ao vivo |

Alternativa considerada: Leptos (UI em Rust). Foi descartada no início por produtividade e ecossistema de componentes (ver [ADR-0006](../adr/0006-ui-sveltekit.md)).

## Infra do projeto

- CI: GitHub Actions (fmt, clippy, testes, `cargo deny`, testes Python com `uv`, build multi-plataforma do agente).
- Release: `cargo-dist` (binários para Linux x86_64/aarch64 e macOS).
- Testes de integração de SSH: container com `sshd` na CI.
- Testes de nuvem: provider *mock* na CI; testes reais manuais e com orçamento mínimo.

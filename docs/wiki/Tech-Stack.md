# Tech Stack

> Exact versions live in the lockfiles. This page records the **what** and the **why**.

## Rust

| Area | Crate | Why |
|---|---|---|
| Async runtime | `tokio` | De facto standard |
| HTTP/API | `axum`, `tower-http` | Ergonomic, tokio ecosystem |
| gRPC | `tonic`, `prost` | Bidirectional streaming |
| Data | `polars`, `arrow`, `parquet` | Performance and interoperability with Python |
| Classical ML | `linfa`, `smartcore` | Native, no Python |
| Native GBM | `perpetual` | Boosting fallback without Python |
| Native deep learning | `burn` | Native multi-backend training (CUDA/ROCm/Metal/wgpu) |
| Inference | `ort` (ONNX Runtime), `candle` | Serving without Python |
| Local SLM | `mistral.rs` | Local quantized inference with structured output (grammar/JSON) |
| SSH | `russh` | Pure Rust SSH, no dependency on the `ssh` binary |
| GPU info | `nvml-wrapper` | NVIDIA inventory and telemetry |
| Persistence | `sqlx` + SQLite (→ Postgres) | Simple single-user, with a path to scale |
| Storage | `object_store` | Local FS, S3, R2, GCS with a single API |
| CLI | `clap` | Standard |
| Embedded UI | `rust-embed` | Single binary |
| Secrets | `keyring`, `age` | OS keyring + encrypted fallback |
| Observability | `tracing`, `tracing-subscriber` | Structured logs |
| Dedup | `gaoya` or a custom MinHash implementation | Near-duplicates |
| Errors | `thiserror` (libs), `anyhow` (bins) | Convention |

## Python (`workers/python`)

| Area | Library |
|---|---|
| Package manager | `uv` (reproducible environments, fast) |
| Data | `pyarrow`, `polars` |
| Classical | `scikit-learn`, `lightgbm`, `xgboost`, `catboost` |
| Deep learning | `torch`, `timm`, `transformers`, `peft`, `accelerate` |
| LLM fine-tuning | `unsloth` (optional, faster and uses less memory) |
| Quality/fairness | `cleanlab`, `fairlearn` |
| Export | `onnx`, `skl2onnx`, `onnxmltools`, `safetensors` |
| Communication | stdout JSONL (no gRPC in the worker) |

Python extras as optional groups (`tabular`, `vision`, `nlp`, `llm`): the agent installs only what the job needs, giving smaller environments and faster boot.

## UI

| Area | Choice | Why |
|---|---|---|
| Framework | SvelteKit (adapter-static) | Lightweight, fast to develop, embeddable static build |
| Charts | uPlot (live curves) + ECharts (reports) | uPlot handles streaming many points |
| Real time | WebSocket | Live metrics/logs |

Alternative considered: Leptos (UI in Rust). It was ruled out early on for productivity and component ecosystem reasons (see [ADR-0006](../adr/0006-ui-sveltekit.md)).

## Project infrastructure

- CI: GitHub Actions (fmt, clippy, tests, `cargo deny`, Python tests with `uv`, multi-platform agent build).
- Release: `cargo-dist` (binaries for Linux x86_64/aarch64 and macOS).
- SSH integration tests: a container with `sshd` in CI.
- Cloud tests: a *mock* provider in CI; real tests run manually with a minimal budget.

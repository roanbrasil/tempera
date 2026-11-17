# ADR-0001: Hybrid architecture: Rust (system) + Python (ML ecosystem)

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
We want a robust system, distributable as a single binary, with reliable remote execution, **and** access to the best of the ML ecosystem (PyTorch, HF, sklearn, LightGBM, Cleanlab, Fairlearn). The Rust ML ecosystem (Burn, Candle, linfa) is promising, but has far fewer pretrained models, algorithms, and data quality libraries.

## Decision
- **Rust** for the system: server, orchestrator, executors, agent, data (Polars), statistical audit, estimates, local SLM, serving.
- **Python** for training when the ecosystem is decisive.
- **Native Rust** for training when it is enough (simple tabular, small MLPs/CNNs). It starts instantly and does not require Python.
- Contracts: Arrow (data), safetensors/ONNX (models), gRPC (control).

## Alternatives considered
- **Everything in Rust:** no language compromises, but the catalog would be years behind. Rejected.
- **Everything in Python:** complete ecosystem, but loses the single binary, the agent's robustness, and data performance, and inherits environment hell. Rejected.

## Consequences
- (+) Each part uses the best tool.
- (−) Two toolchains and a more complex CI; workers need Python when the job requires it (mitigated with `uv` + caching).

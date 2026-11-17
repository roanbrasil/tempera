# ADR-0003: Process boundary (subprocess + JSONL) instead of FFI

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
We can integrate Python via PyO3 (in the same process) or via a subprocess.

## Decision
The Python backend runs as a **subprocess** of the agent and reports events as JSONL on stdout. PyO3 is used only for performance extensions called *from within* Python (e.g., a data loader written in Rust).

## Alternatives considered
- Embedded PyO3: lower latency, but a Python crash or OOM takes down the agent, there is GIL contention, and packaging becomes complex.
- gRPC inside the Python worker: more coupling and dependencies with no real gain.

## Consequences
- (+) Fault isolation, trivial adapters, and backends in any language.
- (−) Process overhead, negligible compared to training time.

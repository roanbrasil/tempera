# ADR-0004: SLM as advisor, statistics as decision-maker

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
LLMs understand column semantics (they can identify leakage and sensitive attributes), but they are non-deterministic, can hallucinate, and carry their own bias.

## Decision
- The SLM **classifies columns, explains findings, and suggests actions** as schema-validated JSON.
- The SLM **never** selects rows, removes data, or computes metrics.
- Every SLM suggestion requires user confirmation, and the decision is recorded.
- It runs locally by default (mistral.rs + a ~3–4B quantized model). Remote APIs only with opt-in.

## Consequences
- (+) Reproducibility (the decisions that count are statistical or human), privacy, and zero API cost.
- (−) Requires ~3 GB of RAM for the SLM; it is optional and the system works without it.

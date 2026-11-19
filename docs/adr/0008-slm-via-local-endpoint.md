# ADR-0008: The SLM advisor talks to a local OpenAI-compatible endpoint

- **Status:** Accepted (refines ADR-0004)
- **Date:** 2026-11-19

## Context
ADR-0004 planned to embed the SLM with `mistral.rs`. Embedding an inference engine in the `tempera`
binary adds a heavy build (CUDA/Metal variants), hundreds of MB of model management code and a second
place where GPUs are configured. Local model runners (Ollama, llama.cpp server, mistral.rs server, LM
Studio, vLLM) all expose the same OpenAI-compatible HTTP API with JSON-schema constrained output.

## Decision
- The advisor calls `POST {TEMPERA_LLM_URL}/chat/completions` (default `http://127.0.0.1:11434/v1`, Ollama)
  with a JSON schema; output that does not match the schema is discarded.
- The model is `audit.advisor.model`; if the server does not have it, the first model of the same family
  (default Qwen3 4B instruct) is used; with no model the audit runs without advice.
- Non-local URLs are refused unless `audit.advisor.remote: true` (column names and sample values leave
  the machine only by explicit choice).
- `TEMPERA_ADVISOR=off` disables it (CI, tests).
- Unchanged from ADR-0004: the advisor only adds non-blocking `SLM:*` suggestions; statistics decide.

## Alternatives considered
- **Embed mistral.rs:** one binary, but a much heavier build and duplicated GPU setup. Can return later as an
  optional feature that serves the same endpoint in-process.
- **Remote API only:** better models, but breaks the local-first and privacy promises.

## Consequences
- (+) Any local runner works; the user picks the model; zero inference code in Têmpera.
- (−) Users need a runner installed for advice (the audit works without it).

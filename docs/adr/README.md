# Architecture Decision Records

| ADR | Title | Status |
|---|---|---|
| [0001](0001-hybrid-rust-python.md) | Hybrid architecture: Rust (system) + Python (ML ecosystem) | Accepted |
| [0002](0002-arrow-data-contract.md) | Apache Arrow/Parquet as the data contract | Accepted |
| [0003](0003-process-boundary.md) | Process boundary (subprocess + JSONL) instead of FFI | Accepted |
| [0004](0004-slm-advisor.md) | SLM as advisor, statistics as decision-maker | Accepted |
| [0005](0005-persistence.md) | SQLite + local FS first, behind traits | Accepted |
| [0006](0006-ui-sveltekit.md) | Static SvelteKit UI embedded in the binary | Accepted |
| [0008](0008-slm-via-local-endpoint.md) | The SLM advisor talks to a local OpenAI-compatible endpoint | Accepted |

New ADR: copy the [template](template.md) with the next number.

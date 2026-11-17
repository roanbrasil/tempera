# tempera-worker

Python side of Têmpera ([ADR-0001](../../docs/adr/0001-hybrid-rust-python.md)). The agent starts it as a
subprocess ([ADR-0003](../../docs/adr/0003-process-boundary.md)); it trains with the Python ML ecosystem and
reports progress as JSONL events on stdout ([Contracts](../../docs/wiki/Contracts.md)).

```bash
uv sync                      # core only
uv sync --extra tabular      # + scikit-learn, LightGBM, XGBoost
uv run tempera-worker run --algorithm dummy --params '{"steps": 5}'
uv run pytest
```

Extras: `tabular`, `vision`, `nlp`, `llm`, `quality`. The agent installs only what a job needs.

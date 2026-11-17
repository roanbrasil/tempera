# tempera-worker

Python side of Têmpera ([ADR-0001](../../docs/adr/0001-hybrid-rust-python.md)). The agent starts it as a
subprocess ([ADR-0003](../../docs/adr/0003-process-boundary.md)); it trains with the Python ML ecosystem and
reports progress as JSONL events on stdout ([Contracts](../../docs/wiki/Contracts.md)).

```bash
uv sync                      # core only
uv sync --extra tabular      # + scikit-learn, LightGBM, XGBoost
uv run tempera-worker list
uv run tempera-worker fit --manifest run.json       # train, save <work_dir>/model/, predict
uv run tempera-worker predict --manifest pred.json  # load a saved model, predict
uv run pytest                # tabular tests are skipped unless the extra is installed
```

Manifests, the model-ready Parquet layout, predictions and exit codes are specified in
[Contracts, section 7](../../docs/wiki/Contracts.md). Tabular algorithms: `logistic_regression`,
`linear_regression`, `ridge`, `random_forest`, `lightgbm`, `xgboost` (plus `dummy` for smoke tests).
`TEMPERA_NUM_THREADS` caps the threads used by native libraries (default: every CPU).

Extras: `tabular`, `vision`, `nlp`, `llm`, `quality`. The agent installs only what a job needs.

# tempera-worker

Python side of Têmpera ([ADR-0001](../../docs/adr/0001-hybrid-rust-python.md)). The agent starts it as a
subprocess ([ADR-0003](../../docs/adr/0003-process-boundary.md)); it trains with the Python ML ecosystem and
reports progress as JSONL events on stdout ([Contracts](../../docs/wiki/Contracts.md)).

```bash
uv sync                      # core only
uv sync --extra tabular      # + scikit-learn, LightGBM, XGBoost, CatBoost
uv run tempera-worker list
uv run tempera-worker fit --manifest run.json       # train, save <work_dir>/model/, predict
uv run tempera-worker predict --manifest pred.json  # load a saved model, predict
uv sync --extra tabular --extra quality              # + Cleanlab, Fairlearn
uv run tempera-worker quality --manifest m.json     # suspected label errors (Data-Audit A7)
uv run tempera-worker fairness --manifest m.json    # per-group metrics of test predictions (A8)
uv sync --extra llm                                  # + torch, transformers, peft, bitsandbytes
uv run tempera-worker finetune --manifest ft.json   # LoRA/QLoRA on chat traces (Contracts §11)
uv run pytest                # tests of a missing extra are skipped; `-m "not gpu"` skips the CUDA run
```

Manifests, the model-ready Parquet layout, predictions and exit codes are specified in
[Contracts, sections 7 and 8](../../docs/wiki/Contracts.md). Tabular algorithms (`tabular` extra):

- supervised: `logistic_regression`, `linear_regression`, `ridge`, `lasso`, `elastic_net`, `sgd`,
  `random_forest`, `extra_trees`, `hist_gradient_boosting`, `adaboost`, `knn`, `svm`, `naive_bayes`, `mlp`,
  `lightgbm`, `xgboost`, `catboost`;
- clustering: `kmeans`, `dbscan`, `pca_kmeans`; anomaly detection: `isolation_forest`;
- plus `dummy` for smoke tests (no extra needed).
`TEMPERA_NUM_THREADS` caps the threads used by native libraries (default: every CPU).

Extras: `tabular`, `vision`, `nlp`, `llm`, `quality`. The agent installs only what a job needs.

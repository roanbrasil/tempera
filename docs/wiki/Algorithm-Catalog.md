# Algorithm Catalog

Each algorithm has **one preferred backend** and, when available, an **alternative backend**. Users don't need to know which one runs: the system picks based on data size, hardware, and availability.

Backend legend: 🦀 native Rust (`tempera-native`) · 🐍 Python (`tempera_worker`)

## Available today

Use the key in `strategy.candidates`, prefix with `native:`/`python:` to force a backend, or write
`candidates: [all]` to try every algorithm that supports the task. `auto` runs a curated ladder (below).

### Tabular: supervised

| Key | Classification | Regression | Backend | ONNX | In `auto` |
|---|---|---|---|---|---|
| `majority` / `mean` | ✓ / — | — / ✓ | builtin (trivial baseline) | — | always |
| `logistic_regression` | ✓ | | 🦀 linfa (🐍 sklearn with `python:`) | 🐍 | ✓ |
| `linear_regression` | | ✓ | 🦀 linfa (🐍 sklearn) | 🐍 | ✓ |
| `ridge` | | ✓ | 🦀 linfa (🐍 sklearn) | 🐍 | ✓ |
| `decision_tree` | ✓ | | 🦀 linfa | — | ✓ |
| `lasso`, `elastic_net` | | ✓ | 🐍 sklearn | 🐍 | |
| `sgd` | ✓ | ✓ | 🐍 sklearn | 🐍 | |
| `random_forest` | ✓ | ✓ | 🐍 sklearn | 🐍 | ✓ |
| `extra_trees` | ✓ | ✓ | 🐍 sklearn | 🐍 | |
| `hist_gradient_boosting` | ✓ | ✓ | 🐍 sklearn | best effort | |
| `adaboost` | ✓ | ✓ | 🐍 sklearn | best effort | |
| **`lightgbm`** | ✓ | ✓ | 🐍 lightgbm | 🐍 | ✓ |
| `xgboost` | ✓ | ✓ | 🐍 xgboost | numeric features only | ✓ |
| `catboost` | ✓ | ✓ | 🐍 catboost (native categoricals) | best effort | ✓ |
| `knn` | ✓ | ✓ | 🐍 sklearn | 🐍 | |
| `svm` | ✓ | ✓ | 🐍 sklearn (subsamples above 20k rows) | 🐍 | |
| `naive_bayes` | ✓ | | 🐍 sklearn | 🐍 | |
| `mlp` | ✓ | ✓ | 🐍 sklearn | 🐍 | |

"best effort": exported when the converter supports the fitted model, otherwise a warning and the
native format only. Every ONNX export is checked against the native model (relative tolerance 1e-5).

### Tabular: unsupervised (contract: [Contracts §8](Contracts.md))

| Key | Task | Backend | Ranked by |
|---|---|---|---|
| `kmeans` | clustering (k chosen by silhouette in 2..10 unless `n_clusters`) | 🐍 sklearn | silhouette |
| `pca_kmeans` | clustering after PCA (95% variance) | 🐍 sklearn | silhouette |
| `dbscan` | clustering with noise (`-1`) | 🐍 sklearn | silhouette |
| `isolation_forest` | anomaly detection | 🐍 sklearn | PR-AUC when labels exist, otherwise unranked |

A clustering job has no `data.target`. An anomaly-detection job may have one: it is used **only** to
evaluate the detector.

### Planned

| Algorithm | Backend | Milestone |
|---|---|---|
| Native k-means / DBSCAN / PCA | 🦀 linfa | M2 |
| Native GBM fallback | 🦀 perpetual | M2 |
| Tabular MLP on GPU | 🦀 Burn / 🐍 PyTorch | M5 |

## Time series

| Approach | Preferred | Milestone |
|---|---|---|
| Baselines (naive, seasonal naive) | 🦀 native | M2 |
| GBM with lags/windows (automatic feature engineering) | 🐍 LightGBM | M2 |
| Pretrained foundation models (zero-shot / fine-tune, e.g., Chronos) | 🐍 PyTorch | M5 |

Validation is always **walk-forward**.

## Computer vision

| Task | From scratch | Pretrained (default) | Milestone |
|---|---|---|---|
| Image classification | Small CNN (🦀 Burn / 🐍 PyTorch) | Fine-tune ResNet / ConvNeXt / ViT (🐍 timm) | M5 |
| Object detection | — | Fine-tune a modern detector (🐍) | post-M7 |

## Text / NLP

| Task | Pretrained (default) | Milestone |
|---|---|---|
| Text classification | Fine-tune an encoder (multilingual BERT-like) via 🐍 HF Transformers | M5 |
| Embeddings + lightweight classifier | Frozen encoder + logistic regression (cheap!) | M5 |
| Small LLM fine-tuning (instruction/chat) | LoRA/QLoRA via 🐍 PEFT/Unsloth | M5 |

## "From scratch" × "pretrained": recommender rules

The recommender produces a suggestion **with a justification**. The user can override it.

| Situation | Recommendation |
|---|---|
| Tabular (any size) | Train from scratch (GBM). Pretrained tabular models are still niche. |
| Images < ~50k examples | Fine-tune a pretrained model. From scratch tends to be worse **and** more expensive. |
| Images from a very different domain (X-ray, satellite) and > 100k | Try both, starting with fine-tuning |
| Text, classification | First frozen embeddings + logistic regression (pennies); then fine-tune the encoder if that isn't enough |
| LLM | Always LoRA/QLoRA. Training from scratch is out of scope. |
| Sensitive data where the pretrained model comes from an unknown source | Flag the license/provenance risk and let the user choose |

## Hyperparameter search

- **Default:** curated search spaces per algorithm, with few trials.
- **Strategy:** *successive halving / ASHA*, which kills bad configurations early.
- **Budget:** the search respects the job's time and cost cap.
- Early stopping on all iterative models.

## Metrics and report

| Task | Metrics |
|---|---|
| Classification | AUC-ROC, PR-AUC, F1, balanced accuracy, log-loss, calibration (ECE), confusion matrix |
| Regression | RMSE, MAE, MAPE, R², residuals |
| Time series | MAE/MASE per horizon |
| Clustering | Silhouette, Davies-Bouldin |
| All | Per-group fairness (if there is a sensitive attribute), cost, time |

## Adding a new algorithm

An algorithm is a **plugin** with:

1. A manifest (`algorithm.yaml`): name, supported tasks, backend, hyperparameter space, memory requirements, and whether it supports checkpointing.
2. An adapter implementation: a Python class that follows the `tempera_worker.Algorithm` protocol, or a Rust struct that implements `trait Algorithm`.
3. A contract test against a small, fixed dataset.

See [Contributing](Contributing.md).

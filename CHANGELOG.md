# Changelog

All notable changes to Têmpera. Versions follow [SemVer](https://semver.org/); while in `0.x`,
minor versions may break the spec format or the CLI.

## [0.1.0] - 2026-11-19

First public preview. Tabular ML end to end, on your machine or on your own machines over SSH.

### Highlights

- **From a file to an evaluated model:** `tempera init` drafts a `tempera.yaml` from your data;
  `tempera run` ingests, profiles, audits, splits, trains candidates, picks a winner on the
  validation set and evaluates it **once** on a held-out test set (enforced in SQLite).
- **Web UI** (`tempera up`): upload, data profile, audit decisions (drop/accept), strategy,
  machine choice, live training with ETA and cancel, HTML report. Embedded in the binary.
- **22 algorithms + 2 baselines:** native Rust (linear, ridge, logistic regression, decision tree)
  and Python (scikit-learn, LightGBM, XGBoost, CatBoost) for regression, binary/multiclass
  classification, clustering (k-means, PCA + k-means, DBSCAN) and anomaly detection
  (Isolation Forest). `candidates: [all]` runs everything that fits the task.
- **Data audit, 13 checks:** leakage, duplicates, repeated entities across splits, imbalance,
  train/test drift (PSI + adversarial AUC), label issues (cleanlab), group fairness (fairlearn),
  sensitive proxies, PII (e-mail, phone, CPF/CNPJ, SSN, cards, IP), unusable columns, small
  samples, informative missingness. Red findings block training until you decide.
  Benchmark: 13/13 planted problems found, zero findings on a clean dataset.
- **Local SLM advisor** (optional): any OpenAI-compatible local endpoint (e.g. Ollama) explains
  columns and suggests actions. It never blocks and never changes data.
- **Budget-aware hyperparameter search:** successive halving over growing train fractions; the
  default configuration is always evaluated, so search never makes the result worse.
- **SSH machines:** `tempera machines add/check`, then `compute: { allow: [gpu-box] }`. The agent
  is installed over the system OpenSSH, data is sent once (content-addressed cache), results
  come back automatically.

### Install

Download the archive for your platform below, extract it and run `./tempera --help`. Keep
`tempera-agent` and `workers/` next to `tempera`. Python algorithms need
[uv](https://docs.astral.sh/uv/); without it, the native Rust algorithms still run.

### Known limitations

- Cloud GPUs (M6) and deep learning (M5) are not in this release.
- Windows: use WSL2.
- Live job events are kept in memory; after a server restart the UI shows the stored results.
- The license is not decided yet (see issue #9).

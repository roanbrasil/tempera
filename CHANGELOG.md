# Changelog

All notable changes to Têmpera. Versions follow [SemVer](https://semver.org/); while in `0.x`,
minor versions may break the spec format or the CLI.

## [Unreleased]

### Added

- `tempera mcp`: MCP server on stdio (protocol 2025-06-18, also 2025-03-26 and 2024-11-05) with
  11 tools: `profile_data`, `init_spec`, `validate_spec`, `spec_schema`, `audit`, `run`,
  `estimate`, `list_jobs`, `show_job`, `report_path`. Tools run the real CLI, so the audit and the
  test-once rule apply to agents too. Paid cloud jobs need `TEMPERA_MCP_ALLOW_CLOUD=1`.
- `tempera traces audit`: reads agent traces (OpenAI chat and ShareGPT/Hermes function-calling
  JSONL), splits them by task, and runs 9 checks before LLM fine-tuning: prompts shared between
  train and test (MinHash/LSH), duplicates, benchmark contamination (Tülu 3 8-gram rule),
  personal data, malformed tool calls against the declared JSON Schema, size, tool balance, no
  "don't call a tool" examples, unreadable lines. Also as the MCP tool `audit_traces`.
- Example agent traces with planted problems and a clean twin (`examples/data/agent_*.jsonl`).
- Literature review for agents and M5: `docs/research/agents-and-m5-literature.md`.

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

### Cloud GPUs (M6, preview)

New `tempera-cloud` crate.
- Providers: RunPod (GraphQL prices, REST pods) and Vast.ai (reliability, bandwidth and disk
  filters). Both were built from public API docs and are not yet tested against a live account.
  A `mock` provider can boot a real sshd container and simulate preemption.
- Pricing catalog: timestamped snapshots under `<TEMPERA_HOME>/pricing/`.
- Estimates before renting: ETA and cost per machine, ranked by `cheapest`, `fastest` or
  `cheapest_under`, with a one-line rationale. This machine competes at $0 when `local` is allowed.
- Guardrails:
  - A job whose estimate is over `budget.max_cost` is refused before anything is rented.
  - Spend alerts at 80% of the budget and the job is cancelled at 100%.
  - Leases with heartbeats are written before launch.
  - The instance is always terminated: on success, error, cancel or Ctrl-C.
  - The orphan reaper runs on `tempera cloud reap` and in the background under `tempera up`.
- Estimated vs. actual cost and a "savings vs. naive run" line appear in the CLI, the report and
  the live job page.
- `tempera cloud offers | sync | prices | estimate | instances | reap`.

### Dependencies

- Linear and logistic regression use the crate's own solvers. This drops the unmaintained `paste`
  crate.
- sqlx 0.9.1, which fixes RUSTSEC-2026-0459. MSRV is now 1.94.

### Install

Download the archive for your platform below, extract it and run `./tempera --help`. Keep
`tempera-agent` and `workers/` next to `tempera`. Python algorithms need
[uv](https://docs.astral.sh/uv/); without it, the native Rust algorithms still run.

### Known limitations

- Cloud GPUs are a preview: the RunPod and Vast.ai clients have not run against a live account
  yet, and cost estimates are not calibrated. Deep learning (M5) is not in this release.
- Windows: use WSL2.
- Live job events are kept in memory; after a server restart the UI shows the stored results.
- The license is not decided yet (see issue #9).

# Data Audit

> **Principle:** statistics decide, and the SLM explains and suggests. The LLM never selects rows on its own (see [ADR-0004](../adr/0004-slm-advisor.md)).

"Biased data" covers very different problems. Each one has the right technique:

## Checks

| # | Problem | How we detect it | Default severity | Suggested action |
|---|---|---|---|---|
| A1 | **Target leakage** | Single-feature importance (AUC of a single-feature model ≈ 1.0); extreme correlation/mutual information; the SLM assesses semantics ("does it only exist after the event?") | 🔴 | Drop the column |
| A2 | **Train↔test leakage** through duplicates | Exact hash + near-duplicates (MinHash/LSH for text, pHash for images) | 🔴 | Deduplicate before the split |
| A3 | **Group leakage** | Entities (e.g., `customer_id`) appearing on both sides; the SLM suggests which column represents the entity | 🔴 | Group split (GroupKFold) |
| A4 | **Temporal leakage** | A time column exists and the split is random | 🔴 | Temporal split, or walk-forward for time series |
| A5 | **Class imbalance** | Target distribution | 🟡 | Stratified split, appropriate metric (PR-AUC/F1), `class_weight` |
| A6 | **Distribution shift** train↔test or train↔production | *Adversarial validation* (train×test classifier; AUC > 0.6 is a warning); PSI/KS per feature | 🟡 | Investigate; reweight; different split |
| A7 | **Noisy labels** | *Confident learning* (Cleanlab) with out-of-fold predictions | 🟡 | Review the top-N suspects; never remove automatically |
| A8 | **Fairness bias** | Per-sensitive-group metrics (demographic parity, equal opportunity) via Fairlearn | 🟡 | Report; optional mitigation |
| A9 | **Sensitive attribute proxies** | A feature predicts the sensitive attribute (e.g., ZIP code → race); the SLM points out semantic proxies | 🟡 | Consider removal or mitigation |
| A10 | **Useless/dangerous columns** | IDs, constants, high cardinality, free text containing personal data | 🟢/🟡 | Drop or treat |
| A11 | **Personal data** (LGPD/GDPR) | Regex + local NER (CPF, SSN, email, phone, name) | 🟡 | Mask or drop; record it |
| A12 | **Sample too small** | Per-task rules (e.g., < 50 per class); learning curve | 🟡 | Suggest a pretrained model or collecting more data |
| A13 | **Structural nulls/outliers** | Profiling; missingness patterns correlated with the target | 🟢/🟡 | Imputation or an indicator feature |

## Selecting "unbiased" training and test data

The system builds the split in this order:

1. **Deduplicate** (A2).
2. **Choose the split strategy** based on the audit: temporal (A4) > group (A3) > stratified (A5) > random.
3. **Hold out the test set first** (default: 15–20%). It is set aside and used **only once** at the end.
4. Train/validation with stratified or group k-fold for model selection.
5. **Adversarial validation** on the final split to confirm that train and test come from the same distribution.
6. Everything is recorded: seed, strategy, and the hashes of each partition.

## The role of the SLM

**Where it helps** (semantics that statistics can't see):

- Reading column names, descriptions, and samples and classifying each one: `identifier`, `time`, `post-event-target` (suspected leakage), `sensitive`, `sensitive-proxy`, `free-text`, `personal-data`.
- Explaining each finding in the report in plain language, with the recommended action.
- Suggesting the group column and the time column.
- Generating a readable description of the dataset for the final report.

**Where it does NOT come in:**

- Choosing which rows go to training or testing.
- Removing data automatically.
- Computing metrics.

**Structured output** (validated by JSON Schema; if the JSON is invalid, the system discards it and falls back to the statistical rule):

```json
{
  "column": "chargeback_amount",
  "role": "post-event-target",
  "confidence": 0.86,
  "evidence": "A chargeback only happens after confirmed fraud; correlation with target = 0.97",
  "suggested_action": "drop",
  "requires_user_confirmation": true
}
```

### Candidate models (local, quantized)

| Model | Size | Approx. RAM (Q4) | Note |
|---|---|---|---|
| Qwen3 4B | 4B | ~3 GB | Good at structured JSON and at Portuguese; **suggested default** |
| Phi-4-mini | 3.8B | ~3 GB | Good reasoning for its size |
| Gemma 3 4B | 4B | ~3 GB | Alternative |
| Llama 3.2 3B | 3B | ~2.5 GB | Lightweight |
| SmolLM3 3B | 3B | ~2.5 GB | Very lightweight, for weak machines |

The model is **configurable**. The default choice comes from an internal column classification benchmark (an M3 issue). A remote API (Claude, OpenAI, Ollama) can also be used by explicit opt-in, with a privacy warning.

## Audit benchmark

To measure whether the audit really works, we build public datasets with **planted problems** (leakage, duplicates, flipped labels, proxies) and measure the recall and precision of each check. This runs in CI.

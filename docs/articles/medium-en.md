# One column, AUC 0.99, and the model was wrong: honest ML training is a systems problem

*I built Têmpera, a platform that audits your data before it trains anything and runs on the cheapest machine that gets the job done. Rust for the system, Python for the ML ecosystem.*

A churn dataset, a column called `refund_amount`, and a single-feature AUC of 0.993. If you've ever built a churn model, you already know the ending. Refunds only exist *after* a customer cancels. The model didn't learn to predict churn; it learned to read the future. It looks great in a notebook. In production that column arrives empty and the model falls apart.

Nobody makes this mistake because they've never heard of target leakage. They make it because nothing in their workflow stops them. That's what convinced me to stop treating "train it properly" as personal discipline and start treating it as a systems engineering problem.

## Discipline doesn't scale. Guardrails do.

The rules of an honest training run fit on an index card. Hold out the test set first and look at it exactly once. Never let the same entity land in both train and test. Distrust any feature that's too good. Always beat a dumb baseline. Each of those rules depends on someone remembering it at 2 a.m. before a deadline.

Têmpera turns them into system state:

- The split happens before training, picking temporal > group > stratified > random depending on what the data supports.
- The test set is held out and can be read **once**. That's recorded in SQLite, and a second read is refused, not merely discouraged.
- A 13-check audit runs before any model trains: single-feature leakage, duplicates, entities repeated across partitions, class imbalance, train/test drift (PSI plus adversarial AUC), suspicious labels via cleanlab, group fairness via fairlearn, proxies for sensitive attributes, PII, tiny samples, and informative missingness.
- A red finding blocks training. You either drop the column or accept it explicitly, and that decision is written into `tempera.yaml`, where a reviewer can see it.

With `refund_amount` still in the data:

```text
  ● Audit     1 red · 2 yellow · 6 features used · 2 columns dropped
      RED    A1:refund_amount       Probable target leakage · needs a decision
             `refund_amount` alone predicts `churned` with single-feature AUC = 0.993.
  ✗ Training blocked by the audit (policy: block_on_red).
```

To check whether the audit is any good, I built a benchmark with one planted problem per check. It catches 13 out of 13, and it reports nothing at all on a clean dataset. I care about the second number as much as the first. An audit that flags everything trains you to ignore it.

## A small local model explains; statistics decide

Statistics can tell you `refund_amount` predicts the target suspiciously well. It can't tell you *why*. For that I added a small language model running locally: Qwen3 4B on Ollama, reading column names, types and summary statistics.

On its own, it labeled `refund_amount` as post-event information with 0.99 confidence. Two unrelated methods landing on the same column is what makes the finding believable.

The architectural rule is strict, though: **statistics decide, the SLM suggests.** The advisor never blocks a run, never touches the data, and by default refuses to talk to any endpoint that isn't on your machine. If you work with health or financial records, nothing leaves the box. A quantized 4B model will hallucinate now and then, and letting it gate a pipeline would just swap one failure mode for another.

## Why not all Rust?

That was the first thing I wanted. Then I looked at what the ML ecosystem needs: LightGBM, XGBoost, CatBoost, cleanlab, fairlearn. Rewriting them in Rust would take years and end up worse. So each language does what it's best at:

| Rust | Python |
|---|---|
| CLI, web server with the UI embedded, orchestration | scikit-learn, LightGBM, XGBoost, CatBoost |
| Ingestion, profiling, splitting, audit (Polars) | cleanlab, fairlearn |
| Final metrics, tracking (SQLite), reports | ONNX export |
| Remote agent over gRPC, SSH execution | |

The boundary is a process, not FFI. Rust writes Parquet plus a JSON manifest, launches the Python worker, and reads JSONL events from its stdout. When a Python run crashes, the system doesn't: that run fails with its stderr attached, and the job moves on.

Metrics are computed in Rust from the predictions each worker returns. A LightGBM model and a native one are scored by the same code, so no library gets to compute PR-AUC its own way.

There's also a plain resource argument. The first version of the profiler used 6.8 GB of RAM on a large file. With Polars streaming aggregates and sampling, it now runs inside a cgroup capped at 2 GB. On an ordinary laptop, that's the difference between a result and a frozen machine.

## The time the great result was the bug

I generated a fraud-detection example and got PR-AUC 1.0 on the first run. The synthetic fraud was simply too easy to separate. I rewrote the generator so 40% of the fraud mimics normal transactions, like it does in real life. Isolation Forest now scores PR-AUC 0.77 on the test set against a 1.5% base rate. It's a less impressive number and a much more honest one.

Same lesson as `refund_amount`: a perfect metric is a symptom. The tool should push you to distrust it.

## The cheapest machine that gets the job done

A lot of ML spend is the wrong hardware: a rented GPU for a tabular problem a CPU handles fine, or a hyperparameter search that trains every bad configuration to completion.

What works today:

- **22 algorithms plus 2 baselines** (majority class and mean) across supervised learning, clustering and anomaly detection. `candidates: [all]` runs everything that fits the task, and everything has to beat the baseline.
- **Successive halving** for hyperparameter search: candidates train on 1/9 of the data, the best move up to 1/3, and only the finalists see all of it. It used about 36% of the compute of a full random search. I aimed for 30% and I'm not there yet. Each algorithm's default config is always trial 1 and is never eliminated, so the search can't make the result worse.
- **Any machine becomes a worker over SSH.** The agent installs itself, and data is sent once into a content-addressed cache. I ran the same job locally and over SSH (in tests, against a container running sshd as the remote box) and got identical scores.

```bash
tempera machines add gpu-box --host 192.168.0.42 --user ana
tempera machines check gpu-box
tempera run churn.yaml   # with compute: { allow: [gpu-box] } in the spec
```

## What isn't done

I'd rather say it before you find out:

- **Cheap cloud GPUs (milestone M6)** don't exist yet. The plan is RunPod and Vast.ai, a memory estimate before renting, the smallest GPU that fits, spot instances with checkpointing, and a hard budget cap. It's next, because it's the core of the "cheap" promise.
- **Deep learning (M5)** hasn't started either: vision, NLP, LoRA fine-tuning. Today Têmpera is strong on tabular data and nothing else.
- The live training dashboard doesn't show cost, GPU usage or overfitting alerts yet.

## Try to break it

The code is at [github.com/roanbrasil/tempera](https://github.com/roanbrasil/tempera). Clone it, run `tempera run churn.yaml` in `examples/`, and watch the audit stop the run before you would have. Then point it at your own data. If it misses a leak you caught by hand, open an issue with the case. That's exactly what I want in the benchmark.

# Cost and Estimation

> **Implemented (M6, preview):** pricing snapshots, the tabular ETA/cost prior, machine ranking
> with a rationale, budget refusal and meter, and the savings line live in `crates/tempera-cloud`
> (`catalog`, `estimate`, `guard`). Deep-learning memory/FLOPs estimation (§1, §2) arrives with M5.

Têmpera treats **cost** and **time** as first-class metrics, and estimates both **before** spending anything.

## 1. Memory estimation (does it fit or not?)

Hitting OOM after 40 paid minutes is the most common kind of waste. The estimate is made before choosing the machine.

### Deep learning

```
mem_total ≈ weights + gradients + optimizer_states + activations + overhead

weights       = P × bytes_per_param                 (fp32=4, bf16=2, int4≈0.5)
gradients     = P_trainable × bytes_grad
optimizer     = P_trainable × 8   (Adam: 2 fp32 states)
activations   ≈ f(architecture, batch, seq_len)     → measured by profiling
overhead      ≈ 10–20% + CUDA context (~0.5–1 GB)
```

- **LoRA/QLoRA:** the trainable parameters are a tiny fraction of the total, and the base model can be in 4 bits. This is what makes a 7–8B LLM feasible on a 24 GB GPU.
- **Calibration by profiling:** whenever possible, run 2–3 steps on an available machine (even on CPU with a reduced batch) to measure the real activations instead of guessing.

### Tabular

A heuristic per algorithm: `rows × columns × bytes × factor` (the GBM factor depends on binning). It usually fits on the CPU.

## 2. Time estimation (ETA)

Three sources, combined:

1. **A priori** (before running, to compare machines):
   - Transformers: `FLOPs ≈ 6 × P × tokens` for full training, and `2 × P × tokens` for the forward pass. Time comes from `FLOPs / (GPU_peak × MFU)`, with an initial MFU (effective utilization) of 30–40%.
   - Other models: an internal benchmark table (algorithm × hardware × size), fed by the runs themselves.
2. **Warm-up** (first N steps, discarding the first one): measures the real throughput and extrapolates.
3. **Online:** exponential moving average of time per step, plus evaluation/checkpoint time, plus boot and download time in the cloud.

`ETA = boot + env_setup + download + remaining_steps × t_step + evaluations + final_upload`

ETA error becomes a product metric (target: < 15% after warm-up).

## 3. Cost estimation

```
cost = hourly_price × (ETA + boot + margin) + storage + egress
```

- **Price catalog:** synced from provider APIs (RunPod, Vast...) and stored with a date, because prices change: `<TEMPERA_HOME>/pricing/<provider>-<unix time>.json`, one file per sync (that is the history). `tempera cloud sync` refreshes; every cloud `tempera run` also saves the prices it used; `tempera cloud prices` shows the latest prices and their age (stale after 6 h).
- **Your own machines:** zero cost by default, with the option to enter the energy cost (W × US$/kWh) for those who want to be exact.
- **Throughput normalization:** "US$ per job" matters more than "US$ per hour". A GPU that is 2x more expensive and 3x faster comes out cheaper overall.

### The tabular prior (implemented)

`work = Σ_algorithms seconds_per_Mcell(algorithm) × rows × columns / 10⁶ × (1 + 0.36 × hpo_trials)`,
plus a per-fit overhead (≈0.2 s native, ≈4 s Python) and a few passes for profiling/audit. On a
machine with `n` cores: `ETA = boot + overhead + work / amdahl(n, p = 0.85)`, with boot ≈ 180 s for
a cloud instance and ≈ 15 s for an SSH machine. Cloud cost = `price × ETA` (billing starts at
launch). The per-algorithm constants are priors, not measurements; calibrating them from finished
runs is the next step toward the "< 20% cost error" target, which is **not yet measured**.

## 4. Machine recommendation

For a job with memory requirement M and an estimated ETA:

1. Filter the machines on which the job **fits** (with margin).
2. Compute the ETA and cost for each one.
3. Sort according to the user's objective:
   - `cheapest`: lowest total cost
   - `fastest`: lowest ETA
   - `cheapest_under: 2h`: the cheapest that finishes in under 2 h (default)
4. Show the justification ("the A100 is 1.4x faster but 3x more expensive for this job").

## 5. Savings strategies (on by default)

| Strategy | Detail |
|---|---|
| **Baseline first** | Cheap models set the bar; deep learning only runs if there is a chance of improvement |
| **Local/LAN first** | If you have your own machine that fits and finishes on time, it comes first |
| **Spot + checkpointing** | Default in the cloud for jobs longer than 15 min; automatic resumption |
| **Early stopping** | On every iterative model |
| **ASHA for HPO** | Kills bad configurations early |
| **Mixed precision (bf16)** | Default on GPUs that support it |
| **Data and environment caching** | Doesn't re-download or reinstall what is already there |
| **Auto-shutdown + reaper** | No GPU is left running for nothing |
| **Budget cap** | Hard limit, with a checkpoint when it is reached |
| **Egress-free storage** | Artifacts in S3-compatible storage with no egress fees (e.g., Cloudflare R2), or local |

## 6. Transparency

At the end of every job, the report shows:

```
Estimated cost: US$ 0.40   Actual cost: US$ 0.36   (−10%)
Estimated ETA:  38 min     Actual:      41 min     (+8%)
Savings vs. naive run (on-demand A100, no early stopping): US$ 2.08 (85%)
```

The **savings** line is the product's selling point. Today it compares the actual cost with the
largest on-demand GPU among the offers, running the same work (×1/0.36 when a successive-halving
search replaced an exhaustive one). It is printed by `tempera run` and `tempera cloud estimate`,
stored in `report.json` (`cost.savings`) and shown in the HTML report and the live job page.

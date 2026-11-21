# Vision and Problem

## Vision

Anyone with data and a question should be able to train a **reliable** model without spending more than necessary, without setting up infrastructure, and without falling into the classic ML traps.

## The problem

Training a model today requires mastering four worlds at once:

| World | Where people get hurt |
|---|---|
| **Data** | Leakage between train and test, duplicates, mislabeled data, class imbalance, sampling bias. The model "works" in the notebook and fails in production. |
| **Modeling** | Choosing between a classical algorithm and deep learning, between training from scratch and fine-tuning, and setting the hyperparameters. |
| **Infrastructure** | CUDA, drivers, PyTorch versions, remote machines, SSH, Docker. |
| **Cost** | A GPU left running, an instance far too expensive for the problem, a training run that runs out of memory after 3 paid hours. |

Existing tools cover **one or two** of these worlds:

- **AutoML** (AutoGluon, H2O, Vertex AutoML): good at modeling, but locked into one cloud or without control over where it runs.
- **GPU orchestration** (SkyPilot, dstack): good at infrastructure and cost, but knows nothing about data or modeling.
- **Tracking** (MLflow, W&B, ClearML): good at monitoring, but you still write everything else.
- **Data quality** (Cleanlab, Evidently, Great Expectations): isolated tools that need an expert to stitch them together.

**Nobody brings all four together in a guided, local-first flow with an obsession for cost.**

## Personas

### 1. Product developer at an SMB ("Ana")
- Knows how to code but is not a data scientist. Has a customer spreadsheet and wants to predict churn.
- **Pain:** doesn't know whether the result is reliable and is afraid of overspending on the cloud.
- **Gain:** a guided flow, an audit that warns "the `cancellation_date` column leaks the target", and a baseline that runs on her CPU for pennies.

### 2. Data scientist with a home lab ("Bruno")
- Has a laptop plus a PC with an RTX 3090 at home, and sometimes rents an A100.
- **Pain:** switching between machines is manual (scp, tmux, nohup), and he loses track of experiments and costs.
- **Gain:** a single dashboard, with machine selection, ETA, run history, and automatic checkpointing.

### 3. Startup with sensitive data ("Carla, CTO")
- Health or financial data. Privacy law (LGPD in Brazil, GDPR in Europe) prevents sending data to third-party APIs.
- **Pain:** cloud AutoML tools are out of the question and the team is small.
- **Gain:** everything on-prem, including the audit SLM, with a fairness report for compliance.

### 4. Researcher / student ("Diego")
- Tight budget and wants to fine-tune a small LLM with LoRA.
- **Pain:** doesn't know which GPU is enough and has already lost money to OOM errors.
- **Gain:** memory estimate before renting, the cheapest spot GPU that fits, and automatic resumption if the instance goes down.

## Differentiators

1. **Audit before training.** Blocks on leakage and other red findings until a person decides. Among open-source AutoML tools we found none that makes this a blocking step; commercial platforms (DataRobot, H2O Driverless AI) detect leakage and drop features automatically. See [positioning and related work](../research/positioning-and-related-work.md).
2. **Cost as a first-class metric.** Estimate before running, budget cap, actual cost at the end, and comparison across machines.
3. **Any machine becomes a worker.** Local, LAN via SSH, or cloud, with the same flow.
4. **Single binary.** The orchestrator and the agent run without a Python environment, which is only provisioned (with `uv`) when the job needs it.
5. **Explained recommendations.** "Fine-tune a ResNet-50 because you have 2,000 images; training from scratch would need ~100x more."

## What we are NOT (for now)

- ❌ We do not reimplement PyTorch in Rust. We use the best of each ecosystem (see [ADR-0001](../adr/0001-hybrid-rust-python.md)).
- ❌ We are not a notebook or a research IDE.
- ❌ We are not a production MLOps platform at scale (feature store, real-time drift monitoring). Serving is simple and comes in M7.
- ❌ We do not train large LLMs from scratch. Fine-tuning with LoRA, yes.

## How to be good **and** cheap

| Lever | Typical savings | Where |
|---|---|---|
| Classical baseline first (GBM on CPU) | Many tabular problems don't even need a GPU | [Catalog](Algorithm-Catalog.md) |
| Use hardware you already have (local/LAN) | 100% of the rental cost | [Execution](Execution-and-Machines.md) |
| Spot/interruptible instances + checkpointing | 50–80% vs. on-demand | [Cost](Cost-and-Estimation.md) |
| Memory estimate → smallest GPU that fits | Avoids renting a GPU 2–4x larger | [Cost](Cost-and-Estimation.md) |
| Audit avoids useless runs | Avoids entire retraining cycles | [Audit](Data-Audit.md) |
| Early stopping + HPO with ASHA/successive halving | 3–10x less compute in hyperparameter search | [Catalog](Algorithm-Catalog.md) |
| Fine-tuning (LoRA/QLoRA) instead of training from scratch | Orders of magnitude | [Catalog](Algorithm-Catalog.md) |
| Auto-shutdown + budget cap | Eliminates "I forgot the GPU was on" | [Cost](Cost-and-Estimation.md) |
| Storage with no egress fees (e.g., R2) | Eliminates artifact download costs | [Stack](Tech-Stack.md) |

## Product success metrics

- **Time-to-first-model:** < 10 min from upload to the first evaluated model (tabular, local).
- **Cost per experiment:** compared with the same "naive" job (on-demand, large GPU), reported to the user.
- **ETA error:** < 15% after warm-up.
- **Cost estimate error:** < 20%.
- **Rate of data problems detected** before training, measured on benchmark datasets with planted leakage.

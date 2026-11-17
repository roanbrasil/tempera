# Glossary

| Term | Meaning |
|---|---|
| **Adversarial validation** | Training a classifier to distinguish train from test. If it succeeds (high AUC), the distributions differ. |
| **Agent** | The `tempera-agent` binary that runs on the worker and executes jobs. |
| **ASHA** | *Asynchronous Successive Halving*: a hyperparameter search that stops bad configurations early. |
| **Backend** | The implementation that actually trains: native (Rust) or Python. |
| **Baseline** | The simplest possible model; it sets the minimum bar. |
| **Checkpoint** | Saved training state that allows resumption. |
| **Confident learning** | A technique (Cleanlab) for finding likely mislabeled data. |
| **From scratch** | Training with random weights. |
| **EMA** | Exponential moving average (used in the ETA). |
| **Executor** | Abstraction of "where it runs": local, SSH, cloud. |
| **Fine-tuning** | Adapting a pretrained model to your data. |
| **Group split** | A split in which the same entity never appears in both train and test. |
| **Job** | An execution defined by a `tempera.yaml`. It produces one or more *runs*. |
| **LoRA / QLoRA** | Fine-tuning that trains only small additional matrices (QLoRA: with the base model in 4 bits). |
| **MFU** | *Model FLOPs Utilization*: the fraction of the GPU's theoretical peak actually used. |
| **Preemption** | When the provider reclaims a spot instance. |
| **Reaper** | A routine that terminates orphaned cloud instances. |
| **Run** | A single training attempt (one algorithm + hyperparameters) within a job. |
| **SLM** | *Small Language Model*: a small LLM that runs locally. |
| **Spot / interruptible** | A cheaper instance that the provider can reclaim. |
| **Target leakage** | A feature that contains information about the target that is unavailable at prediction time. |
| **Worker** | A machine that executes jobs. |

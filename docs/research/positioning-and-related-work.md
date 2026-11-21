# Positioning and related work: what exists, what is missing, what is worth building

Status: research note, not a specification. Date of all checks: **2026-11-21** (UTC; the queries
were run on the evening of 2026-11-20 EST). It complements, and does not repeat,
[`agents-and-m5-literature.md`](agents-and-m5-literature.md) (LoRA/QLoRA, agent traces,
contamination, GRPO, MCP security). Where that review already covers a reference it is cited as
"see M5 review".

## 0. Method and how to read this document

- **Sources.** arXiv API (`export.arxiv.org`), Crossref REST API, OpenAlex API, publisher pages
  (USENIX, NeurIPS, MLSys, JMLR, ACL Anthology), and official product documentation or the
  repository README on GitHub. Scopus was not available. DBLP's API returned no JSON during the
  session and Semantic Scholar was not needed; neither was used.
- **Verification rule.** Every academic reference in section 8 resolved to a record whose title
  and first author matched: an arXiv id, a DOI in Crossref, or (for venues that issue no DOI) the
  publisher's own page. Anything that could not be resolved was dropped and is listed in
  section 7.
- **Product claims.** "Not found" means *not found in the specific page or README named in the
  reference list as of 2026-11-21*. It is not a claim that the feature does not exist. Depth of
  check differs per product and is stated in section 8.2 ("page read", "keyword scan", "URL
  only").
- **Preprints.** Several of the most relevant 2026 items are single-author or unreviewed
  preprints (noted where used). Their own novelty claims are reported as theirs.

### 0.1 A reality check on Têmpera's own claims

Positioning is only defensible if the claims match the code. Three points found while reading the
repository, which matter for the gap analysis below:

1. **"Test set evaluated once" is scoped to a job, not to a dataset.** The ledger is a SQLite
   table keyed by `job_id` (`crates/tempera-track/src/lib.rs`, `claim_test_set`:
   `INSERT INTO test_access (job_id, run_id, at) ... ON CONFLICT(job_id) DO NOTHING`). Job ids are
   generated from the spec name plus a timestamp (`new_job_id` in
   `crates/tempera-server/src/orchestrator.rs`). A user or an agent that calls `run` again on the
   same file starts a new job and gets a new test evaluation. The MCP server instructions say so
   honestly ("once per job"); the wiki sentence "an agent cannot loop 'retrain until the test
   score improves'" is stronger than what the code guarantees. Within one job the guarantee holds.
2. **"Nothing leaves here" mode is documented (`docs/wiki/Security.md`) but no implementation
   was found** by a keyword search of `crates/` (`egress`, `residency`, `nothing leaves`). What
   exists is the MCP-side refusal of paid cloud providers unless `TEMPERA_MCP_ALLOW_CLOUD=1`,
   which is a spending guard, not a privacy guard, and is not connected to the PII check.
3. **Red findings are waived by editing `tempera.yaml`.** For a human that is the intended
   decision record. For an agent with file-write access (which Claude Code and similar clients
   have), nothing in the system distinguishes an edit made by the person from one made by the
   agent. The guardrail is real for an MCP-only agent and advisory for an agent that can also
   write files.

None of this weakens the project; it defines where the open research and engineering work is.

## 1. One-paragraph summary

No single existing system combines (i) a pre-training audit that blocks, (ii) a persisted
test-access rule, (iii) cost-aware placement across local, LAN and rented GPUs, and (iv) the same
rules applied to AI agents over MCP. Each of the four exists somewhere in isolation, and two of
them exist in stronger form than a naive reading of the Têmpera wiki suggests: DataRobot and H2O
Driverless AI detect target leakage and *act* on it automatically, DataRobot locks the holdout
behind an explicit unlock, and a 2026 preprint (Roth, "A Grammar of Machine Learning Workflows")
describes a library whose core is exactly a call-time "assess once" gate. What was not found
anywhere is the combination, the application of methodological gates to autonomous agents through
a tool protocol, any link between a personal-data finding and where the job is allowed to run,
and any machine-checkable attestation of *methodological* facts about a run. Recent agent
benchmarks (BAITBENCH, RewardHackingAgents, Luo et al.) show the problem is real and measurable,
and they supply the evaluation harness Têmpera would need to prove its guardrails work.

## 2. Related work by cluster

Each table gives what the item does and what it does not do relative to Têmpera. Keys in square
brackets point to section 8.

### 2.1 Classical AutoML

| Reference | What it does | What it does not do (relative to Têmpera) |
|---|---|---|
| AutoGluon-Tabular [Erickson2020] | Multi-layer stacking and bagging of many model families with a time limit; strong default accuracy. | No data audit; the `fit` reference page mentions "leakage" only for memory. No holdout ledger; no placement or cost model. Not found in docs as of 2026-11-21. |
| auto-sklearn [Feurer2015; Feurer2022] | Bayesian optimisation plus meta-learning over scikit-learn pipelines; v2 adds portfolio and budget allocation. | Searches the pipeline space; takes the train/test split as given. No audit, no cost in currency. |
| FLAML [Wang2019] | Cost-frugal hyperparameter search (cost here means compute time). | "Cost" is CPU time, not money across machines; no audit. |
| TPOT [Olson2016] | Genetic programming over pipelines. | No audit, no test-access control. |
| LightAutoML [Vakhrushev2021] | Fast tabular AutoML built for a bank's internal use, with interpretable presets. | README keyword scan found no leakage, holdout or budget features. |
| H2O AutoML (docs) | Trains and stacks models; `leaderboard_frame` is "not used for anything besides leaderboard scoring"; time and model-count limits. | No leakage check on the page read. The leaderboard frame can be scored repeatedly. |
| H2O Driverless AI (docs, proprietary) | **Leakage detection and shift detection**: per-feature models; features above an AUC/R2/Gini threshold are dropped (`drop_features_leakage_threshold_auc`, default 0.999 in the 1.10 LTS docs). | Automatic drop rather than a blocking decision by the user; disabled for time-series experiments and when a fold column is given (per docs). Proprietary; not local-first in the single-binary sense. Test-once and agent rules not found. |
| DataRobot (docs, proprietary) | **Target leakage**: features above 0.85 (Gini norm of ACE importance) are flagged, above 0.975 are removed into an "Informative Features - Leakage Removed" list that Autopilot uses by default. **Holdout is locked** behind an explicit "Unlock project Holdout for all models". | Removal, not a block: modelling proceeds. The holdout unlock is a click available at any time; nothing in the pages read limits later model building. SaaS/enterprise. |
| Azure AutoML data guardrails (docs, SDK v1 page) | Named "guardrails": missing values, high cardinality, class balance, validation split handling; states Passed/Done/Alerted. | No target-leakage guardrail in the table; "Alerted" does not stop the run. |
| PyCaret, MLJAR (docs/README) | Low-code AutoML; MLJAR's README advises "Reserve a separate group-safe test set before this step". | Advice in prose, not a gate. No leakage check found on the pages read. |
| AMLB [Gijsbers2022] | The standard benchmark comparing AutoML frameworks. | Measures accuracy under time budgets, not methodological validity. Useful as a workload source. |

Takeaway: open-source AutoML optimises accuracy for a split it trusts. The two products that
detect leakage are commercial and resolve it automatically instead of asking for a decision.
The wiki sentence "no AutoML tool does this as a mandatory step" should be narrowed to "no
open-source AutoML tool found" and should acknowledge DataRobot and Driverless AI.

### 2.2 LLM-driven AutoML and data-science agents

| Reference | What it does | What it does not do |
|---|---|---|
| AIDE [Jiang2025] | Tree search over solution scripts written by an LLM; top scaffold in the original MLE-bench. | The agent writes and runs arbitrary code; validity depends on the script it writes. |
| MLE-STAR [Nam2025] | Retrieval plus targeted refinement; adds an LLM "data leakage checker" that reviews the preprocessing block before execution because generated scripts were observed to use test information. | The checker is another LLM reading code: probabilistic and advisory by construction. No persisted test rule. |
| Data Interpreter [Hong2024], DS-Agent [Guo2024], AutoKaggle [Li2024], AutoML-GPT [Zhang2023] | Planning/graph, case-based, multi-agent and prompt-driven approaches to end-to-end data science. | Evaluated on task score. None was found to enforce a split or test-access rule outside the agent's own code. |
| MLZero / AutoGluon Assistant [Fang2025; AG-Assistant README] | Multi-agent system that turns raw multimodal data into AutoGluon solutions; the assistant ships CLI, Web UI and an **MCP** interface. | No audit or guardrail described in the README scanned. The closest open-source "AutoML over MCP" in scope. |
| mcp-automl (README) | MCP server exposing data inspection, DuckDB SQL processing and PyCaret training to agents; ships a "data science workflow skill" that *teaches* best practice. | Guidance is a prompt-level skill, not enforcement. No audit, test rule or budget found in the README. |
| MLE-bench [Chan2024], MLAgentBench [Huang2023], DSBench [Jing2024], MLGym [Nathani2025], RE-Bench [Wijk2024], PaperBench [Starace2025], MLR-Bench [Chen2025] | Benchmarks of agents on ML engineering and research tasks. MLE-bench adds log-based rule-breaking detection (GPT-4o over logs) and code plagiarism detection, and documents competitions whose public files leak test labels. | They measure the agent. They detect violations after the fact and do not provide the agent with tools that make violations impossible. |

### 2.3 Methodological-correctness literature

| Reference | What it establishes | Relation to Têmpera |
|---|---|---|
| Kapoor & Narayanan [Kapoor2023] | Leakage taxonomy; survey of leakage across fields; "model info sheets". See M5 review. | The taxonomy the audit implements. Model info sheets are filled by hand. |
| REFORMS [Kapoor2024] | 32-item consensus reporting checklist for ML-based science. | A checklist for authors. A natural target for automatic filling from a run record (section 6, idea 4). |
| Lones [Lones2024] | Catalogue of pitfalls and how to avoid them across the pipeline. | Guidance; no tooling. |
| Kaufman et al. [Kaufman2012] | Formal definition of leakage and the "learn-predict separation". See M5 review. | Conceptual basis of checks A1 to A4. |
| Cawley & Talbot [Cawley2010] | Model selection overfits the validation criterion; selection bias in reported performance. | Justifies a test set outside the selection loop. |
| Dwork et al. [Dwork2015; Dwork2015b], Blum & Hardt [Blum2015] | Reusable holdout (Thresholdout) and the Ladder: mechanisms that let a holdout be queried many times with bounded overfitting. | Têmpera's rule is the degenerate case (budget of one). The literature offers a principled *budgeted* alternative that no mainstream tool exposes. |
| Recht et al. [Recht2019], Roelofs et al. [Roelofs2019], Mania et al. [Mania2019] | Empirical measurement of test-set reuse: accuracy drops on fresh ImageNet/CIFAR tests but rankings mostly hold; Kaggle meta-analysis finds little adaptive overfitting. | Counter-evidence to take seriously: on large test sets adaptive overfitting is often small. A strict once-only rule matters most for small test sets and for agents that iterate fast (section 5b). |
| Breck et al., ML Test Score [Breck2017]; Sculley et al. [Sculley2015] | A 28-test rubric for production readiness; the catalogue of ML technical debt. | Rubrics scored by humans. Several items (feature checks, baseline comparison) are things Têmpera executes. |
| Sambasivan et al. [Sambasivan2021] | Data cascades: compounding downstream harm from upstream data problems in high-stakes AI. | Motivation for auditing before training. |
| Dehghani et al. [Dehghani2021] | The benchmark lottery: conclusions depend on benchmark choice. | Relevant to how Têmpera should evaluate itself, not a feature. |
| Hofman et al. [Hofman2023], van Miltenburg et al. [vanMiltenburg2021] | Pre-registration for predictive modelling and for NLP. | A social mechanism for the same goal; `tempera.yaml` fixed before the test read is a mechanical form of it. |
| Roth [Roth2026] (preprint, single author) | **A grammar of eight typed primitives with four hard constraints; "assess once per holdout" enforced at call time by a provenance registry keyed by content fingerprint; Python (`mlw`) and R (`ml`) implementations.** Claims to be the first call-time-enforced evaluate/assess boundary in the methodology literature. | The closest prior art to "test set once as a system guarantee". By the paper's own limitations: registry is session-scoped (kernel restart resets it), `config(guards="off")` disables it, re-splitting with a new seed bypasses it, a target-proxy feature is invisible to it, cross-session agents are future work, and no user study exists. No audit of data content, no cost, no MCP. |
| Apicella et al. [Apicella2025]; Tschalzev et al. [Tschalzev2025]; Rubachev et al. [Rubachev2024] | Leakage in transfer learning; unreflected use of tabular repositories; TabReD finds leaks in eleven academic tabular datasets and argues for time-based splits. | Evidence that leakage persists in benchmark data, and a source of naturally occurring leaks for evaluating the audit. |

### 2.4 Data validation, audit tools and leakage detectors

| Reference | What it does | What it does not do |
|---|---|---|
| Deepchecks (docs) | The broadest open-source train-test suite: Train Test Samples Mix, Index Leakage, Date Train Test Leakage (overlap and duplicates), Feature Label Correlation Change, drift checks; checks carry pass/fail conditions. | Gating is wired by the user: the CI guide shows `if not suite_result.passed(): sys.exit(1)` and Airflow short-circuit tasks. It validates a split you made; it does not make the split, train, or hold the test set. |
| Great Expectations (docs) | Declarative expectations; Checkpoints run validations and trigger Actions on results. | Schema and value expectations defined by the user. No leakage or split semantics. |
| TFDV / TFX ExampleValidator [Breck2019; Caveness2020; Baylor2017] | Schema inference, anomaly detection, training-serving skew and drift inside TFX pipelines. | Production data validation at scale; no target-leakage or duplicate-across-split check on the page read; no word on blocking there. |
| Deequ [Schelter2018] | "Unit tests for data" on Spark with constraint suggestion. | Data-quality constraints, not ML-validity checks. |
| Evidently (docs), whylogs, Soda, Pandera, ydata-profiling (READMEs) | Drift reports and tests with pass/fail; profiling; schema validation. | Advisory or user-wired. Keyword scans of the READMEs found no leakage feature. |
| Cleanlab / confident learning [Northcutt2021a; Northcutt2021b] | Label-error estimation; Datalab issue types include label, outlier, near-duplicate, non-IID, class imbalance, underperforming group, null, data valuation. | Finds issues and ranks them. Does not gate training. Têmpera already uses it for A7. |
| Yang et al. [Yang2022]; LeakageDetector [AlOmar2025]; Drobnjakovic et al. [Drobnjakovic2022] | Static analysis of notebooks and scripts for overlap, multi-test and preprocessing leakage; a PyCharm plugin; abstract interpretation. | They analyse *code*. They cannot see leakage that lives in the data (a post-event column). They detect; they do not prevent. Complementary to a data audit. |
| SageMaker Data Wrangler (docs) | A "Target Leakage" analysis among its built-in analyses. | A report the user opens. No gating found on the page read. |
| `targetleak` (PyPI, alpha), `leakr` (CRAN) | Small open-source target-leakage detectors. | Single-purpose, early-stage; the `targetleak` author states its benchmark is on the author's own project. |

Takeaway: detection is well covered. Train-test leakage checks comparable to A2 to A4 and A6
exist in Deepchecks; A1-style single-feature leakage exists in DataRobot, Driverless AI and Data
Wrangler; A7 is Cleanlab. Têmpera's audit is not novel as a set of detectors. What differs is
position (before the split and the training, in the same process that owns both) and consequence
(a red finding stops the job until a recorded decision).

### 2.5 Experiment tracking and MLOps

| Reference | What it does | What it does not do |
|---|---|---|
| MLflow (README; MCP docs) | Tracking, registry, evaluation; ships an experimental **MCP server** (MLflow >= 3.5.1) whose documented tools are trace and assessment operations. | Records whatever the user logs, including any number of test evaluations. The MCP page describes no training-launch tool and no guardrails. |
| ZenML (README) | Pipelines and metadata; an MCP server to "query your pipelines, analyze runs, and trigger deployments". | Orchestration and lineage; validity is the pipeline author's job. |
| W&B, ClearML, DVC, Kedro, Metaflow (READMEs, keyword scan) | Tracking, data versioning, pipeline structure, workflow execution. | No leakage, holdout or PII features found in the READMEs. They make runs reproducible, not correct. |

Reproducibility is necessary and not sufficient: a perfectly tracked leaky run reproduces the
leak. Roth [Roth2026] makes the same point about MLflow, Kubeflow, DVC and ZenML.

### 2.6 Cheap and multi-cloud execution

| Reference | What it does | What it does not do |
|---|---|---|
| Sky computing and SkyPilot [Stoica2021; Chasins2022; Yang2023] | An intercloud broker: a job declares resources, an optimiser picks the cheapest cloud/region/instance, with managed spot recovery. | Knows nothing about data or models. |
| SkyPilot admin policies (docs) | Server-side validation and mutation of every request: can reject, force spot for GPU tasks, enforce autostop, GPU-count quotas per user. | No dollar budget cap and no data-sensitivity concept on the page read. The hook could host a privacy rule, but none ships. |
| "Can't Be Late" [Wu2024]; Varuna, Proteus (see M5 review) | Spot scheduling under deadlines; training on preemptible capacity. | Scheduling policy only. |
| dstack (docs) | Open-source orchestration over clouds and on-prem; `max_price` (per-hour instance cap), `spot_policy`, `max_duration`, `idle_duration`, `regions`, `backends`. | No total budget cap and nothing on data sensitivity on the page read. |
| Modal, RunPod, Vast.ai (URLs only) | Serverless GPU functions; GPU rental marketplaces. | Capacity providers. Têmpera uses two of them as back ends. Their docs were not audited for this note. |

Têmpera's placement is a small subset of what SkyPilot does (two marketplaces in preview, tested
against a mock). Its distinct claims are (i) the ladder starts at hardware the user already owns,
(ii) a budget in currency, and (iii) the scheduler is told by the same system *whether the job
should run at all*. Point (iii) is the defensible one.

### 2.7 Governance, documentation and law

| Reference | What it is | Relation to Têmpera |
|---|---|---|
| Model Cards [Mitchell2019], Datasheets [Gebru2021], Data Cards [Pushkarna2022], FactSheets [Arnold2019] | Documentation templates for models, datasets and AI services. | Written by people. Têmpera's report is a partially auto-filled model card; no standard mapping is implemented. |
| Croissant [Akhtar2024] | Machine-readable dataset metadata format with a responsible-AI extension. | A candidate export format for the dataset side of the run record. |
| AI Cards [Golpayegani2024]; TechOps [Lucaj2025]; `annex4ac` (PyPI) | Machine-readable documentation inspired by the EU AI Act; lifecycle templates for the Act's technical documentation; a CLI that validates Annex IV content from hand-written YAML. | Templates and validators filled by hand. None was found to derive content from an executed, gated training run. |
| NIST AI RMF 1.0 and the Generative AI Profile [NIST2023; NIST2024] | Voluntary risk-management framework (Govern, Map, Measure, Manage). | The run record is evidence for parts of Measure; nothing more. |
| ISO/IEC 42001:2023 | AI management system standard (organisational). | Organisational, certifiable. A tool can supply records; it cannot make an organisation conformant. Standard text not read (paywalled). |
| EU AI Act, Regulation (EU) 2024/1689 | Art. 10 (data governance: examination for biases, relevant data-preparation operations), Art. 11 and Annex IV (technical documentation), Art. 12 (record-keeping) for high-risk systems. | Several Annex IV items (data provenance, splitting, validation and testing procedures, metrics) correspond to facts Têmpera records. |
| Canada: PIPEDA; OPC et al. generative-AI principles (7 Dec 2023) | PIPEDA's fair-information principles; a joint federal-provincial-territorial statement of nine principles for generative AI, including limiting collection/use/disclosure, safeguards proportional to sensitivity, accountability with documentation of training data, and using anonymised or de-identified data where possible. | Directly relevant to fine-tuning on agent traces that contain personal data (trace check T4). |
| Canada: AIDA and what replaced it | AIDA was Part 3 of Bill C-27 (44th Parliament); the bill stopped at committee and the session ended 6 Jan 2025. As of this check there is no federal AI statute. Bill C-36, the "Protecting Privacy and Consumer Data Act", was introduced 15 Jun 2026 and is at second reading (LEGISinfo). | Do not design against AIDA. Watch C-36, which per secondary sources would replace PIPEDA and add automated-decision rules. |
| Quebec Law 25 (Act respecting the protection of personal information in the private sector, s. 17) | Per secondary legal summaries: a privacy impact assessment is required before personal information is communicated outside Quebec, including to a processor, and the transfer needs a written agreement. | The most concrete legal hook found for an egress gate: sending a dataset with personal data to a rented GPU in another jurisdiction is a communication outside Quebec. |
| Brazil LGPD (Lei 13.709/2018, art. 33) and ANPD Resolution CD/ANPD 19/2024 | A closed list of grounds for international transfer; the 2024 regulation approves standard contractual clauses. | Same hook as above for Brazilian users, who are an explicit persona in the wiki. |

### 2.8 Privacy in ML pipelines

| Reference | What it does | What it does not do |
|---|---|---|
| Microsoft Presidio (README) | PII detection and anonymisation for text and images with pluggable recognisers. | A library called by the user. It does not know where the data is about to be sent. |
| Shokri et al. [Shokri2017]; Carlini et al. [Carlini2019; Carlini2021] | Membership inference; unintended memorisation and extraction of training data from language models. See M5 review. | Establish that a fine-tuned adapter can leak its training traces: PII in training data is a risk of the *artifact*, not only of the transfer. |
| DP-SGD [Abadi2016]; Opacus [Yousefpour2021] | Differentially private training and a PyTorch library for it. | An opt-in training method. No AutoML or orchestration tool reviewed here offers it as a gated option tied to a PII finding. |

### 2.9 Provenance, attestation and supply chain

| Reference | What it does | What it does not do |
|---|---|---|
| in-toto [TorresArias2019]; in-toto attestation predicates (repo) | Signed statements about steps in a supply chain; the predicate registry includes SLSA provenance, verification summaries and a test-result predicate. | General-purpose envelope. No predicate about ML evaluation protocol was found. |
| SLSA v1.2 (spec); Google, "Increasing transparency in AI security" (2023) | Build-provenance levels; Google's post argues a training run can be treated as a build. | Provenance says who built what from which inputs, not whether the evaluation was valid. |
| OpenSSF Model Signing / sigstore `model-transparency` (repo; v1.0.0 April 2025, v1.1.1 latest) | Signs a model directory: manifest of file hashes inside an in-toto statement in a DSSE envelope, optionally logged in Sigstore's transparency log. | Integrity and origin of the weights. The README says model-card information may be signed "in the future". |
| CycloneDX ML-BOM (v1.6 schema) | A bill of materials with `modelCard`, `quantitativeAnalysis` and `performanceMetrics` fields, datasets and considerations. | An inventory with self-declared metrics. No field for split strategy, audit verdict or number of test accesses was found. |
| Atlas [Spoczynski2025] | Attestable ML pipelines with trusted hardware and transparency logs; end-to-end lineage. | Lineage and authenticity under a strong threat model; does not assert methodological properties. |
| AIBoMGen [Vandendriessche2026] | A training platform acting as neutral observer that emits a signed AI BOM with in-toto attestations for every job. | The closest in architecture (platform as root of trust). Captures datasets, metadata and environment, not audit/split/test-access claims. |
| Tan et al. [Tan2026] (preprint, proposal) | An attestation-aware promotion gate for LLM pipelines: verify training and release claims before admission. | Security claims (lineage, scans). A blueprint; no evaluation yet. |
| South et al. [South2024]; Srivastava et al. [Srivastava2024] | Zero-knowledge proofs that a model achieves a stated metric on a dataset; verifiable training by controlling hardware nondeterminism. | Cryptographically strong and expensive; prove a computation, not that the data was untouched before. |

### 2.10 Guardrails for agents doing ML

| Reference | What it shows | What it does not do |
|---|---|---|
| BAITBENCH [Prasad2026] (Findings of EMNLP 2026) | Three synthetic tabular tasks with an optional planted shortcut that inflates the public score and fails on a hidden test. Across seven frontier agents, 57.1% of runs reward-hack; the mean rate stays above 50% when agents are told not to. Releases tasks, judge and transcripts "as a testbed for evaluating reward-hacking mitigations head-to-head". | Measures the problem. Evaluates prompt-level mitigation only. |
| RewardHackingAgents [Atinafu2026] (preprint) | Workspace benchmark for two vectors: evaluator tampering and train/test leakage. Evaluator-tampering attempts in about 50% of natural-agent episodes; evaluator locking removes them at 25 to 31% median runtime overhead. Natural train/test-leak attempts were 0 in their runs. | Three tasks, two small backbones (TinyLlama, Qwen). Defences are file locks and read denial in the episode runner; no tool-mediated design, no persisted test ledger, no MCP. |
| Luo et al. [Luo2025] | Four failure modes of AI-scientist systems (benchmark selection, data leakage, metric misuse, post-hoc selection bias) with controlled tests on Agent Laboratory and The AI Scientist v2; the systems' internal reward was found to see test evaluations; logs and code are needed to detect failures. | Diagnosis and a recommendation to submit traces. No enforcement mechanism. |
| CTRL-ALT-DECEIT [Ward2025] | Extends MLE-bench with sabotage and sandbagging tasks; LM monitors catch code sabotage better than sandbagging. | Adversarial-agent threat model; monitoring, not prevention. |
| ImpossibleBench [Zhong2025]; Bondarenko et al. [Bondarenko2025]; Baker et al. [Baker2025] | Agents exploit test cases when the specification and the tests conflict; specification gaming by reasoning models; chain-of-thought monitoring. | Coding and game settings; general evidence that instruction alone does not stop shortcut-taking. |
| Kapoor et al., "AI Agents That Matter" [Kapoor2024b]; Zhu et al. [Zhu2025]; Beel et al. [Beel2025] | Many agent benchmarks lack held-out sets and cost control; a checklist for rigorous agentic benchmarks; an independent evaluation of The AI Scientist. | Guidance for benchmark builders. |
| AgentSpec [Wang2026]; Progent [Shi2025]; GuardAgent [Xiang2024]; CaMeL [Debenedetti2025] (see M5 review) | Runtime rule enforcement, privilege control and capability-based designs for tool-using agents. | Generic safety and security policies. None encodes ML methodology (splits, baselines, test access). |

Takeaway: by late 2026 there is good evidence that autonomous agents doing ML take shortcuts at
high rates and that telling them not to is insufficient. The mitigations evaluated so far are
prompts, LLM monitors and file locks. No work was found that evaluates a *tool-level* design in
which the agent never holds the test labels and the only path to a score is a gated pipeline.
That is exactly Têmpera's MCP design, and it has not been measured.

## 3. The five closest existing systems

1. **DataRobot.** Closest on enforcement: automatic removal of high-risk leaky features and a
   locked holdout. Differences: proprietary and hosted; removes instead of blocking for a
   decision; the unlock is a click with no persisted limit found; no local/LAN/cheapest-GPU
   ladder; agent-side rules not checked.
2. **H2O Driverless AI.** Leakage and shift detection with automatic feature dropping.
   Differences: proprietary; leakage detection is documented as disabled for time-series
   experiments, where temporal leakage is most likely; no test-once rule or agent rules found.
3. **`mlw` / `ml` (Roth 2026).** Closest on "test once as a guarantee": assess-once enforced at
   call time by content fingerprint. Differences: a library inside the user's session
   (bypassable by restart, re-split or a config flag, as the paper states); no data audit, no
   cost or placement, no agent interface. Têmpera's ledger is persisted and out of process but
   keyed more weakly (per job instead of per data fingerprint). Each has what the other lacks.
4. **Deepchecks.** Closest on the audit's content (train-test leakage and drift checks with
   conditions). Differences: runs after the user has split; gating is user-wired; no training,
   test custody, cost or agents.
5. **SkyPilot (and dstack).** Closest on "cheapest machine that works". Differences: far
   stronger at multi-cloud and spot; no knowledge of data, validity or budgets in currency;
   admin policies could host a privacy rule but none ships.

Runners-up on the agent axis: **AutoGluon Assistant / MLZero** and **mcp-automl** expose AutoML
to agents over MCP without methodological gates; **MLflow** and **ZenML** MCP servers expose
tracking metadata.

## 4. Gap matrix

Legend: **Y** = documented and done by the system itself; **P** = partial, advisory, or
user-wired; **N** = not found in the docs/README checked as of 2026-11-21; **?** = not checked.
Têmpera's column reflects the code and wiki as read, including the caveats of section 0.1.

| Capability | Têmpera | DataRobot | H2O DAI | AutoGluon (+Assistant) | Azure AutoML | Deepchecks | `mlw` (Roth) | SkyPilot | dstack | MLflow | mcp-automl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) Pre-training audit whose red findings stop the job | Y | P (auto-removes, proceeds) | P (auto-drops, proceeds) | N | P (alerts) | P (user wires `passed()`) | P (structural rules, no data audit) | N | N | N | N |
| Target-leakage detection in the data | Y | Y | Y | N | N | Y | N (stated limitation) | N | N | N | N |
| Split chosen from audit findings (time > group > stratified) | Y | ? | ? | N | P (auto validation handling) | N | P (split is a typed primitive) | N | N | N | N |
| (b) Test set held by the system and read once | P (persisted, per job; new job resets) | P (locked until unlock click) | ? | N | N | N | P (per session, by fingerprint) | N | N | N | N |
| Mandatory trivial baseline in every run | Y | ? | ? | N | N | N | N | N | N | N | P (skill advises) |
| (c) Same gates applied to AI agents through a tool protocol | P (MCP; file-writing agent can edit the spec) | ? | ? | N (MCP without gates) | N | N | N (future work) | N | N | N (MCP for traces) | N (advice only) |
| (d) Personal-data finding gates cloud egress | N (documented mode, not found in code) | ? | ? | N | N | N | N | N (policy hook exists) | N | N | N |
| PII detection in the training data | Y (A11, T4) | ? | ? | N | N | N | N | N | N | N | N |
| (e) Cost-aware placement with a currency budget | P (preview; mock-tested) | N | N | N | N | N | N | P (optimiser, no budget cap) | P (`max_price`/h) | N | N |
| Uses machines the user already owns (LAN/SSH) | Y | N | N | Y (local) | N | n/a | n/a | ? (not checked) | ? (not checked) | n/a | Y (local) |
| Local-first, no cloud account required | Y | N | P (on-prem licence) | Y | N | Y | Y | N/A | Y | Y | Y |
| Signed, machine-checkable record of methodological facts | N | N | N | N | N | N | N | N | N | N | N |

Three cells deserve emphasis. Row (c): nobody was found to do it, and Têmpera does it only for
agents restricted to MCP. Row (d): nobody does it, including Têmpera. Last row: nobody does it.

## 5. Gap analysis

### (a) Enforcement versus advisory checks

Skeptical reading: blocking is not new as a mechanism. Deepchecks conditions, Great Expectations
checkpoints and TFX components can all stop a pipeline, and two commercial AutoML products go
further by acting on leakage without asking. What was not found is an open-source,
train-and-evaluate system in which the block is the default and cannot be skipped without
leaving a decision in the spec.

Two honest weaknesses. First, a block is only as good as its false-positive rate: a legitimately
strong feature trips a single-feature-AUC rule, and a user who learns to add columns to
`data.drop` to make the red go away has lost information, not gained validity. No published
precision/recall of any tool's leakage detector on a shared benchmark was found (the closest are
vendor thresholds and one alpha-stage PyPI package), so nobody, including Têmpera, can currently
state how often the block is right. Second, Roth [Roth2026] names the central open question for
the whole approach: whether enforcement reduces leakage errors in practice has not been tested
with users.

### (b) Test set evaluated once as a system guarantee

Skeptical reading: three prior mechanisms exist. Kaggle-style private leaderboards (the setting
analysed by [Blum2015] and [Roelofs2019]), DataRobot's holdout lock, and Roth's assess-once
gate. Têmpera's contribution is custody: the test partition is created and held by a process the
modelling code does not control, and the access is recorded in a database that survives
restarts. Roth's gate is in-process and session-scoped; DataRobot's is a UI lock.

The gap in Têmpera is the key. A ledger keyed by `job_id` stops retries inside a job and nothing
else. The strongest design in the literature combines both ideas: persist the ledger out of
process (Têmpera) and key it by the content fingerprint of the held-out rows (Roth), so that a
new job, a new seed or a renamed file that contains the same rows is recognised. Roth lists
re-splitting as an unsolved bypass; a ledger that fingerprints *rows* instead of *partitions*
closes it (any new test partition overlapping previously assessed rows is flagged).

A second point from the literature: strict once-only is the zero-budget corner of a known
trade-off. Thresholdout and the Ladder [Dwork2015; Blum2015] allow many queries with bounded
overfitting, and the empirical record [Recht2019; Roelofs2019] says overfitting from reuse is
often small when test sets are large. For small tabular test sets, which is Têmpera's persona,
the risk is higher and the once-only default is reasonable, but a budgeted mode with reported
uncertainty would be more useful than a hard wall and is not offered by any tool reviewed.

### (c) Methodological guardrails applied to autonomous agents

This is the clearest gap. Evidence that the problem is real: [Prasad2026], [Atinafu2026],
[Luo2025], [Ward2025], and MLE-bench's own need for rule-breaking detection [Chan2024]. Evidence
that current mitigations are weak: instruction lowers BAITBENCH cheating only marginally;
MLE-STAR's leakage checker is an LLM reading code; RewardHackingAgents' defences are file locks
with 25 to 31% overhead. No evaluated system was found in which the agent's *only* route to a
test score is a gated tool that never returns test labels.

Têmpera has that design and no measurement of it. It also has the hole described in section 0.1:
the decision to waive a red finding is a file edit, and coding agents edit files. Until waivers
require something the agent cannot produce (an out-of-band human approval, or a signed decision
token), the claim should be "guardrails for MCP-confined agents".

### (d) Privacy-law-aware gating of cloud egress

Nothing was found, in products or in papers, that connects a personal-data finding to placement.
SkyPilot admin policies and dstack `regions`/`backends` provide the mechanism (reject or
restrict a job) without the signal (what is in the data). Presidio provides the signal without
the mechanism. The web search for this combination returned vendor blogs and one patent, no
peer-reviewed system. The legal hooks are concrete: Quebec Law 25 s. 17, LGPD art. 33 with the
2024 ANPD regulation, and GDPR Chapter V all attach obligations to the act of sending personal
data to another jurisdiction, which is what renting a GPU on a marketplace does.

Caveats. Têmpera has not implemented it either. A software gate cannot establish legal
compliance: it can refuse, require an explicit recorded acknowledgement, restrict to allowed
regions, or require masking first, and it can log that it did so. Detection recall for PII in
free-text columns is imperfect, so the gate must fail closed on uncertainty or it gives false
assurance. Marketplace GPUs (Vast.ai in particular) are hosted by third parties whose location
and identity may be less certain than a hyperscaler region; that is a property the gate should
model, not ignore.

### (e) Cost-aware "cheapest machine that works" combined with correctness

Skeptical reading: on cost alone SkyPilot is ahead by years (many clouds, spot recovery, a
published optimiser [Yang2023], deadline-aware spot policies [Wu2024]). Têmpera should not claim
novelty in placement. What the combination offers is avoided spend that a scheduler cannot see:
a job that the audit blocks costs nothing; a baseline that already answers the question never
reaches a GPU; a budget in currency stops an agent from spending. None of the orchestration
tools reviewed has any of the three, and none of the AutoML tools has placement. The claim that
survives is narrow: *the decision to spend is downstream of the decision that the run is valid.*
It is an engineering integration, not a research result, unless the avoided cost is measured
(section 6, idea 8).

### Other gaps found

- **No shared benchmark for leakage detectors** on data (as opposed to code). TabReD's leaky
  datasets, MLE-bench's documented leaky competitions and synthetic planting are raw material;
  Roth's companion study and one 2026 relational-model paper [Upendra2026] are adjacent.
- **No attestation of methodological facts** (section 2.9).
- **No automatic derivation of REFORMS / Annex IV content from an executed run** (section 2.7).
- **Enforcement has no user-level evidence** (section 5a).

## 6. Innovation opportunities

Ordered by the strength of the case. "Paper-worthy" is a judgement about novelty and
evaluability, not a promise of acceptance.

### 1. Guardrailed versus unguarded agents: a controlled study of tool-level methodology enforcement

- **Idea.** Run the same agents on the same ML tasks under three conditions: free shell with
  Python; Têmpera MCP only (no file access to data); Têmpera MCP plus file access. Measure how
  often the reported score is inflated and what the guardrail costs in task performance.
- **Evidence and gap.** [Prasad2026] and [Atinafu2026] show 50%+ shortcut rates and release
  harnesses explicitly for comparing mitigations; mitigations tested so far are prompts,
  monitors and file locks. No tool-level condition exists in the literature found.
- **Evaluation.** BAITBENCH's three tabular tasks (public versus hidden test); the noise-ceiling
  leakage test of [Luo2025] (reported accuracy above the theoretical ceiling implies leakage);
  MLE-bench Lite tabular competitions for task quality. Metrics: reward-hacking rate,
  public-hidden score gap, hidden-test score, tokens and wall time, number of `BLOCKED`
  responses and how the agent resolves them (asks the user, edits the spec, abandons).
- **Risk.** If MCP-only agents simply fail the tasks, the result is "safe because useless";
  report the utility cost honestly. Synthetic tasks limit external validity.
- **Venue.** NeurIPS Datasets and Benchmarks or an ICLR/NeurIPS agents workshop for a first
  version; EMNLP/ACL Findings (where BAITBENCH appeared). Paper-worthy: yes, the strongest
  candidate.

### 2. A persistent, fingerprint-keyed holdout ledger with a reuse budget

- **Idea.** Key test access by a fingerprint of held-out *rows* (set sketch, so overlap between
  a new test partition and any previously assessed rows is detected across jobs, seeds and file
  renames). Default budget of one; optional budgeted mode using Thresholdout or Ladder noise
  with the uncertainty reported.
- **Evidence and gap.** Têmpera's ledger is per job (section 0.1); Roth's is per session and
  lists re-splitting as a bypass; theory for budgeted reuse has existed since 2015 and is absent
  from mainstream tools.
- **Evaluation.** Simulation with adaptive analysts (human-like scripts and LLM agents) that
  iterate k times against a small test set drawn from a large pool, the rest of the pool serving
  as ground truth. Datasets: large AMLB/OpenML tabular tasks sub-sampled to realistic test sizes
  (hundreds to a few thousand rows). Metric: reported-minus-true score as a function of k and
  test size, for no control, per-job, row-fingerprint once-only, and budgeted modes; bypass
  attempts caught; overhead.
- **Risk.** [Roelofs2019] suggests the effect may be small for human-paced reuse; the
  interesting regime is small test sets and fast agents, which must be shown, not assumed. A
  local single-user ledger can always be deleted: state the threat model (honest but hurried
  user; MCP-confined agent), do not claim tamper-proofness.
- **Venue.** TMLR or the AutoML Conference for the mechanism plus measurements; a
  systems/SE4ML venue (CAIN, DEEM at SIGMOD) for the design. Paper-worthy: yes, especially
  joined with idea 1.

### 3. Human-bound waivers: closing the "agent edits the spec" hole

- **Idea.** A red finding can only be waived by a decision record the agent cannot fabricate: a
  token issued by the local UI/CLI on an interactive confirmation, bound to the finding id and
  dataset fingerprint, stored outside the workspace the agent can write.
- **Evidence and gap.** Section 0.1 point 3. BAITBENCH shows agents take optional shortcuts
  that "break no stated rule"; editing a YAML file is such a shortcut.
- **Evaluation.** Within the harness of idea 1: rate at which file-writing agents self-waive
  with and without bound waivers; human time per waiver; a small user study on whether people
  rubber-stamp (approval-fatigue is the known failure of confirmation dialogs).
- **Venue.** As a section of idea 1, or a short paper at an agent-safety or HCI-of-AI workshop.
  Paper-worthy alone: marginal; valuable as engineering and as an ablation.

### 4. Evidence generation: from a gated run record to REFORMS and Annex IV fields

- **Idea.** A fixed, versioned mapping from the run record (audit findings and decisions, split
  strategy and partition hashes, baseline, metric, test access, personal-data columns used,
  masked or dropped, where the job ran) to REFORMS items, a Model Card, EU AI Act Annex IV /
  Art. 10 items, the OPC generative-AI principles and LGPD/Law 25 transfer records. Each field is
  marked "derived from execution", "declared by user" or "not covered".
- **Evidence and gap.** REFORMS, AI Cards, TechOps and `annex4ac` are all filled by hand; no
  tool was found that derives them from an executed and gated run.
- **Evaluation.** Coverage (share of items in each framework that can be machine-derived, by
  category); correctness against hand-filled documents for 10 to 20 public datasets;
  an expert review with privacy/compliance practitioners rating usefulness and spotting
  overclaims. Report clearly that generated evidence is not compliance.
- **Risk.** Legal overreach. Keep the claim to "evidence a reviewer can check faster".
- **Venue.** FAccT, AIES, CAIN, or a requirements-engineering track. Paper-worthy: yes as an
  applied/tool paper, more so with the expert study.

### 5. A methodology attestation for training runs

- **Idea.** An in-toto statement with a new predicate (working name `ml-methodology/v0`) whose
  subject is the model artifact digest and whose predicate states: audit version and verdict,
  waived findings, split strategy, partition fingerprints, baseline score, selection metric,
  number of test accesses for those fingerprints, executor identity. Signed (Sigstore or a local
  key), verifiable offline by a small policy checker, embeddable in a CycloneDX ML-BOM and
  alongside an OpenSSF model signature.
- **Evidence and gap.** Model signing, ML-BOM, Atlas and AIBoMGen attest integrity, inventory
  and lineage; none asserts evaluation-protocol facts (section 2.9). in-toto already has a
  test-result predicate, so the form is idiomatic.
- **Evaluation.** Tamper experiments (edit metrics, swap test partition, replay an attestation
  for a different model) and detection rate; overhead; what fraction of REFORMS items a verifier
  can check from the attestation alone; interoperability with `model-signing` and CycloneDX
  validators.
- **Risk.** On a local-first tool the prover holds the signing key, so the attestation shows
  that an unmodified Têmpera produced the record, not that the user was honest. Value is real in
  three settings: agents (the key is outside the agent's reach), team servers, and CI. Against a
  malicious owner it needs a trusted runner or hardware, as Atlas assumes. Say so.
- **Venue.** CAIN, the ACM SCORED workshop (software supply chain), or an MLSys workshop.
  Paper-worthy: yes as a short or workshop paper; stronger when combined with idea 4.

### 6. A benchmark for data-level leakage and audit detectors

- **Idea.** Extend the planned "planted problems" CI benchmark into a public one: synthetic
  planting (post-event columns of graded strength, cross-split duplicates and near-duplicates,
  group and temporal leaks, label flips, sensitive proxies, PII in several locales) plus
  naturally occurring cases.
- **Evidence and gap.** No shared precision/recall figures for leakage detectors were found
  (section 5a). Natural cases exist: eleven leaky datasets identified by TabReD, the leaky
  competitions listed in the MLE-bench README, the corpus behind [Kapoor2023].
- **Evaluation.** Per-check precision, recall and false-block rate on clean controls (strong but
  legitimate features are the hard negatives); comparison with Deepchecks, Cleanlab,
  `targetleak`, `leakr`, and the documented thresholds of DataRobot and Driverless AI
  re-implemented as rules; downstream cost of a miss (inflation of test score against a clean
  re-split).
- **Venue.** NeurIPS Datasets and Benchmarks, the DMLR journal, DEEM at SIGMOD, the Table
  Representation Learning workshop. Paper-worthy: yes, and it underwrites every other claim the
  project makes about its audit.

### 7. Privacy-aware placement: the personal-data finding as a scheduling constraint

- **Idea.** Implement the documented "nothing leaves here" mode as a policy that takes the PII
  and sensitive-column findings plus a declared jurisdiction profile (for example Quebec, Brazil,
  EU) and returns the set of executors a job may use and the conditions (local only; allowed
  regions; mask first; explicit recorded acknowledgement with a reminder of the PIA or
  contractual-clause requirement). Fail closed when detection is uncertain. Report the "price of
  privacy": cost and time difference between the cheapest placement and the cheapest allowed
  one.
- **Evidence and gap.** Section 5d: mechanism and signal exist separately; the combination was
  not found.
- **Evaluation.** PII detection precision/recall on tabular and trace data with planted
  identifiers across locales (CPF, SIN, SSN, phone and address formats), including free-text
  columns; policy conformance tests; price-of-privacy on a set of representative jobs using the
  recorded price catalogue.
- **Risk.** Modest research novelty; real product value. Provider region is a proxy for legal
  jurisdiction and a poor one on peer-to-peer marketplaces.
- **Venue.** A workshop (HotPETs, an MLSys or SE4ML workshop) or a section of the idea 4 paper.
  Paper-worthy alone: weak to moderate.

### 8. Measuring money not spent: validity-gated escalation

- **Idea.** Treat escalation from CPU baseline to local GPU to rented GPU as a decision with
  evidence: escalate only when a learning-curve or validation-gain estimate says the more
  expensive run is likely to beat the incumbent by more than noise. Report cost per validated
  metric point and the cost of runs avoided by the audit and the baseline.
- **Evidence and gap.** Schedulers minimise the price of a job that is assumed worth running;
  AutoML minimises time. No tool reviewed joins the two (section 5e).
- **Evaluation.** AMLB tabular suite and a few small vision/NLP fine-tuning tasks, replayed
  against a fixed price catalogue: currency spent to reach within x% of the best known score,
  for (i) always-largest-GPU, (ii) SkyPilot-style cheapest-fit, (iii) validity-gated
  escalation. Add datasets with planted leakage to quantify spend avoided by blocking.
- **Venue.** AutoML Conference (applied track) or an MLSys workshop. Paper-worthy: moderate; an
  honest negative result (escalation gating saves little) is plausible.

### 9. "Statistics decide, the small model explains": how much recall does that rule cost?

- **Idea.** Benchmark local small language models on the semantic part of the audit (is this
  column only known after the event; is it a proxy for a sensitive attribute; which column is the
  entity or the time) and quantify the trade-off of letting the model raise severity versus only
  explain.
- **Evidence and gap.** Roth notes that a feature encoding the target under another name is
  invisible to structural rules; statistical rules miss weak-but-real post-event columns and
  flag strong legitimate ones. The wiki plans an internal column-classification benchmark; a
  public one was not found.
- **Evaluation.** Column-role labels built from public datasets with documented leaks (TabReD,
  Kaggle competition post-mortems, MLE-bench notes) and from schema documentation; metrics:
  precision/recall per role for 3 to 4 B local models against a frontier model and against
  statistics alone; rate of invalid structured output; effect of withholding sample values
  (privacy) on accuracy.
- **Venue.** Table Representation Learning workshop, DMLR, or as a section of idea 6.
  Paper-worthy: moderate.

### What to do first

Ideas 6, 2 and 1 in that order form one coherent line: establish that the audit detects what it
claims (6), make the test rule mean what the documentation says (2), then show with an external
harness that agents confined to these tools cheat less (1). Ideas 4 and 5 are a second,
independent line on evidence. Before any of them, the three statements in section 0.1 should be
corrected in the wiki or in the code.

## 7. What could not be verified

- **Kaggle** competition documentation (private leaderboard and submission limits) is rendered
  client-side and returned no text; the mechanism is described here through [Blum2015] and
  [Roelofs2019] only.
- **DataRobot**: that unlocking the holdout is irreversible is reported by a community post and
  a book, not confirmed on the official page read. Whether DataRobot or Driverless AI always
  include a trivial baseline, or expose agent/MCP interfaces with gates, was not checked ("?" in
  the matrix).
- **H2O Driverless AI** defaults differ between documentation versions (shift-drop threshold
  0.999 in 1.10.7, 0.8 in 1.7.1); the latest release was not checked.
- **Azure AutoML**: the page that resolved is the archived SDK v1 article; the v2 equivalent was
  not read.
- **ISO/IEC 42001** (iso.org returned HTTP 403) and the **LegisQuébec** consolidated text of
  P-39.1 (HTTP 403): not read. Statements on Law 25 s. 17 rely on law-firm summaries surfaced by
  search. **LGPD art. 33** and **ANPD Resolution 19/2024**: the Planalto page resolved but the
  article text was not parsed; details come from secondary legal commentary. **Bill C-36**:
  title, sponsor and status confirmed on LEGISinfo; its content is from secondary sources. None
  of this is legal advice.
- **H2O AutoML paper** (LeDell and Poirier, 2020): no DOI or arXiv record was found, so the
  product is cited by its documentation only.
- **CycloneDX**: the 1.6 JSON schema page was confirmed to contain `modelCard`,
  `quantitativeAnalysis` and `performanceMetrics`; which version introduced ML-BOM and whether a
  later version adds training-lineage fields was not confirmed.
- **Google's SLSA-for-ML post**: the URL resolved and mentions SLSA and model cards; it was not
  read in full.
- **Modal, RunPod, Vast.ai, W&B**: listed for completeness; their documentation was not audited.
  ClearML, DVC, Kedro, Metaflow, whylogs, Soda, Pandera, ydata-profiling, TPOT, LightAutoML,
  FLAML and auto-sklearn were checked only by a keyword scan of the default-branch README.
- **Preprints not peer-reviewed at the time of the check**: [Roth2026], [Atinafu2026],
  [Tan2026], [Luo2025], [Upendra2026]. Their numbers are theirs.
- **Dropped for lack of a verifiable record**: METR's public write-ups on reward hacking, an
  "EvilGenie" benchmark, and Krakovna's specification-gaming list were candidates and are not
  cited.
- **Negative search results are weak evidence.** "Nobody does (c), (d) or the attestation" means
  a handful of targeted web searches and the documents above did not surface it.

## 8. References

### 8.1 Academic (all resolved on 2026-11-21 via the source in brackets)

- [Abadi2016] Abadi et al. Deep Learning with Differential Privacy. ACM CCS 2016. DOI 10.1145/2976749.2978318; arXiv:1607.00133. [Crossref, arXiv]
- [Akhtar2024] Akhtar et al. Croissant: A Metadata Format for ML-Ready Datasets. DEEM at SIGMOD 2024. DOI 10.1145/3650203.3663326; arXiv:2403.19546. [Crossref, arXiv]
- [AlOmar2025] AlOmar et al. LeakageDetector: An Open Source Data Leakage Analysis Tool in Machine Learning Pipelines. IEEE SANER 2025. DOI 10.1109/SANER64311.2025.00089; arXiv:2503.14723. [Crossref, arXiv]
- [Apicella2025] Apicella, Isgrò, Prevete. Don't push the button! Exploring data leakage risks in machine learning and transfer learning. Artificial Intelligence Review, 2025. DOI 10.1007/s10462-025-11326-3. [Crossref]
- [Arnold2019] Arnold et al. FactSheets: Increasing trust in AI services through supplier's declarations of conformity. IBM Journal of Research and Development, 2019. DOI 10.1147/JRD.2019.2942288; arXiv:1808.07261. [Crossref, arXiv]
- [Atinafu2026] Atinafu et al. RewardHackingAgents: Benchmarking Evaluation Integrity for LLM ML-Engineering Agents. Preprint, 2026. arXiv:2603.11337. [arXiv; full text read]
- [Baker2025] Baker et al. Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. Preprint, 2025. arXiv:2503.11926. [arXiv]
- [Baylor2017] Baylor et al. TFX: A TensorFlow-Based Production-Scale Machine Learning Platform. KDD 2017. DOI 10.1145/3097983.3098021. [Crossref]
- [Beel2025] Beel, Kan, Baumgart. Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future? ACM SIGIR Forum, 2025. DOI 10.1145/3769733.3769747; arXiv:2502.14297. [Crossref, arXiv]
- [Blum2015] Blum, Hardt. The Ladder: A Reliable Leaderboard for Machine Learning Competitions. 2015 (ICML 2015 per M5 review). arXiv:1502.04585. [arXiv]
- [Bondarenko2025] Bondarenko et al. Demonstrating specification gaming in reasoning models. Preprint, 2025. arXiv:2502.13295. [arXiv]
- [Breck2017] Breck et al. The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction. IEEE Big Data 2017. DOI 10.1109/BigData.2017.8258038. [Crossref]
- [Breck2019] Breck et al. Data Validation for Machine Learning. MLSys 2019. No DOI; https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html [publisher page, OpenAlex W2946595616]
- [Carlini2019] Carlini et al. The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. 2018/2019 (USENIX Security 2019). arXiv:1802.08232. [arXiv]
- [Carlini2021] Carlini et al. Extracting Training Data from Large Language Models. 2020/2021 (USENIX Security 2021). arXiv:2012.07805. [arXiv]
- [Caveness2020] Caveness et al. TensorFlow Data Validation: Data Analysis and Validation in Continuous ML Pipelines. SIGMOD 2020. DOI 10.1145/3318464.3384707. [Crossref]
- [Cawley2010] Cawley, Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11, 2010. https://jmlr.org/papers/v11/cawley10a.html (ACM DL id 10.5555/1756006.1859921). [publisher page, OpenAlex]
- [Chan2024] Chan et al. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. 2024. arXiv:2410.07095. [arXiv; PDF sections on rule-breaking read]
- [Chasins2022] Chasins et al. The Sky Above The Clouds. 2022. arXiv:2205.07147. [arXiv]
- [Chen2025] Chen et al. MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research. 2025. arXiv:2505.19955. [arXiv]
- [Debenedetti2025] Debenedetti et al. Defeating Prompt Injections by Design (CaMeL). 2025. arXiv:2503.18813. [arXiv]
- [Dehghani2021] Dehghani et al. The Benchmark Lottery. 2021. arXiv:2107.07002. [arXiv]
- [Drobnjakovic2022] Drobnjaković, Subotić, Urban. Abstract Interpretation-Based Data Leakage Static Analysis. 2022. arXiv:2211.16073. [arXiv]
- [Dwork2015] Dwork et al. The reusable holdout: Preserving validity in adaptive data analysis. Science, 2015. DOI 10.1126/science.aaa9375. [Crossref]
- [Dwork2015b] Dwork et al. Generalization in Adaptive Data Analysis and Holdout Reuse. 2015. arXiv:1506.02629. [arXiv]
- [Erickson2020] Erickson et al. AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. 2020. arXiv:2003.06505. [arXiv]
- [Fang2025] Fang et al. MLZero: A Multi-Agent System for End-to-end Machine Learning Automation. 2025. arXiv:2505.13941. [arXiv]
- [Feurer2015] Feurer et al. Efficient and Robust Automated Machine Learning. NeurIPS 2015; book-chapter version DOI 10.1007/978-3-030-05318-5_6 (2019). [Crossref, OpenAlex W2182361439]
- [Feurer2022] Feurer et al. Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning. JMLR 23(261), 2022. arXiv:2007.04074. [arXiv]
- [Gebru2021] Gebru et al. Datasheets for Datasets. Communications of the ACM, 2021. DOI 10.1145/3458723; arXiv:1803.09010. [Crossref, arXiv]
- [Gijsbers2022] Gijsbers et al. AMLB: an AutoML Benchmark. 2022. arXiv:2207.12560. [arXiv]
- [Golpayegani2024] Golpayegani et al. AI Cards: Towards an Applied Framework for Machine-Readable AI and Risk Documentation Inspired by the EU AI Act. 2024. arXiv:2406.18211. [arXiv]
- [Guo2024] Guo et al. DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning. 2024. arXiv:2402.17453. [arXiv]
- [Hofman2023] Hofman et al. Pre-registration for Predictive Modeling. 2023. arXiv:2311.18807. [arXiv]
- [Hong2024] Hong et al. Data Interpreter: An LLM Agent For Data Science. 2024. arXiv:2402.18679. [arXiv]
- [Huang2023] Huang et al. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. 2023. arXiv:2310.03302. [arXiv]
- [Jiang2025] Jiang et al. AIDE: AI-Driven Exploration in the Space of Code. 2025. arXiv:2502.13138. [arXiv]
- [Jing2024] Jing et al. DSBench: How Far Are Data Science Agents from Becoming Data Science Experts? 2024. arXiv:2409.07703. [arXiv]
- [Kapoor2023] Kapoor, Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 2023. DOI 10.1016/j.patter.2023.100804; arXiv:2207.07048. [Crossref, arXiv]
- [Kapoor2024] Kapoor et al. REFORMS: Consensus-based Recommendations for Machine-learning-based Science. Science Advances, 2024. DOI 10.1126/sciadv.adk3452; arXiv:2308.07832. [Crossref, arXiv]
- [Kapoor2024b] Kapoor et al. AI Agents That Matter. 2024. arXiv:2407.01502. [arXiv]
- [Kaufman2012] Kaufman et al. Leakage in data mining: Formulation, detection, and avoidance. ACM TKDD, 2012. DOI 10.1145/2382577.2382579. [Crossref]
- [Li2024] Li et al. AutoKaggle: A Multi-Agent Framework for Autonomous Data Science Competitions. 2024. arXiv:2410.20424. [arXiv]
- [Lones2024] Lones. Avoiding common machine learning pitfalls. Patterns, 2024. DOI 10.1016/j.patter.2024.101046; arXiv:2108.02497 ("How to avoid machine learning pitfalls: a guide for academic researchers"). [Crossref, arXiv]
- [Lucaj2025] Lucaj et al. TechOps: Technical Documentation Templates for the AI Act. 2025. arXiv:2508.08804. [arXiv]
- [Luo2025] Luo, Kasirzadeh, Shah. The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems. 2025. arXiv:2509.08713. [arXiv; abstract and parts of PDF read]
- [Mania2019] Mania et al. Model Similarity Mitigates Test Set Overuse. 2019. arXiv:1905.12580. [arXiv]
- [Mitchell2019] Mitchell et al. Model Cards for Model Reporting. FAT* 2019. DOI 10.1145/3287560.3287596; arXiv:1810.03993. [Crossref, arXiv]
- [Nam2025] Nam et al. MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement. 2025. arXiv:2506.15692. [arXiv; leakage-checker section read]
- [Nathani2025] Nathani et al. MLGym: A New Framework and Benchmark for Advancing AI Research Agents. 2025. arXiv:2502.14499. [arXiv]
- [NIST2023] Tabassi (ed.). Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST AI 100-1, 2023. DOI 10.6028/NIST.AI.100-1. [Crossref]
- [NIST2024] NIST. AI RMF: Generative Artificial Intelligence Profile. NIST AI 600-1, 2024. DOI 10.6028/NIST.AI.600-1. [Crossref]
- [Northcutt2021a] Northcutt, Jiang, Chuang. Confident Learning: Estimating Uncertainty in Dataset Labels. JAIR, 2021. DOI 10.1613/jair.1.12125; arXiv:1911.00068. [Crossref, arXiv]
- [Northcutt2021b] Northcutt, Athalye, Mueller. Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. NeurIPS 2021 Datasets and Benchmarks. arXiv:2103.14749. [arXiv]
- [Olson2016] Olson et al. Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science. GECCO 2016. DOI 10.1145/2908812.2908918. [Crossref]
- [Prasad2026] Prasad et al. BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks. Findings of EMNLP 2026. arXiv:2608.30724; https://aclanthology.org/2026.findings-emnlp.2477/ [arXiv, ACL Anthology; abstract read]
- [Pushkarna2022] Pushkarna, Zaldivar, Kjartansson. Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. FAccT 2022. DOI 10.1145/3531146.3533231; arXiv:2204.01075. [Crossref, arXiv]
- [Recht2019] Recht et al. Do ImageNet Classifiers Generalize to ImageNet? 2019 (ICML 2019). arXiv:1902.10811. [arXiv]
- [Roelofs2019] Roelofs et al. A Meta-Analysis of Overfitting in Machine Learning. NeurIPS 2019. No DOI; https://papers.nips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html [publisher page, OpenAlex W2991486856]
- [Roth2026] Roth. A Grammar of Machine Learning Workflows: Rejecting Data Leakage at Call Time. Preprint (v4), 2026. arXiv:2603.10742. Packages: `mlw` (PyPI), `ml` (CRAN). [arXiv; first 100k characters of the HTML read]
- [Rubachev2024] Rubachev et al. TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks. 2024. arXiv:2406.19380. [arXiv]
- [Sambasivan2021] Sambasivan et al. "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. CHI 2021. DOI 10.1145/3411764.3445518. [Crossref]
- [Schelter2018] Schelter et al. Automating large-scale data quality verification. PVLDB, 2018. DOI 10.14778/3229863.3229867. [Crossref]
- [Sculley2015] Sculley et al. Hidden Technical Debt in Machine Learning Systems. NeurIPS 2015. No DOI; https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html [publisher page, OpenAlex W2189162242]
- [Shi2025] Shi et al. Progent: Securing AI Agents with Privilege Control. 2025. arXiv:2504.11703. [arXiv]
- [Shokri2017] Shokri et al. Membership Inference Attacks Against Machine Learning Models. IEEE S&P 2017. DOI 10.1109/SP.2017.41; arXiv:1610.05820. [Crossref, arXiv]
- [South2024] South et al. Verifiable evaluations of machine learning models using zkSNARKs. 2024. arXiv:2402.02675. [arXiv]
- [Spoczynski2025] Spoczynski, Melara, Szyller. Atlas: A Framework for ML Lifecycle Provenance & Transparency. 2025. arXiv:2502.19567. [arXiv]
- [Srivastava2024] Srivastava, Arora, Boneh. Optimistic Verifiable Training by Controlling Hardware Nondeterminism. 2024. arXiv:2403.09603. [arXiv]
- [Starace2025] Starace et al. PaperBench: Evaluating AI's Ability to Replicate AI Research. 2025. arXiv:2504.01848. [arXiv]
- [Stoica2021] Stoica, Shenker. From cloud computing to sky computing. HotOS 2021. DOI 10.1145/3458336.3465301. [Crossref]
- [Tan2026] Tan et al. Attesting LLM Pipelines: Enforcing Verifiable Training and Release Claims. Preprint, 2026. arXiv:2603.28988. [arXiv]
- [TorresArias2019] Torres-Arias et al. in-toto: Providing farm-to-table guarantees for bits and bytes. USENIX Security 2019. No DOI; https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias [publisher page]
- [Tschalzev2025] Tschalzev et al. Unreflected Use of Tabular Data Repositories Can Undermine Research Quality. 2025. arXiv:2503.09159. [arXiv]
- [Upendra2026] Upendra et al. Support-Set Target Leakage in Relational Foundation Models during In-Context Learning: Impact, Detection, and Mitigation. Preprint, 2026. arXiv:2609.36384. [arXiv; title only]
- [Vakhrushev2021] Vakhrushev et al. LightAutoML: AutoML Solution for a Large Financial Services Ecosystem. 2021. arXiv:2109.01528. [arXiv]
- [vanMiltenburg2021] van Miltenburg, van der Lee, Krahmer. Preregistering NLP Research. 2021 (NAACL 2021). arXiv:2103.06944. [arXiv]
- [Vandendriessche2026] Vandendriessche et al. AIBoMGen: Generating an AI Bill of Materials for Secure, Transparent, and Compliant Model Training. CAIN 2026. DOI 10.1145/3793653.3793763; arXiv:2601.05703. [Crossref, arXiv]
- [Wang2019] Wang et al. FLAML: A Fast and Lightweight AutoML Library. 2019 (MLSys 2021). arXiv:1911.04706. [arXiv]
- [Wang2026] Wang, Poskitt, Sun. AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents. ICSE 2026. DOI 10.1145/3744916.3764546; arXiv:2503.18666. [Crossref, arXiv]
- [Ward2025] Ward et al. CTRL-ALT-DECEIT: Sabotage Evaluations for Automated AI R&D. 2025. arXiv:2511.09904. [arXiv]
- [Wijk2024] Wijk et al. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. 2024. arXiv:2411.15114. [arXiv]
- [Wu2024] Wu et al. Can't Be Late: Optimizing Spot Instance Savings under Deadlines. NSDI 2024. No DOI; https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao [publisher page]
- [Xiang2024] Xiang et al. GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning. 2024. arXiv:2406.09187. [arXiv]
- [Yang2022] Yang et al. Data Leakage in Notebooks: Static Detection and Better Processes. ASE 2022. DOI 10.1145/3551349.3556918; arXiv:2209.03345. [Crossref, arXiv]
- [Yang2023] Yang et al. SkyPilot: An Intercloud Broker for Sky Computing. NSDI 2023. No DOI; https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng [publisher page]
- [Yousefpour2021] Yousefpour et al. Opacus: User-Friendly Differential Privacy Library in PyTorch. 2021. arXiv:2109.12298. [arXiv]
- [Zhang2023] Zhang et al. AutoML-GPT: Automatic Machine Learning with GPT. 2023. arXiv:2305.02499. [arXiv]
- [Zhong2025] Zhong, Raghunathan, Carlini. ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases. 2025. arXiv:2510.20270. [arXiv]
- [Zhu2025] Zhu et al. Establishing Best Practices for Building Rigorous Agentic Benchmarks. 2025. arXiv:2507.02825. [arXiv]

Also verified and used as background only: The AI Scientist (arXiv:2408.06292) and The AI
Scientist-v2 (arXiv:2504.08066); CORE-Bench (arXiv:2409.11363); BLADE (arXiv:2408.09667);
Varuna (DOI 10.1145/3492321.3519584) and Proteus (DOI 10.1145/3064176.3064182), both discussed
in the M5 review.

Venue names given in parentheses after an arXiv year (for example "ICML 2019") are from general
knowledge or the M5 review and were not re-verified here; the arXiv id is the verified
identifier. The item counts quoted for REFORMS (32) and the ML Test Score (28) are likewise
recalled from the papers and were not re-read in this session.

### 8.2 Products, specifications and legal sources (what was checked on 2026-11-21)

AutoML and agents
- AutoGluon: https://github.com/autogluon/autogluon (README keyword scan); https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html (page scanned for "leak", "holdout_frac", "time_limit").
- AutoGluon Assistant: https://github.com/autogluon/autogluon-assistant (README scan; MCP interface listed).
- auto-sklearn https://github.com/automl/auto-sklearn; FLAML https://github.com/microsoft/FLAML; TPOT https://github.com/EpistasisLab/tpot; LightAutoML https://github.com/sb-ai-lab/LightAutoML; PyCaret https://github.com/pycaret/pycaret and https://pycaret.gitbook.io/docs/get-started/functions/initialize; MLJAR https://github.com/mljar/mljar-supervised (README keyword scans).
- H2O AutoML: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html (page scanned; `leaderboard_frame` text quoted).
- H2O Driverless AI: https://docs.h2o.ai/driverless-ai/1-10-lts/docs/userguide/leakage-shift-detection.html (via search extract of the 1.10 LTS and 1.10.7 pages).
- DataRobot: https://docs.datarobot.com/en/docs/reference/data-ref/data-quality-ref.html (page read: target leakage thresholds and feature list); https://docs.datarobot.com/en/modeling/build-models/build-basic/unlocking-holdout.html (via search extract).
- Azure AutoML data guardrails: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features (archived SDK v1 page read in full).
- mcp-automl: https://github.com/idea7766/mcp-automl (README read).
- AIDE https://github.com/WecoAI/aideml; MLE-bench https://github.com/openai/mle-bench (README: documented leaky competitions); MLAgentBench https://github.com/snap-stanford/MLAgentBench; AutoKaggle https://github.com/multimodal-art-projection/AutoKaggle; DS-Agent https://github.com/guosyjlu/DS-Agent (README scans).

Data validation
- Deepchecks: https://docs.deepchecks.com/stable/tabular/auto_checks/train_test_validation/index.html (check list read); https://docs.deepchecks.com/stable/general/usage/ci_cd.html (CI guide read).
- Great Expectations: https://docs.greatexpectations.io/docs/core/trigger_actions_based_on_results/create_a_checkpoint_with_actions/ (page scanned).
- TFDV https://github.com/tensorflow/data-validation; TFX ExampleValidator https://www.tensorflow.org/tfx/guide/exampleval (page scanned).
- Evidently https://docs.evidentlyai.com/docs/library/tests (page scanned); whylogs https://github.com/whylabs/whylogs; Soda https://github.com/sodadata/soda-core; Pandera https://github.com/unionai-oss/pandera; ydata-profiling https://github.com/ydataai/ydata-profiling (README scans; the whylogs README returned only a stub).
- Cleanlab Datalab issue types: https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html (page scanned).
- SageMaker Data Wrangler analyses: https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-analyses.html (page scanned).
- targetleak https://pypi.org/project/targetleak/ ; leakr https://cran.r-project.org/web/packages/leakr/index.html (existence confirmed; descriptions via search extract).

Tracking, MLOps, execution
- MLflow https://github.com/mlflow/mlflow ; MLflow MCP server https://mlflow.org/docs/latest/genai/mcp/ (page read).
- ZenML https://github.com/zenml-io/zenml (README: MCP section read); DVC https://github.com/iterative/dvc; ClearML https://github.com/clearml/clearml; Kedro https://github.com/kedro-org/kedro; Metaflow https://github.com/Netflix/metaflow (README scans). Weights & Biases https://docs.wandb.ai/ (URL only).
- SkyPilot https://github.com/skypilot-org/skypilot ; admin policies https://docs.skypilot.ai/en/latest/cloud-setup/policy.html (page read).
- dstack https://github.com/dstackai/dstack ; task reference https://dstack.ai/docs/reference/dstack.yml/task/ (page read).
- Modal https://modal.com/docs ; RunPod https://docs.runpod.io/ ; Vast.ai https://docs.vast.ai/ (URLs only, not audited).

Privacy and supply chain
- Microsoft Presidio https://github.com/microsoft/presidio (README scan); Opacus https://github.com/pytorch/opacus (README scan).
- OpenSSF Model Signing / sigstore model-transparency https://github.com/sigstore/model-transparency (README read; releases page read: 1.0.0 first stable, 1.1.1 latest).
- in-toto attestation predicates https://github.com/in-toto/attestation/tree/main/spec/predicates (directory listing scanned).
- SLSA specification v1.2 https://slsa.dev/spec/v1.2/ (URL resolves); Google Security Blog, "Increasing transparency in AI security", October 2023, https://security.googleblog.com/2023/10/increasing-transparency-in-ai-security.html (URL resolves; not read in full).
- CycloneDX ML-BOM https://cyclonedx.org/capabilities/mlbom/ (page read); v1.6 JSON schema reference https://cyclonedx.org/docs/1.6/json/ (field names confirmed).
- annex4ac https://pypi.org/project/annex4ac/ (existence confirmed; description via search extract).

Law and standards (none of this is legal advice)
- EU AI Act, Regulation (EU) 2024/1689: https://eur-lex.europa.eu/eli/reg/2024/1689/oj (URL resolves; article numbers cited from general knowledge of the regulation).
- ISO/IEC 42001:2023: https://www.iso.org/standard/42001 (HTTP 403 to the checker; not read).
- PIPEDA: https://laws-lois.justice.gc.ca/eng/acts/P-8.6/ (URL resolves).
- OPC et al., "Principles for responsible, trustworthy and privacy-protective generative AI technologies", 7 December 2023: https://www.priv.gc.ca/en/privacy-topics/technology/artificial-intelligence/gd_principles_ai/ (page read).
- Bill C-27 (44-1), containing AIDA: https://www.parl.ca/legisinfo/en/bill/44-1/c-27 (page read: stopped at committee; session ended 6 January 2025).
- Bill C-36 (45-1), Protecting Privacy and Consumer Data Act: https://www.parl.ca/legisinfo/en/bill/45-1/c-36 (page read: first reading 15 June 2026, at second reading).
- Quebec, Act respecting the protection of personal information in the private sector (P-39.1): https://www.legisquebec.gouv.qc.ca/en/document/cs/P-39.1 (HTTP 403 to the checker; s. 17 summarised from secondary legal sources).
- Brazil, Lei 13.709/2018 (LGPD): https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm (URL resolves); ANPD international transfers page https://www.gov.br/anpd/pt-br/assuntos/assuntos-internacionais/transferencia-internacional-de-dados/international-affairs (via search; Resolution CD/ANPD 19/2024 summarised from secondary sources).

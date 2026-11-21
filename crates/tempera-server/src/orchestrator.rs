//! Runs a job end to end: ingest → profile → infer → audit → split →
//! candidates (via the local agent) → leaderboard → one-time test evaluation →
//! report. See docs/wiki/User-Journey.md.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, anyhow, bail};
use tempera_audit::{AuditReport, Severity};
use tempera_core::metrics::higher_is_better;
use tempera_core::spec::{ExportFormat, JobSpec, TaskType};
use tempera_data::eval::{Evaluation, evaluate, metric_name};
use tempera_data::split::{FeatureKind, Partition, Split};
use tempera_data::{Dataset, infer_task, ingest, make_split, profile_dataset};
use tempera_exec::{Hub, LocalExecutor, local::LOCAL_MACHINE};
use tempera_native::{FeatureKind as NFeatureKind, FeatureSpec, Manifest, PredictManifest, PredictTarget};
use tempera_proto::v1::{Backend, Command as JobCommand, StartJob, agent_message};
use tempera_report::{CostSummary, DatasetSummary, ReportData, RunSummary, Winner};
use tempera_track::{MetricPoint, Tracker};

use crate::builtin;
use crate::candidates::{BackendKind, Candidate, candidates};
use crate::events::{self, Eta, JobEvent};
use crate::hpo;
use crate::ui;
use crate::workspace::{Workspace, python_available, worker_dir};

/// Result of a finished (or blocked) job.
#[derive(Debug)]
#[allow(dead_code)] // job_id/report_path are part of the API for the M2 server
pub struct JobOutcome {
    pub job_id: String,
    pub blocked: bool,
    pub report_path: Option<PathBuf>,
}

/// Everything needed before training: shared by `tempera audit` and `tempera run`.
pub struct Prepared {
    pub spec: JobSpec,
    pub ds: Dataset,
    pub inference: tempera_data::TaskInference,
    pub audit: AuditReport,
}

pub fn load_spec(path: &Path) -> anyhow::Result<(JobSpec, String)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let spec = parse_spec(&text, base).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    Ok((spec, text))
}

/// Parses a spec; relative data paths are resolved against `base`.
pub fn parse_spec(text: &str, base: &Path) -> anyhow::Result<JobSpec> {
    let mut spec = JobSpec::from_yaml(text).map_err(|e| anyhow!("{e}"))?;
    let source = Path::new(&spec.data.source);
    if source.is_relative() && !spec.data.source.contains("://") {
        spec.data.source = base.join(source).display().to_string();
    }
    Ok(spec)
}

/// Ingest, profile, infer and audit (blocking work; call from `spawn_blocking`).
pub fn prepare(ws: &Workspace, spec: JobSpec) -> anyhow::Result<Prepared> {
    let t = Instant::now();
    let ds = ingest(Path::new(&spec.data.source), &ws.root)?;
    ui::step(&format!(
        "Data      {} rows × {} columns  ({})",
        ui::count(ds.rows),
        ds.columns.len(),
        ui::secs(t.elapsed().as_secs_f64())
    ));
    let profile = profile_dataset(&ds)?;
    let inference = infer_task(&spec, &profile)?;
    ui::step(&format!(
        "Task      {} · metric {}",
        ui::snake(&inference.task),
        metric_name(inference.metric)
    ));
    for r in &inference.reasons {
        ui::detail(r);
    }
    let mut audit = tempera_audit::audit(&ds, &spec, &profile, &inference)?;
    // The local SLM advisor suggests; it never blocks (ADR-0004).
    match tempera_audit::advisor::advise(&spec.audit.advisor, &profile, spec.data.target.as_deref()) {
        tempera_audit::advisor::AdvisorResult::Advice { model, columns } => {
            // Columns already flagged or not used need no extra advice.
            let flagged: Vec<&str> = audit
                .findings
                .iter()
                .filter_map(|f| f.column.as_deref())
                .chain(audit.plan.dropped.keys().map(String::as_str))
                .collect();
            let extra = tempera_audit::advisor::findings(&columns, &profile, &flagged, &spec.data.sensitive);
            ui::detail(&format!("Advisor ({model}): {} suggestion(s)", extra.len()));
            tempera_audit::add_findings(&mut audit, &spec, extra);
        }
        tempera_audit::advisor::AdvisorResult::Skipped(why) => ui::detail(&format!("Advisor skipped: {why}")),
    }
    tempera_audit::save(&ds, &audit)?;
    Ok(Prepared {
        spec,
        ds,
        inference,
        audit,
    })
}

pub fn print_audit(audit: &AuditReport) {
    let reds = audit
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Red)
        .count();
    let yellows = audit
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Yellow)
        .count();
    ui::step(&format!(
        "Audit     {reds} red · {yellows} yellow · {} features used · {} columns dropped",
        audit.plan.features.len() + audit.plan.datetime.len(),
        audit.plan.dropped.len()
    ));
    for f in &audit.findings {
        if f.severity == Severity::Green {
            continue;
        }
        ui::finding(f);
    }
}

pub async fn run_job(ws: &Workspace, spec_path: &Path) -> anyhow::Result<JobOutcome> {
    let (spec, spec_text) = load_spec(spec_path)?;
    run_spec(ws, spec, spec_text, &spec_path.display().to_string(), None).await
}

/// Runs a parsed spec. `job_id` is pre-assigned by the HTTP server (which has
/// already registered it for live events); the CLI lets it be generated.
pub async fn run_spec(
    ws: &Workspace,
    spec: JobSpec,
    spec_text: String,
    spec_label: &str,
    job_id: Option<String>,
) -> anyhow::Result<JobOutcome> {
    let job_id = match job_id {
        Some(id) => id,
        None => {
            // CLI: register the job so budget caps, preemption and Ctrl-C can
            // cancel it cleanly (a rented instance is then terminated, not orphaned).
            let id = new_job_id(&spec.name);
            events::register(&id);
            let cancel_id = id.clone();
            tokio::spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    eprintln!("\ncancelling… (cleaning up; press Ctrl-C again to force)");
                    events::request_cancel(&cancel_id);
                    if tokio::signal::ctrl_c().await.is_ok() {
                        std::process::exit(130);
                    }
                }
            });
            id
        }
    };
    let result = run_spec_inner(ws, spec, spec_text, spec_label, &job_id).await;
    let (status, error, report_url) = match &result {
        Ok(o) if o.blocked => ("blocked", None, None),
        Ok(_) if events::is_cancelled(&job_id) => {
            ("cancelled", None, Some(format!("/api/v1/jobs/{job_id}/report")))
        }
        Ok(_) => ("succeeded", None, Some(format!("/api/v1/jobs/{job_id}/report"))),
        Err(e) => ("failed", Some(format!("{e:#}")), None),
    };
    events::emit(
        &job_id,
        JobEvent::JobFinished {
            status,
            report_url,
            error,
        },
    );
    result
}

pub fn generate_job_id(name: &str) -> String {
    new_job_id(name)
}

fn stage(job_id: &str, stage: &'static str, message: String) {
    events::emit(job_id, JobEvent::Stage { stage, message });
}

async fn run_spec_inner(
    ws: &Workspace,
    spec: JobSpec,
    spec_text: String,
    spec_label: &str,
    job_id: &str,
) -> anyhow::Result<JobOutcome> {
    let started = Instant::now();
    let job_id = job_id.to_owned();
    let tracker = Tracker::open(&ws.db_path()).await?;
    tracker.create_job(&job_id, &spec.name, &spec_text).await?;
    stage(&job_id, "ingest", format!("reading {}", spec.data.source));
    ui::header(&format!("Job {job_id}"));

    let ws2 = ws.clone();
    let prepared = tokio::task::spawn_blocking(move || prepare(&ws2, spec)).await?;
    let prepared = match prepared {
        Ok(p) => p,
        Err(e) => {
            tracker
                .set_job_status(&job_id, "failed", Some(&e.to_string()))
                .await?;
            return Err(e);
        }
    };
    let Prepared {
        spec,
        ds,
        inference,
        mut audit,
    } = prepared;
    print_audit(&audit);
    events::emit(
        &job_id,
        JobEvent::Audit {
            inference: serde_json::to_value(&inference)?,
            audit: serde_json::to_value(&audit)?,
        },
    );
    if audit.blocked {
        tracker
            .update_job_context(
                &job_id,
                &ds.id,
                None,
                &ui::snake(&inference.task),
                &metric_name(inference.metric),
            )
            .await?;
        tracker
            .set_job_status(&job_id, "blocked", Some("audit found unresolved red findings"))
            .await?;
        ui::blocked(&audit);
        events::emit(
            &job_id,
            JobEvent::Blocked {
                findings: serde_json::to_value(audit.blocking().collect::<Vec<_>>())?,
            },
        );
        return Ok(JobOutcome {
            job_id,
            blocked: true,
            report_path: None,
        });
    }

    stage(&job_id, "split", "building train / valid / test".into());
    let (spec2, ds2, plan, task) = (spec.clone(), ds.clone(), audit.plan.clone(), inference.task);
    let split = tokio::task::spawn_blocking(move || make_split(&ds2, &spec2, task, &plan)).await??;
    // A6: does the held-out test set look like the training data?
    let drift = {
        let (train, test, features, strategy) = (
            split.path(Partition::Train),
            split.path(Partition::Test),
            split.features.clone(),
            split.strategy,
        );
        tokio::task::spawn_blocking(move || {
            tempera_audit::advanced::drift(&train, &test, &features, strategy)
        })
        .await??
    };
    for f in &drift {
        ui::finding(f);
    }
    tempera_audit::add_findings(&mut audit, &spec, drift);
    tempera_audit::save(&ds, &audit)?;
    tracker
        .update_job_context(
            &job_id,
            &ds.id,
            Some(&split.id),
            &ui::snake(&task),
            &metric_name(inference.metric),
        )
        .await?;
    ui::step(&format!(
        "Split     {} · train {} · valid {} · test {} (held out)",
        ui::snake(&split.strategy),
        ui::count(split.rows["train"]),
        ui::count(split.rows["valid"]),
        ui::count(split.rows["test"]),
    ));

    let job_dir = ws.job_dir(&job_id);
    std::fs::create_dir_all(&job_dir)?;
    let result = train_and_report(
        ws, &tracker, &job_id, &job_dir, &spec, &spec_text, spec_label, &ds, &inference, &mut audit, &split,
        started,
    )
    .await;
    match &result {
        Ok(_) if events::is_cancelled(&job_id) => tracker.set_job_status(&job_id, "cancelled", None).await?,
        Ok(_) => tracker.set_job_status(&job_id, "succeeded", None).await?,
        Err(e) => {
            tracker
                .set_job_status(&job_id, "failed", Some(&e.to_string()))
                .await?
        }
    }
    result.map(|report| JobOutcome {
        job_id,
        blocked: false,
        report_path: Some(report),
    })
}

#[allow(clippy::too_many_arguments)]
async fn train_and_report(
    ws: &Workspace,
    tracker: &Tracker,
    job_id: &str,
    job_dir: &Path,
    spec: &JobSpec,
    spec_text: &str,
    spec_label: &str,
    ds: &Dataset,
    inference: &tempera_data::TaskInference,
    audit: &mut AuditReport,
    split: &Split,
    started: Instant,
) -> anyhow::Result<PathBuf> {
    let task = inference.task;
    let metric = inference.metric;
    let cands = candidates(spec, task).map_err(|e| anyhow!("strategy.candidates: {e}"))?;
    let worker = worker_dir();
    let python = python_available(worker.as_deref());

    // Bring up an agent only if something needs it: on this machine, or on the
    // first SSH machine named in `compute.allow` (see `tempera machines`).
    let needs_agent = cands.iter().any(|c| c.backend != BackendKind::Builtin);
    let hub = Hub::new();
    let mut agent: Option<Agent> = None;
    // A rented instance: terminated by `finish` below, or on drop if anything fails.
    let mut cloud: Option<(crate::cloud::Plan, crate::cloud::CloudInstance)> = None;
    if needs_agent {
        let (addr, _server) = hub.serve("127.0.0.1:0".parse()?).await?;
        let registry =
            tempera_exec::machines::MachinesFile::load(&tempera_exec::machines::machines_path(&ws.root))?;
        let ssh_name = spec
            .compute
            .allow
            .iter()
            .find(|n| n.as_str() != "local" && registry.get(n).is_some());
        let cloud_name = crate::cloud::provider_in(spec, |n| registry.get(n).is_some());
        // Ship the Python worker only when a candidate needs it (native jobs need no uv).
        let needs_python = cands.iter().any(|c| c.backend == BackendKind::Python);
        let ssh_worker = if needs_python { worker.clone() } else { None };
        let started = match (ssh_name, cloud_name) {
            (Some(name), _) => {
                let ssh = registry
                    .get(name)
                    .ok_or_else(|| anyhow!("machine `{name}` vanished"))?
                    .to_ssh()?;
                start_ssh(
                    &hub,
                    addr,
                    ssh,
                    format!("{name} (ssh)"),
                    ssh_worker,
                    job_id,
                    job_dir,
                )
                .await?
            }
            (None, Some(provider)) => {
                stage(job_id, "train", format!("estimating machines on {provider}"));
                let w = crate::cloud::workload(spec, ds.rows, ds.columns.len() as u64, &cands);
                let include_local = spec.compute.allow.iter().any(|n| n == "local");
                let plan = crate::cloud::plan(&ws.root, spec, &provider, w, include_local).await?;
                if plan.chosen().is_some() {
                    crate::cloud::check_budget(&plan)?;
                    let (inst, ssh) = crate::cloud::CloudInstance::launch(&ws.root, job_id, &plan).await?;
                    let label = format!("{} (cloud)", inst.offer.label());
                    // On error `inst` drops here and terminates the instance.
                    let a = start_ssh(&hub, addr, ssh, label, ssh_worker, job_id, job_dir).await?;
                    cloud = Some((plan, inst));
                    a
                } else if plan.recommendation.best().is_some() {
                    ui::detail("this machine wins; nothing to rent");
                    Agent::Local(
                        LocalExecutor::start(&hub, &format!("http://{addr}"), worker.as_deref()).await?,
                    )
                } else {
                    return Err(anyhow!("no {provider} offer fits this job"));
                }
            }
            (None, None) => {
                Agent::Local(LocalExecutor::start(&hub, &format!("http://{addr}"), worker.as_deref()).await?)
            }
        };
        ui::step(&format!("Machine   {}", started.describe()));
        agent = Some(started);
    }
    let machine = agent.as_ref().map_or_else(|| "local".to_owned(), Agent::describe);

    ui::step(&format!(
        "Training  {} candidates · ranked by {} on validation",
        cands.len(),
        metric_name(metric)
    ));
    // A7: suspected label errors (Python worker, cleanlab); never blocks.
    if python.is_ok() && agent.is_some() && !events::is_cancelled(job_id) {
        stage(job_id, "audit", "checking labels for likely errors".into());
        if let Some(f) = crate::quality::label_issues(tracker, &hub, job_id, job_dir, split).await {
            ui::finding(&f);
            tempera_audit::add_findings(audit, spec, vec![f]);
        }
    }
    stage(job_id, "train", format!("{} candidates", cands.len()));
    let mut summaries: Vec<RunSummary> = Vec::new();
    for (i, cand) in cands.iter().enumerate() {
        let run_id = format!("{job_id}-r{}", i + 1);
        events::emit(
            job_id,
            JobEvent::RunStarted {
                run_id: run_id.clone(),
                algorithm: cand.algorithm.clone(),
                backend: cand.backend.as_str().to_owned(),
                index: i + 1,
                total: cands.len(),
            },
        );
        // Job ETA: the runs after this one at the median duration of the finished ones.
        let mut done: Vec<f64> = summaries
            .iter()
            .filter(|s| s.status == "succeeded")
            .map(|s| s.duration_s)
            .collect();
        done.sort_by(f64::total_cmp);
        let rest_s = done.get(done.len() / 2).map(|m| m * (cands.len() - i - 1) as f64);
        // Test hook: cancel deterministically once `n` candidates have run.
        if std::env::var("TEMPERA_TEST_CANCEL_AFTER_RUNS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            == Some(i)
        {
            events::request_cancel(job_id);
        }
        let over_budget = budget_left(spec, started).is_some_and(|left| left <= 0.0);
        let opts = RunOpts {
            rest_s,
            ..RunOpts::default()
        };
        let summary = if events::is_cancelled(job_id) {
            skipped(tracker, job_id, &run_id, cand, "cancelled").await?
        } else if over_budget {
            skipped(tracker, job_id, &run_id, cand, "time budget exhausted").await?
        } else if let (BackendKind::Python, Err(why)) = (cand.backend, &python) {
            skipped(tracker, job_id, &run_id, cand, why).await?
        } else if let (Some(h), Some(space)) = (
            &spec.strategy.hpo,
            hpo::space(&cand.algorithm, cand.backend.as_str()),
        ) {
            let search = Search {
                max_trials: h.max_trials as usize,
                space,
                started,
            };
            run_search(
                tracker, &hub, job_id, &run_id, job_dir, cand, spec, split, task, metric, &search,
            )
            .await?
        } else {
            run_candidate(
                tracker, &hub, job_id, &run_id, job_dir, cand, spec, split, task, metric, &opts,
            )
            .await?
        };
        ui::run_line(&summary, metric);
        events::emit(
            job_id,
            JobEvent::RunFinished {
                run_id: run_id.clone(),
                status: summary.status.clone(),
                valid_score: summary.valid_score.filter(|v| v.is_finite()),
                valid_metrics: serde_json::to_value(&summary.valid_metrics)?,
                duration_s: summary.duration_s,
                error: summary.error.clone(),
            },
        );
        summaries.push(summary);
    }
    stage(
        job_id,
        "evaluate",
        "evaluating the winner once on the test set".into(),
    );

    // Leaderboard: best validation score first; failed/skipped last.
    let hib = higher_is_better(metric);
    summaries.sort_by(|a, b| {
        match (
            a.valid_score.filter(|v| v.is_finite()),
            b.valid_score.filter(|v| v.is_finite()),
        ) {
            (Some(x), Some(y)) => {
                if hib {
                    y.total_cmp(&x)
                } else {
                    x.total_cmp(&y)
                }
            }
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });

    let unranked = task == TaskType::AnomalyDetection && split.classes.is_empty();
    // A cancelled job (user, budget, preemption) stops before the test set: evaluating it
    // would spend the one look at the test data on a selection that never finished (#89).
    let cancelled = events::is_cancelled(job_id);
    if cancelled {
        ui::detail("cancelled: the test set was not evaluated and remains unused");
    }
    let winner = match summaries
        .iter()
        .find(|s| !cancelled && s.status == "succeeded" && (s.valid_score.is_some() || unranked))
    {
        None => None,
        Some(best) => {
            let cand = cands
                .iter()
                .find(|c| c.algorithm == best.algorithm && c.backend.as_str() == best.backend)
                .cloned();
            match cand {
                None => None,
                Some(cand) => {
                    Some(evaluate_winner(tracker, &hub, job_id, job_dir, best, &cand, split, task).await?)
                }
            }
        }
    };
    // A8: fairness of the winner's test predictions per sensitive attribute.
    if let Some(w) = &winner
        && python.is_ok()
        && agent.is_some()
        && !spec.data.sensitive.is_empty()
    {
        let test_pred = job_dir.join("runs").join(&w.run_id).join("test_pred.parquet");
        let found = crate::quality::fairness(tracker, &hub, job_id, job_dir, split, &test_pred).await;
        for f in &found {
            ui::finding(f);
        }
        tempera_audit::add_findings(audit, spec, found);
    }
    tempera_audit::save(ds, audit)?;
    if let Some(a) = agent {
        a.stop(&hub).await;
    }
    crate::remote::remove(job_id);
    // Naive baseline for the savings line: exhaustive search instead of successive halving.
    let naive_multiplier = if spec.strategy.hpo.is_some() {
        1.0 / 0.36
    } else {
        1.0
    };
    let cloud_cost = match cloud.take() {
        Some((plan, inst)) => Some(inst.finish(&plan, naive_multiplier).await),
        None => None,
    };
    if let Some(c) = &cloud_cost {
        ui::step(&format!(
            "Cost      estimated {}  ·  actual {}",
            tempera_cloud::estimate::fmt_usd(c.estimated_usd),
            tempera_cloud::estimate::fmt_usd(c.actual_usd)
        ));
        if let Some(s) = &c.savings {
            ui::detail(&s.line());
        }
        events::emit(
            job_id,
            JobEvent::Cost {
                machine: c.machine.clone(),
                usd_per_hour: 0.0,
                estimated_usd: c.estimated_usd,
                spent_usd: c.actual_usd,
                budget_usd: crate::cloud::budget_usd(spec).ok().flatten(),
                status: "terminated",
                savings: c.savings.as_ref().map(tempera_cloud::estimate::Savings::line),
            },
        );
    }
    if let Some(w) = &winner {
        ui::verdict(w, metric, &summaries);
        events::emit(
            job_id,
            JobEvent::Winner {
                run_id: w.run_id.clone(),
                algorithm: w.algorithm.clone(),
                test: serde_json::to_value(&w.test)?,
            },
        );
    }
    stage(job_id, "report", "writing the report".into());

    let data = ReportData {
        job_id: job_id.to_owned(),
        job_name: spec.name.clone(),
        created_at: tempera_track::now(),
        tempera_version: env!("CARGO_PKG_VERSION").to_owned(),
        dataset: DatasetSummary {
            id: ds.id.clone(),
            source: ds.source.display().to_string(),
            rows: ds.rows,
            columns: ds.columns.len() as u64,
        },
        task,
        metric,
        task_reasons: inference.reasons.clone(),
        audit: audit.clone(),
        split: split.clone(),
        leaderboard: summaries,
        winner: winner.clone(),
        cost: CostSummary {
            wall_time_s: started.elapsed().as_secs_f64(),
            machine,
            cost_usd: cloud_cost.as_ref().map_or(0.0, |c| c.actual_usd),
            estimated_cost_usd: cloud_cost.as_ref().map(|c| c.estimated_usd),
            savings: cloud_cost
                .as_ref()
                .and_then(|c| c.savings.as_ref().map(tempera_cloud::estimate::Savings::line)),
        },
        spec_yaml: spec_text.to_owned(),
        reproduce_command: format!("tempera run {spec_label}"),
    };
    let report = job_dir.join("report.html");
    std::fs::write(&report, tempera_report::render(&data))?;
    std::fs::write(job_dir.join("report.json"), serde_json::to_string_pretty(&data)?)?;
    let test_json = winner
        .as_ref()
        .map(|w| serde_json::to_string(&w.test))
        .transpose()?;
    tracker
        .finish_job(
            job_id,
            winner.as_ref().map(|w| w.run_id.as_str()),
            test_json.as_deref(),
            Some(&report.display().to_string()),
        )
        .await?;
    let _ = ws;
    ui::done(&report, started.elapsed().as_secs_f64());
    Ok(report)
}

/// Seconds left of `budget.max_time` (None when there is no time budget).
fn budget_left(spec: &JobSpec, started: Instant) -> Option<f64> {
    spec.budget
        .max_time
        .map(|t| t.as_duration().as_secs_f64() - started.elapsed().as_secs_f64())
}

struct Search {
    max_trials: usize,
    space: Vec<(&'static str, hpo::Dist)>,
    started: Instant,
}

fn best_first(
    hib: bool,
) -> impl Fn(&(f64, usize, RunSummary), &(f64, usize, RunSummary)) -> std::cmp::Ordering {
    move |a, b| {
        if hib {
            b.0.total_cmp(&a.0)
        } else {
            a.0.total_cmp(&b.0)
        }
    }
}

/// Successive halving over training-set fractions (see `hpo`). Returns the best
/// trial of the largest rung reached; every trial is a tracked run.
#[allow(clippy::too_many_arguments)]
async fn run_search(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    run_id: &str,
    job_dir: &Path,
    cand: &Candidate,
    spec: &JobSpec,
    split: &Split,
    task: TaskType,
    metric: tempera_core::spec::Metric,
    search: &Search,
) -> anyhow::Result<RunSummary> {
    let hib = higher_is_better(metric);
    let salt = cand
        .algorithm
        .bytes()
        .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    let mut sampler = hpo::Sampler::new(spec.split.seed ^ salt);
    // Trial 1 is always the backend defaults, so a search never does worse than not searching.
    let mut alive: Vec<serde_json::Map<String, serde_json::Value>> = std::iter::once(serde_json::Map::new())
        .chain((1..search.max_trials.max(1)).map(|_| sampler.sample(&search.space)))
        .collect();
    let mut trial = 0usize;
    let mut best: Option<RunSummary> = None;
    'rungs: for (r, frac) in hpo::RUNGS.iter().enumerate() {
        let train = {
            let (split, f, seed) = (split.clone(), *frac, spec.split.seed);
            tokio::task::spawn_blocking(move || tempera_data::split::train_fraction(&split, f, seed))
                .await??
        };
        let mut scored: Vec<(f64, usize, RunSummary)> = Vec::new();
        for (i, params) in alive.iter().enumerate() {
            let stop = if events::is_cancelled(job_id) {
                Some("cancelled")
            } else if budget_left(spec, search.started).is_some_and(|l| l <= 0.0) {
                Some("time budget exhausted")
            } else {
                None
            };
            if let Some(why) = stop {
                ui::detail(&format!(
                    "{}: search stopped at rung {} ({why})",
                    cand.algorithm,
                    r + 1
                ));
                scored.sort_by(best_first(hib));
                if let Some(top) = scored.into_iter().next() {
                    best = Some(top.2);
                }
                break 'rungs;
            }
            trial += 1;
            let trial_id = format!("{run_id}-t{trial}");
            events::emit(
                job_id,
                JobEvent::RunStarted {
                    run_id: trial_id.clone(),
                    algorithm: format!("{} · trial {trial} · {:.0}% data", cand.algorithm, frac * 100.0),
                    backend: cand.backend.as_str().to_owned(),
                    index: i + 1,
                    total: alive.len(),
                },
            );
            let opts = RunOpts {
                params: params.clone(),
                train: Some(train.clone()),
                rest_s: None,
            };
            let s = run_candidate(
                tracker, hub, job_id, &trial_id, job_dir, cand, spec, split, task, metric, &opts,
            )
            .await?;
            events::emit(
                job_id,
                JobEvent::RunFinished {
                    run_id: trial_id,
                    status: s.status.clone(),
                    valid_score: s.valid_score.filter(|v| v.is_finite()),
                    valid_metrics: serde_json::to_value(&s.valid_metrics)?,
                    duration_s: s.duration_s,
                    error: s.error.clone(),
                },
            );
            if let Some(v) = s.valid_score.filter(|v| v.is_finite()) {
                scored.push((v, i, s));
            }
        }
        scored.sort_by(best_first(hib));
        let last = r + 1 == hpo::RUNGS.len();
        if !last && !scored.is_empty() {
            let keep = hpo::survivors(alive.len());
            let mut next: Vec<_> = scored
                .iter()
                .take(keep)
                .map(|(_, i, _)| alive[*i].clone())
                .collect();
            // Small rungs are noisy: the defaults always reach the full-data rung.
            if !next.iter().any(serde_json::Map::is_empty) && alive.iter().any(serde_json::Map::is_empty) {
                next.push(serde_json::Map::new());
            }
            alive = next;
        }
        if let Some(top) = scored.into_iter().next() {
            best = Some(top.2);
        } else {
            break;
        }
        if last {
            break;
        }
    }
    let mut best = best.ok_or_else(|| anyhow!("every {} trial failed", cand.algorithm))?;
    best.valid_metrics.insert("hpo_trials".into(), trial as f64);
    Ok(best)
}

async fn skipped(
    tracker: &Tracker,
    job_id: &str,
    run_id: &str,
    cand: &Candidate,
    why: &str,
) -> anyhow::Result<RunSummary> {
    tracker
        .start_run(
            run_id,
            job_id,
            &cand.algorithm,
            cand.backend.as_str(),
            cand.is_baseline,
            &serde_json::json!({}),
        )
        .await?;
    tracker
        .finish_run(run_id, "skipped", None, &BTreeMap::new(), None, Some(why), 0.0)
        .await?;
    Ok(summary(
        run_id,
        cand,
        "skipped",
        None,
        BTreeMap::new(),
        0.0,
        Some(why.to_owned()),
    ))
}

fn summary(
    run_id: &str,
    cand: &Candidate,
    status: &str,
    valid_score: Option<f64>,
    valid_metrics: BTreeMap<String, f64>,
    duration_s: f64,
    error: Option<String>,
) -> RunSummary {
    RunSummary {
        run_id: run_id.to_owned(),
        algorithm: cand.algorithm.clone(),
        backend: cand.backend.as_str().to_owned(),
        is_baseline: cand.is_baseline,
        status: status.to_owned(),
        valid_score,
        valid_metrics,
        duration_s,
        error,
    }
}

/// Per-run overrides: hyperparameters (HPO trials) and a smaller train set (HPO rungs).
#[derive(Debug, Default, Clone)]
struct RunOpts {
    params: serde_json::Map<String, serde_json::Value>,
    train: Option<PathBuf>,
    rest_s: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
async fn run_candidate(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    run_id: &str,
    job_dir: &Path,
    cand: &Candidate,
    spec: &JobSpec,
    split: &Split,
    task: TaskType,
    metric: tempera_core::spec::Metric,
    opts: &RunOpts,
) -> anyhow::Result<RunSummary> {
    let t = Instant::now();
    let work = job_dir.join("runs").join(run_id);
    std::fs::create_dir_all(&work)?;
    let valid_pred = work.join("valid_pred.parquet");
    tracker
        .start_run(
            run_id,
            job_id,
            &cand.algorithm,
            cand.backend.as_str(),
            cand.is_baseline,
            &serde_json::Value::Object(opts.params.clone()),
        )
        .await?;

    let outcome: anyhow::Result<(Option<String>, BTreeMap<String, f64>)> = match cand.backend {
        BackendKind::Builtin => {
            let (train, valid, out) = (
                split.path(Partition::Train),
                split.path(Partition::Valid),
                valid_pred.clone(),
            );
            let classes = split.classes.clone();
            let algo = cand.algorithm.clone();
            tokio::task::spawn_blocking(move || match algo.as_str() {
                "majority" => builtin::majority(&train, &classes, &valid, &out),
                _ => builtin::mean(&train, &valid, &out),
            })
            .await?
            .map(|()| (None, BTreeMap::new()))
        }
        BackendKind::Native | BackendKind::Python => {
            let mut manifest = fit_manifest(cand, spec, split, task, &work, &valid_pred);
            manifest.params = opts.params.clone();
            if let Some(train) = &opts.train {
                manifest.train = abs(train);
            }
            dispatch(
                tracker,
                hub,
                job_id,
                run_id,
                cand.backend,
                JobCommand::Fit,
                serde_json::to_string(&manifest)?,
                true,
                opts.rest_s,
            )
            .await
            .map(|evals| (Some(work.join("model").display().to_string()), evals))
        }
    };
    let duration = t.elapsed().as_secs_f64();
    match outcome {
        Ok((model_dir, evals)) => {
            let mut e = evaluate(
                task,
                &split.classes,
                split.positive_class.as_deref(),
                &split.path(Partition::Valid),
                &valid_pred,
            )?;
            e.metrics.extend(evals);
            let score = e.get(metric);
            tracker
                .finish_run(
                    run_id,
                    "succeeded",
                    score,
                    &e.metrics,
                    model_dir.as_deref(),
                    None,
                    duration,
                )
                .await?;
            Ok(summary(
                run_id,
                cand,
                "succeeded",
                score,
                e.metrics,
                duration,
                None,
            ))
        }
        Err(err) => {
            let msg = err.to_string();
            tracker
                .finish_run(
                    run_id,
                    "failed",
                    None,
                    &BTreeMap::new(),
                    None,
                    Some(&msg),
                    duration,
                )
                .await?;
            Ok(summary(
                run_id,
                cand,
                "failed",
                None,
                BTreeMap::new(),
                duration,
                Some(msg),
            ))
        }
    }
}

/// Run-id suffix of the winner's test prediction.
const TEST_RUN_SUFFIX: &str = "-test";

/// Sends one job to the local agent and drains its messages into the tracker.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    run_id: &str,
    backend: BackendKind,
    command: JobCommand,
    manifest_json: String,
    record: bool,
    rest_s: Option<f64>,
) -> anyhow::Result<BTreeMap<String, f64>> {
    let start = StartJob {
        job_id: job_id.to_owned(),
        run_id: run_id.to_owned(),
        backend: if backend == BackendKind::Python {
            Backend::Python
        } else {
            Backend::Native
        } as i32,
        command: command as i32,
        manifest_json,
        ..Default::default()
    };
    // On an SSH machine, inputs go to its cache and outputs come back after the command.
    let remote = crate::remote::get(job_id);
    let machine = remote
        .as_ref()
        .map_or_else(|| LOCAL_MACHINE.to_owned(), |r| r.machine_id.clone());
    let mut start = start;
    let mut plan = None;
    if let Some(r) = &remote {
        let mut v: serde_json::Value = serde_json::from_str(&start.manifest_json)?;
        plan = Some(r.rewrite(&mut v).await?);
        start.manifest_json = v.to_string();
    }
    // The test prediction runs to completion once started: the test set is already claimed,
    // so interrupting it would burn the one evaluation and still fail the job (#89).
    let cancellable = !run_id.ends_with(TEST_RUN_SUFFIX);
    let mut rx = hub.start(&machine, start).await?;
    let mut points = Vec::new();
    // Backend-computed evaluation metrics (clustering quality, §8), last value wins.
    let mut evals = BTreeMap::new();
    let mut eta = Eta::default();
    let mut cancel_sent = false;
    // ETA quality (#27): predicted total duration taken at ~10% progress vs the real one.
    let started = Instant::now();
    let mut predicted_total: Option<f64> = None;
    loop {
        let m = tokio::select! {
            m = rx.recv() => m,
            () = events::cancelled(job_id), if cancellable && !cancel_sent => {
                cancel_sent = true;
                let _ = hub.cancel(&machine, job_id, true).await;
                continue;
            }
        };
        let Some(m) = m else { break };
        match m.msg {
            Some(agent_message::Msg::Progress(p)) => {
                let eta_s = eta.update(p.step, p.total_steps);
                if predicted_total.is_none()
                    && let (Some(e), Some(total)) = (eta_s, p.total_steps)
                    && total > 0
                    && p.step * 10 >= total
                {
                    predicted_total = Some(started.elapsed().as_secs_f64() + e);
                }
                events::emit(
                    job_id,
                    JobEvent::Progress {
                        run_id: run_id.to_owned(),
                        step: p.step,
                        total_steps: p.total_steps,
                        eta_s,
                        job_eta_s: eta_s.map(|e| e + rest_s.unwrap_or(0.0)),
                    },
                );
            }
            Some(agent_message::Msg::Metrics(b)) => {
                if record {
                    for p in &b.metrics {
                        events::emit(
                            job_id,
                            JobEvent::Metric {
                                run_id: run_id.to_owned(),
                                name: p.name.clone(),
                                step: p.step.map(|s| s as i64),
                                value: p.value,
                            },
                        );
                    }
                }
                for p in &b.metrics {
                    if let Some(k) = p.name.strip_prefix("eval/")
                        && p.value.is_finite()
                    {
                        evals.insert(k.to_owned(), p.value);
                    }
                }
                points.extend(b.metrics.into_iter().map(|p| MetricPoint {
                    name: p.name,
                    step: p.step.map(|s| s as i64),
                    value: p.value,
                    unix_ms: p.unix_ms,
                }));
                if record && points.len() >= 500 {
                    tracker.add_metrics(run_id, &std::mem::take(&mut points)).await?;
                }
            }
            Some(agent_message::Msg::Logs(l)) => {
                for line in l.lines {
                    let level = match tempera_proto::v1::LogLevel::try_from(line.level) {
                        Ok(tempera_proto::v1::LogLevel::Warn) => "warn",
                        Ok(tempera_proto::v1::LogLevel::Error) => "error",
                        Ok(tempera_proto::v1::LogLevel::Debug) => "debug",
                        Ok(tempera_proto::v1::LogLevel::Raw) => "raw",
                        _ => "info",
                    };
                    events::emit(
                        job_id,
                        JobEvent::Log {
                            run_id: run_id.to_owned(),
                            level,
                            message: line.message.clone(),
                        },
                    );
                    ui::backend_log(line.level, &line.message);
                }
            }
            Some(agent_message::Msg::Finished(mut f)) => {
                if let (Some(r), Some(p)) = (&remote, &plan) {
                    r.pull(p).await?;
                    for a in &mut f.artifacts {
                        if let Some(rest) = a.uri.strip_prefix(&r.remote_job_dir) {
                            a.uri = format!("{}{rest}", r.job_dir.display());
                        }
                    }
                }
                if record {
                    if let Some(pred) = predicted_total {
                        let actual = started.elapsed().as_secs_f64().max(1e-6);
                        points.push(MetricPoint {
                            name: "eta/relative_error".into(),
                            step: None,
                            value: (pred - actual) / actual,
                            unix_ms: 0,
                        });
                    }
                    tracker.add_metrics(run_id, &points).await?;
                    for a in f.artifacts {
                        tracker.add_artifact(run_id, &a.kind, &a.format, &a.uri).await?;
                    }
                }
                return Ok(evals);
            }
            Some(agent_message::Msg::Failed(f)) => {
                if record {
                    tracker.add_metrics(run_id, &points).await?;
                }
                let tail = f
                    .stderr_tail
                    .lines()
                    .rev()
                    .take(3)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join(" | ");
                bail!(
                    "{}{}",
                    f.reason,
                    if tail.is_empty() {
                        String::new()
                    } else {
                        format!(" ({tail})")
                    }
                );
            }
            _ => {}
        }
    }
    bail!("agent stream ended unexpectedly")
}

fn fit_manifest(
    cand: &Candidate,
    spec: &JobSpec,
    split: &Split,
    task: TaskType,
    work: &Path,
    valid_pred: &Path,
) -> Manifest {
    // Classes go only to classifiers; anomaly labels are for evaluation (§8).
    let classification = matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    );
    Manifest {
        algorithm: cand.algorithm.clone(),
        task,
        classes: classification.then(|| split.classes.clone()),
        positive_class: split.positive_class.clone(),
        features: split
            .features
            .iter()
            .map(|f| FeatureSpec {
                name: f.name.clone(),
                kind: match f.kind {
                    FeatureKind::Numeric => NFeatureKind::Numeric,
                    FeatureKind::Categorical => NFeatureKind::Categorical,
                    FeatureKind::Boolean => NFeatureKind::Boolean,
                },
            })
            .collect(),
        params: serde_json::Map::new(),
        seed: spec.split.seed,
        train: abs(&split.path(Partition::Train)),
        valid: Some(abs(&split.path(Partition::Valid))),
        predict: vec![PredictTarget {
            input: abs(&split.path(Partition::Valid)),
            output: abs(valid_pred),
        }],
        work_dir: abs(work),
        export: {
            // The native backend only has its own format (no ONNX yet, see #15).
            let mut v = if cand.backend == BackendKind::Native {
                Vec::new()
            } else {
                spec.output.formats.clone()
            };
            if !v.contains(&ExportFormat::Native) {
                v.push(ExportFormat::Native);
            }
            v
        },
    }
}

/// Evaluates the winner once on the test set (claimed in the tracker first).
#[allow(clippy::too_many_arguments)]
async fn evaluate_winner(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    job_dir: &Path,
    best: &RunSummary,
    cand: &Candidate,
    split: &Split,
    task: TaskType,
) -> anyhow::Result<Winner> {
    tracker.claim_test_set(job_id, &best.run_id).await?;
    let work = job_dir.join("runs").join(&best.run_id);
    let test_pred = work.join("test_pred.parquet");
    let test = split.path(Partition::Test);
    let mut test_evals = BTreeMap::new();
    match cand.backend {
        BackendKind::Builtin => {
            let (train, classes, t, out, algo) = (
                split.path(Partition::Train),
                split.classes.clone(),
                test.clone(),
                test_pred.clone(),
                cand.algorithm.clone(),
            );
            tokio::task::spawn_blocking(move || match algo.as_str() {
                "majority" => builtin::majority(&train, &classes, &t, &out),
                _ => builtin::mean(&train, &t, &out),
            })
            .await??;
        }
        _ => {
            let pm = PredictManifest {
                algorithm: cand.algorithm.clone(),
                task,
                model_dir: abs(&work.join("model")),
                input: abs(&test),
                output: abs(&test_pred),
            };
            test_evals = dispatch(
                tracker,
                hub,
                job_id,
                &format!("{}{TEST_RUN_SUFFIX}", best.run_id),
                cand.backend,
                JobCommand::Predict,
                serde_json::to_string(&pm)?,
                false,
                None,
            )
            .await
            .context("predicting the test set with the winner")?;
        }
    }
    let mut eval: Evaluation = evaluate(
        task,
        &split.classes,
        split.positive_class.as_deref(),
        &test,
        &test_pred,
    )?;
    eval.metrics.extend(test_evals);
    let mut importance: Vec<(String, f64)> = tracker
        .metrics(&best.run_id, None)
        .await?
        .into_iter()
        .filter_map(|p| {
            p.name
                .strip_prefix("importance/")
                .map(|n| (n.to_owned(), p.value))
        })
        .collect();
    importance.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    let artifacts = tracker
        .artifacts(&best.run_id)
        .await?
        .into_iter()
        .filter(|(kind, _, _)| kind == "model")
        .map(|(_, format, path)| (format, path))
        .collect();
    Ok(Winner {
        run_id: best.run_id.clone(),
        algorithm: best.algorithm.clone(),
        test: eval,
        importance,
        artifacts,
    })
}

fn abs(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

pub(crate) fn new_job_id(name: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stamp = tempera_track::now()
        .replace(['-', ':'], "")
        .replace('T', "-")
        .trim_end_matches('Z')
        .to_owned();
    let suffix = (nanos ^ u128::from(std::process::id())) % 0x10000;
    format!("{name}-{stamp}-{suffix:04x}")
}

/// The agent running this job's runs (one per job, so its size does not matter).
/// Starts the agent on an SSH machine (LAN or rented) and registers the remote mirror.
async fn start_ssh(
    hub: &Hub,
    addr: std::net::SocketAddr,
    ssh: tempera_exec::SshMachine,
    label: String,
    worker_dir: Option<PathBuf>,
    job_id: &str,
    job_dir: &Path,
) -> anyhow::Result<Agent> {
    let opts = tempera_exec::ssh::StartOpts {
        worker_dir,
        ..Default::default()
    };
    let session = ssh.start(hub, addr, &opts).await?;
    let remote_job_dir = format!("{}/jobs/{job_id}", ssh.remote_root().await?);
    crate::remote::set(
        job_id,
        crate::remote::Remote {
            ssh,
            machine_id: session.machine_id.clone(),
            job_dir: job_dir.to_path_buf(),
            remote_job_dir,
        },
    );
    Ok(Agent::Ssh(label, Box::new(session)))
}

#[allow(clippy::large_enum_variant)]
enum Agent {
    Local(LocalExecutor),
    Ssh(String, Box<tempera_exec::ssh::SshSession>),
}

impl Agent {
    fn hardware(&self) -> &tempera_proto::v1::Hardware {
        match self {
            Self::Local(e) => &e.hardware,
            Self::Ssh(_, s) => &s.hardware,
        }
    }

    fn describe(&self) -> String {
        let hw = self.hardware();
        let gpus = if hw.gpus.is_empty() {
            "no GPU".to_owned()
        } else {
            hw.gpus
                .iter()
                .map(|g| g.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let name = match self {
            Self::Local(_) => "local".to_owned(),
            Self::Ssh(n, _) => n.clone(),
        };
        format!("{name} · {} CPU · {gpus}", hw.cpu_cores)
    }

    async fn stop(self, hub: &Hub) {
        match self {
            Self::Local(e) => e.stop(hub).await,
            Self::Ssh(_, s) => (*s).stop(hub).await,
        }
    }
}

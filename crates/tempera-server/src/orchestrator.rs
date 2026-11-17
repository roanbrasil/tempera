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
    let mut spec = JobSpec::from_yaml(&text).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    // Data paths are relative to the spec file.
    let source = Path::new(&spec.data.source);
    if source.is_relative() && !spec.data.source.contains("://") {
        let base = path.parent().unwrap_or(Path::new("."));
        spec.data.source = base.join(source).display().to_string();
    }
    Ok((spec, text))
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
    let audit = tempera_audit::audit(&ds, &spec, &profile, &inference)?;
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
    let started = Instant::now();
    let (spec, spec_text) = load_spec(spec_path)?;
    let tracker = Tracker::open(&ws.db_path()).await?;
    let job_id = new_job_id(&spec.name);
    tracker.create_job(&job_id, &spec.name, &spec_text).await?;
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
        audit,
    } = prepared;
    print_audit(&audit);
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
        return Ok(JobOutcome {
            job_id,
            blocked: true,
            report_path: None,
        });
    }

    let (spec2, ds2, plan, task) = (spec.clone(), ds.clone(), audit.plan.clone(), inference.task);
    let split = tokio::task::spawn_blocking(move || make_split(&ds2, &spec2, task, &plan)).await??;
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
        ws, &tracker, &job_id, &job_dir, &spec, &spec_text, spec_path, &ds, &inference, &audit, &split,
        started,
    )
    .await;
    match &result {
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
    spec_path: &Path,
    ds: &Dataset,
    inference: &tempera_data::TaskInference,
    audit: &AuditReport,
    split: &Split,
    started: Instant,
) -> anyhow::Result<PathBuf> {
    let task = inference.task;
    let metric = inference.metric;
    let cands = candidates(spec, task).map_err(|e| anyhow!("strategy.candidates: {e}"))?;
    let worker = worker_dir();
    let python = python_available(worker.as_deref());

    // Bring up the local agent only if something needs it.
    let needs_agent = cands.iter().any(|c| c.backend != BackendKind::Builtin);
    let hub = Hub::new();
    let mut agent = None;
    if needs_agent {
        let (addr, _server) = hub.serve("127.0.0.1:0".parse()?).await?;
        let exec = LocalExecutor::start(&hub, &format!("http://{addr}"), worker.as_deref()).await?;
        ui::step(&format!(
            "Machine   local · {} CPU · {}",
            exec.hardware.cpu_cores,
            if exec.hardware.gpus.is_empty() {
                "no GPU".to_owned()
            } else {
                exec.hardware
                    .gpus
                    .iter()
                    .map(|g| g.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        ));
        agent = Some(exec);
    }
    let machine = agent.as_ref().map_or_else(
        || "local".to_owned(),
        |a| {
            format!(
                "local ({} CPU{})",
                a.hardware.cpu_cores,
                if a.hardware.gpus.is_empty() {
                    ", no GPU"
                } else {
                    ""
                }
            )
        },
    );

    ui::step(&format!(
        "Training  {} candidates · ranked by {} on validation",
        cands.len(),
        metric_name(metric)
    ));
    let mut summaries = Vec::new();
    for (i, cand) in cands.iter().enumerate() {
        let run_id = format!("{job_id}-r{}", i + 1);
        let summary = if cand.backend == BackendKind::Python {
            if let Err(why) = &python {
                skipped(tracker, job_id, &run_id, cand, why).await?
            } else {
                run_candidate(
                    tracker, &hub, job_id, &run_id, job_dir, cand, spec, split, task, metric,
                )
                .await?
            }
        } else {
            run_candidate(
                tracker, &hub, job_id, &run_id, job_dir, cand, spec, split, task, metric,
            )
            .await?
        };
        ui::run_line(&summary, metric);
        summaries.push(summary);
    }

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

    let winner = match summaries
        .first()
        .filter(|s| s.status == "succeeded" && s.valid_score.is_some())
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
    if let Some(exec) = agent {
        exec.stop(&hub).await;
    }
    if let Some(w) = &winner {
        ui::verdict(w, metric, &summaries);
    }

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
            cost_usd: 0.0,
        },
        spec_yaml: spec_text.to_owned(),
        reproduce_command: format!("tempera run {}", spec_path.display()),
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
            &serde_json::json!({}),
        )
        .await?;

    let outcome: anyhow::Result<Option<String>> = match cand.backend {
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
            .map(|()| None)
        }
        BackendKind::Native | BackendKind::Python => {
            let manifest = fit_manifest(cand, spec, split, task, &work, &valid_pred);
            dispatch(
                tracker,
                hub,
                job_id,
                run_id,
                cand.backend,
                JobCommand::Fit,
                serde_json::to_string(&manifest)?,
                true,
            )
            .await
            .map(|()| Some(work.join("model").display().to_string()))
        }
    };
    let duration = t.elapsed().as_secs_f64();
    match outcome {
        Ok(model_dir) => {
            let e = evaluate(
                task,
                &split.classes,
                split.positive_class.as_deref(),
                &split.path(Partition::Valid),
                &valid_pred,
            )?;
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

/// Sends one job to the local agent and drains its messages into the tracker.
#[allow(clippy::too_many_arguments)]
async fn dispatch(
    tracker: &Tracker,
    hub: &Hub,
    job_id: &str,
    run_id: &str,
    backend: BackendKind,
    command: JobCommand,
    manifest_json: String,
    record: bool,
) -> anyhow::Result<()> {
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
    let mut rx = hub.start(LOCAL_MACHINE, start).await?;
    let mut points = Vec::new();
    while let Some(m) = rx.recv().await {
        match m.msg {
            Some(agent_message::Msg::Metrics(b)) => {
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
                    ui::backend_log(line.level, &line.message);
                }
            }
            Some(agent_message::Msg::Finished(f)) => {
                if record {
                    tracker.add_metrics(run_id, &points).await?;
                    for a in f.artifacts {
                        tracker.add_artifact(run_id, &a.kind, &a.format, &a.uri).await?;
                    }
                }
                return Ok(());
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
    let classification = !split.classes.is_empty();
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
            dispatch(
                tracker,
                hub,
                job_id,
                &format!("{}-test", best.run_id),
                cand.backend,
                JobCommand::Predict,
                serde_json::to_string(&pm)?,
                false,
            )
            .await
            .context("predicting the test set with the winner")?;
        }
    }
    let eval: Evaluation = evaluate(
        task,
        &split.classes,
        split.positive_class.as_deref(),
        &test,
        &test_pred,
    )?;
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

fn new_job_id(name: &str) -> String {
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

//! `tempera up`: HTTP API + WebSocket live events + embedded web UI
//! (docs/wiki/Contracts.md §9).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Multipart, Path as UrlPath, Query, State};
use axum::http::{StatusCode, Uri, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tempera_core::{JobSpec, SpecError};
use tempera_track::Tracker;

use crate::events::{self, JobEvent};
use crate::workspace::Workspace;
use crate::{candidates, draft, orchestrator};

#[derive(rust_embed::Embed)]
#[folder = "$CARGO_MANIFEST_DIR/../../ui/build"]
#[allow_missing = true]
struct UiAssets;

#[derive(Clone)]
struct AppState {
    ws: Workspace,
    tracker: Tracker,
}

struct ApiError(StatusCode, Value);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "error": format!("{e:#}") }),
        )
    }
}

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, json!({ "error": msg.into() }))
}

fn not_found(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, json!({ "error": msg.into() }))
}

type ApiResult<T> = Result<T, ApiError>;

pub fn router(ws: Workspace, tracker: Tracker) -> Router {
    let api = Router::new()
        .route("/health", get(health))
        .route("/datasets", post(upload_dataset))
        .route("/datasets/{id}", get(get_dataset))
        .route("/specs/validate", post(validate_spec))
        .route("/specs/draft", post(draft_spec))
        .route("/audit", post(audit))
        .route("/algorithms", get(algorithms))
        .route("/machines", get(machines))
        .route("/jobs", get(list_jobs).post(create_job))
        .route("/jobs/{id}", get(get_job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/jobs/{id}/report", get(job_report))
        .route("/jobs/{id}/events", get(job_events))
        .route("/runs/{id}/metrics", get(run_metrics))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024 * 1024));
    Router::new()
        .nest("/api/v1", api)
        .fallback(static_ui)
        .with_state(AppState { ws, tracker })
}

pub async fn serve(ws: Workspace, addr: SocketAddr) -> anyhow::Result<()> {
    std::fs::create_dir_all(&ws.root)?;
    let tracker = Tracker::open(&ws.db_path()).await?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?;
    println!("\n  têmpera is running at http://{bound}  (API: http://{bound}/api/v1)\n  Ctrl+C to stop.\n");
    if UiAssets::get("index.html").is_none() {
        println!(
            "  note: the web UI was not built into this binary (cd ui && npm run build, then rebuild).\n"
        );
    }
    axum::serve(listener, router(ws, tracker))
        .with_graceful_shutdown(shutdown())
        .await?;
    Ok(())
}

async fn shutdown() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true, "version": env!("CARGO_PKG_VERSION") }))
}

// ---------------------------------------------------------------- datasets

async fn upload_dataset(State(st): State<AppState>, mut mp: Multipart) -> ApiResult<Json<Value>> {
    let uploads = st.ws.root.join("uploads");
    std::fs::create_dir_all(&uploads)?;
    while let Some(mut field) = mp.next_field().await.map_err(|e| bad_request(e.to_string()))? {
        if field.name() != Some("file") {
            continue;
        }
        let name = sanitize(field.file_name().unwrap_or("upload.csv"));
        let tmp = uploads.join(format!(".{}-{name}", std::process::id()));
        {
            use tokio::io::AsyncWriteExt;
            let mut f = tokio::fs::File::create(&tmp).await?;
            while let Some(chunk) = field.chunk().await.map_err(|e| bad_request(e.to_string()))? {
                f.write_all(&chunk).await?;
            }
            f.flush().await?;
        }
        let ws = st.ws.clone();
        let (ds, profile) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            // Name the stored file after its content hash so re-uploads are free.
            let hash = tempera_data::ingest::hash_file(&tmp)?;
            let dest = uploads.join(format!("{}-{name}", &hash[..12]));
            if dest.exists() {
                std::fs::remove_file(&tmp)?;
            } else {
                std::fs::rename(&tmp, &dest)?;
            }
            let dest = std::path::absolute(&dest)?;
            let ds = tempera_data::ingest(&dest, &ws.root)?;
            let profile = tempera_data::profile_dataset(&ds)?;
            Ok((ds, profile))
        })
        .await?
        .map_err(|e| bad_request(format!("{e:#}")))?;
        return Ok(Json(json!({ "dataset": ds, "profile": profile })));
    }
    Err(bad_request("multipart field `file` is required"))
}

async fn get_dataset(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> ApiResult<Json<Value>> {
    if !id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad_request("invalid dataset id"));
    }
    let dir = st.ws.root.join("datasets").join(&id);
    let (ds, profile) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let ds = tempera_data::Dataset::load(&dir)?;
        let profile = tempera_data::profile_dataset(&ds)?;
        Ok((ds, profile))
    })
    .await?
    .map_err(|_| not_found(format!("dataset {id} not found")))?;
    Ok(Json(json!({ "dataset": ds, "profile": profile })))
}

fn sanitize(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("upload.csv");
    base.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ---------------------------------------------------------------- specs & audit

#[derive(Deserialize)]
struct YamlBody {
    yaml: String,
}

async fn validate_spec(Json(b): Json<YamlBody>) -> Json<Value> {
    let issues = match JobSpec::from_yaml(&b.yaml) {
        Ok(_) => Vec::new(),
        Err(SpecError::Invalid(issues)) => issues
            .into_iter()
            .map(|i| json!({ "path": i.path, "message": i.message }))
            .collect(),
        Err(SpecError::Parse { path, message }) => vec![json!({ "path": path, "message": message })],
        Err(e) => vec![json!({ "path": "", "message": e.to_string() })],
    };
    Json(json!({ "valid": issues.is_empty(), "issues": issues }))
}

#[derive(Deserialize)]
struct DraftBody {
    dataset_id: String,
    target: Option<String>,
    name: Option<String>,
}

async fn draft_spec(State(st): State<AppState>, Json(b): Json<DraftBody>) -> ApiResult<Json<Value>> {
    if !b.dataset_id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad_request("invalid dataset id"));
    }
    let dir = st.ws.root.join("datasets").join(&b.dataset_id);
    let yaml = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
        let ds = tempera_data::Dataset::load(&dir)?;
        let profile = tempera_data::profile_dataset(&ds)?;
        let name = b.name.unwrap_or_else(|| {
            ds.source
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("job")
                .trim_start_matches(|c: char| c.is_ascii_hexdigit() || c == '-')
                .to_owned()
        });
        let name = if name.is_empty() { "job".to_owned() } else { name };
        draft::draft_yaml(
            &profile,
            &ds.source.display().to_string(),
            b.target.as_deref(),
            &name,
        )
    })
    .await?
    .map_err(|e| bad_request(format!("{e:#}")))?;
    Ok(Json(json!({ "yaml": yaml })))
}

/// Specs from the API resolve relative data paths against the server's working directory.
fn parse(yaml: &str) -> ApiResult<JobSpec> {
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    orchestrator::parse_spec(yaml, &base).map_err(|e| bad_request(e.to_string()))
}

async fn audit(State(st): State<AppState>, Json(b): Json<YamlBody>) -> ApiResult<Json<Value>> {
    let spec = parse(&b.yaml)?;
    let ws = st.ws.clone();
    let p = tokio::task::spawn_blocking(move || orchestrator::prepare(&ws, spec))
        .await?
        .map_err(|e| bad_request(format!("{e:#}")))?;
    Ok(Json(json!({ "inference": p.inference, "audit": p.audit })))
}

async fn algorithms() -> Json<Value> {
    Json(json!(candidates::catalog()))
}

async fn machines() -> Json<Value> {
    let hw = tokio::task::spawn_blocking(tempera_exec::system::hardware)
        .await
        .unwrap_or_default();
    Json(json!([{
        "id": "local",
        "kind": "local",
        "status": "ready",
        "hardware": {
            "os": hw.os, "arch": hw.arch, "cpu_cores": hw.cpu_cores, "ram_bytes": hw.ram_bytes,
            "python_available": hw.python_available,
            "gpus": hw.gpus.iter().map(|g| json!({ "name": g.name, "vram_bytes": g.vram_bytes })).collect::<Vec<_>>(),
        }
    }]))
}

// ---------------------------------------------------------------- jobs

async fn create_job(
    State(st): State<AppState>,
    Json(b): Json<YamlBody>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let spec = parse(&b.yaml)?;
    let job_id = orchestrator::generate_job_id(&spec.name);
    events::register(&job_id);
    let (ws, id) = (st.ws.clone(), job_id.clone());
    tokio::spawn(async move {
        if let Err(e) = orchestrator::run_spec(&ws, spec, b.yaml, "tempera.yaml", Some(id.clone())).await {
            tracing::warn!(job = %id, error = %e, "job failed");
        }
    });
    Ok((StatusCode::ACCEPTED, Json(json!({ "job_id": job_id }))))
}

async fn list_jobs(State(st): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(st.tracker.jobs(200).await?)))
}

async fn get_job(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> ApiResult<Json<Value>> {
    let job = st.tracker.job(&id).await.map_err(|e| not_found(e.to_string()))?;
    let runs = st.tracker.runs(&job.id).await?;
    Ok(Json(json!({ "job": job, "runs": runs })))
}

async fn cancel_job(UrlPath(id): UrlPath<String>) -> ApiResult<(StatusCode, Json<Value>)> {
    if events::request_cancel(&id) {
        Ok((StatusCode::ACCEPTED, Json(json!({}))))
    } else {
        Err(ApiError(
            StatusCode::CONFLICT,
            json!({ "error": "job is not running in this server" }),
        ))
    }
}

async fn job_report(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> ApiResult<Response> {
    let job = st.tracker.job(&id).await.map_err(|e| not_found(e.to_string()))?;
    let path = job
        .report_path
        .ok_or_else(|| not_found("this job has no report"))?;
    let html = tokio::fs::read_to_string(&path)
        .await
        .map_err(|_| not_found("report file is missing"))?;
    Ok(Html(html).into_response())
}

#[derive(Deserialize)]
struct MetricsQuery {
    name: Option<String>,
}

async fn run_metrics(
    State(st): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<MetricsQuery>,
) -> ApiResult<Json<Value>> {
    Ok(Json(json!(st.tracker.metrics(&id, q.name.as_deref()).await?)))
}

async fn job_events(
    State(st): State<AppState>,
    UrlPath(id): UrlPath<String>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| stream_events(socket, st, id))
}

async fn stream_events(mut socket: WebSocket, st: AppState, id: String) {
    let send = |ev: &JobEvent| serde_json::to_string(ev).map(|t| Message::Text(t.into()));
    match events::subscribe(&id) {
        Some((history, mut rx)) => {
            let mut finished = false;
            for ev in &history {
                finished |= ev.is_terminal();
                if let Ok(m) = send(ev)
                    && socket.send(m).await.is_err()
                {
                    return;
                }
            }
            while !finished {
                match rx.recv().await {
                    Ok(ev) => {
                        finished = ev.is_terminal();
                        if let Ok(m) = send(&ev)
                            && socket.send(m).await.is_err()
                        {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                }
            }
        }
        None => {
            // Not run by this server process (e.g. CLI job or before a restart): summarize from the DB.
            let status = st
                .tracker
                .job(&id)
                .await
                .map(|j| j.status)
                .unwrap_or_else(|_| "unknown".into());
            let ev = json!({
                "type": "job_finished",
                "status": status,
                "report_url": format!("/api/v1/jobs/{id}/report"),
                "error": null,
            });
            let _ = socket.send(Message::Text(ev.to_string().into())).await;
        }
    }
    let _ = socket.send(Message::Close(None)).await;
}

// ---------------------------------------------------------------- static UI

async fn static_ui(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(f) = UiAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        return (
            [(header::CONTENT_TYPE, mime.as_ref().to_owned())],
            f.data.into_owned(),
        )
            .into_response();
    }
    // SPA fallback: client-side routes render index.html.
    match UiAssets::get("index.html") {
        Some(f) => Html(f.data.into_owned()).into_response(),
        None => Html(
            "<!doctype html><meta charset=utf-8><title>têmpera</title><body style=\"font-family:system-ui;padding:2rem\"><h1>têmpera</h1><p>The web UI is not built into this binary. The API is available under <code>/api/v1</code>.</p>",
        )
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn app() -> (tempfile::TempDir, Router) {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace {
            root: tmp.path().to_path_buf(),
        };
        let tracker = Tracker::open(&ws.db_path()).await.unwrap();
        (tmp, router(ws, tracker))
    }

    async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    fn post_json(uri: &str, body: Value) -> Request<Body> {
        Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn health_validate_algorithms() {
        let (_t, app) = app().await;
        let (s, v) = call(&app, Request::get("/api/v1/health").body(Body::empty()).unwrap()).await;
        assert_eq!((s, v["ok"].as_bool()), (StatusCode::OK, Some(true)));
        let (_, v) = call(
            &app,
            post_json(
                "/api/v1/specs/validate",
                json!({"yaml": "version: 1\nname: x\ndata: {source: a.csv}\nsplit: {test_size: 0.9}\n"}),
            ),
        )
        .await;
        assert_eq!(v["valid"], false);
        assert_eq!(v["issues"][0]["path"], "split.test_size");
        let (_, v) = call(
            &app,
            Request::get("/api/v1/algorithms").body(Body::empty()).unwrap(),
        )
        .await;
        assert!(v.as_array().unwrap().iter().any(|a| a["key"] == "lightgbm"));
        let (s, _) = call(
            &app,
            Request::get("/api/v1/jobs/nope").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upload_draft_and_audit() {
        let (_t, app) = app().await;
        let csv = "a,b,y\n1,x,yes\n2,y,no\n3,x,yes\n4,y,no\n5,x,yes\n6,y,no\n7,x,yes\n8,y,no\n9,x,yes\n10,y,no\n11,x,no\n12,y,yes\n";
        let boundary = "XBOUNDARYX";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"small data.csv\"\r\nContent-Type: text/csv\r\n\r\n{csv}\r\n--{boundary}--\r\n"
        );
        let req = Request::post("/api/v1/datasets")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap();
        let (s, v) = call(&app, req).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["dataset"]["rows"], 12);
        let id = v["dataset"]["id"].as_str().unwrap().to_owned();

        let (s, v) = call(
            &app,
            Request::get(format!("/api/v1/datasets/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["profile"]["columns"].as_array().unwrap().len(), 3);

        let (s, v) = call(
            &app,
            post_json("/api/v1/specs/draft", json!({"dataset_id": id, "target": "y"})),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let yaml = v["yaml"].as_str().unwrap().to_owned();
        assert!(yaml.contains("target: y"));

        let (s, v) = call(&app, post_json("/api/v1/audit", json!({"yaml": yaml}))).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["inference"]["task"], "binary_classification");
        assert!(v["audit"]["findings"].is_array());
    }

    #[test]
    fn sanitizes_upload_names() {
        assert_eq!(sanitize("../../etc/passwd"), "passwd");
        assert_eq!(sanitize("my data (1).csv"), "my_data__1_.csv");
    }
}

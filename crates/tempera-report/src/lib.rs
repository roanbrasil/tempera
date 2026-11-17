//! Self-contained HTML report for a finished job (docs/wiki/User-Journey.md, step 7).
//!
//! The orchestrator fills a [`ReportData`]; [`render`] turns it into one HTML
//! file with inline CSS and SVG charts: no network, no JavaScript required.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use tempera_audit::AuditReport;
use tempera_core::spec::{Metric, TaskType};
use tempera_data::eval::Evaluation;
use tempera_data::split::Split;

mod format;
mod html;
mod svg;
#[cfg(test)]
mod tests;

/// Everything the report shows. Serializable so it can also be stored as JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReportData {
    pub job_id: String,
    pub job_name: String,
    /// RFC 3339 timestamp.
    pub created_at: String,
    pub tempera_version: String,
    pub dataset: DatasetSummary,
    pub task: TaskType,
    pub metric: Metric,
    /// Why this task/metric (from inference).
    pub task_reasons: Vec<String>,
    pub audit: AuditReport,
    pub split: Split,
    /// All candidates, best first by the main validation metric.
    pub leaderboard: Vec<RunSummary>,
    pub winner: Option<Winner>,
    pub cost: CostSummary,
    /// The exact `tempera.yaml` used.
    pub spec_yaml: String,
    /// Command line to reproduce the job.
    pub reproduce_command: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetSummary {
    pub id: String,
    pub source: String,
    pub rows: u64,
    pub columns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub algorithm: String,
    /// `native`, `python` or `builtin` (trivial baselines).
    pub backend: String,
    pub is_baseline: bool,
    /// `succeeded` | `failed` | `skipped`.
    pub status: String,
    /// Main metric on the validation set.
    pub valid_score: Option<f64>,
    pub valid_metrics: BTreeMap<String, f64>,
    pub duration_s: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Winner {
    pub run_id: String,
    pub algorithm: String,
    /// Evaluated once, on the held-out test set.
    pub test: Evaluation,
    /// Feature → importance (any scale), largest first.
    pub importance: Vec<(String, f64)>,
    /// Exported model files: (format, path).
    pub artifacts: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostSummary {
    pub wall_time_s: f64,
    /// Where it ran, e.g. `local (8 CPU, no GPU)`.
    pub machine: String,
    pub cost_usd: f64,
}

/// Renders the full HTML document: one self-contained file with inline CSS
/// and SVG charts (no JavaScript, no network). Never panics on NaN or empty data.
pub fn render(data: &ReportData) -> String {
    html::document(data)
}

/// Minimal HTML escaping for text and attribute values.
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

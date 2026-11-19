//! The SLM advisor (#34, ADR-0004, ADR-0008): a small local language model
//! reads column names and sample values and points at things statistics
//! cannot see (a column only known *after* the outcome, a sensitive
//! attribute, personal data). It only **suggests**: its findings never block
//! training and never change the data.
//!
//! Talks to any OpenAI-compatible endpoint (Ollama, llama.cpp server,
//! mistral.rs server). Remote endpoints require `audit.advisor.remote: true`.

use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tempera_core::spec::AdvisorSpec;
use tempera_data::profile::{ColumnKind, Profile};

use crate::{Finding, Resolution, Severity};

pub const DEFAULT_URL: &str = "http://127.0.0.1:11434/v1";
const MIN_CONFIDENCE: f64 = 0.6;
const MAX_COLUMNS: usize = 60;

/// Column roles the model may assign (a closed set, enforced by the schema).
const ROLES: [&str; 7] = [
    "feature",
    "identifier",
    "timestamp",
    "post_outcome",
    "sensitive",
    "personal_data",
    "free_text",
];

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ColumnAdvice {
    pub column: String,
    pub role: String,
    pub confidence: f64,
    pub reason: String,
}

#[derive(Debug, Deserialize)]
struct Advice {
    columns: Vec<ColumnAdvice>,
}

/// Outcome of asking the advisor.
#[derive(Debug, Clone, PartialEq)]
pub enum AdvisorResult {
    Advice {
        model: String,
        columns: Vec<ColumnAdvice>,
    },
    /// Disabled, unreachable, refused for privacy, or invalid output: the audit goes on without it.
    Skipped(String),
}

fn is_local(url: &str) -> bool {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let host = rest.split(['/', ':']).next().unwrap_or("");
    let host = if rest.starts_with('[') {
        rest.split(']').next().unwrap_or("").trim_start_matches('[')
    } else {
        host
    };
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .timeout_global(Some(timeout))
            .build(),
    )
}

/// Asks the model about the columns of `profile`.
pub fn advise(spec: &AdvisorSpec, profile: &Profile, target: Option<&str>) -> AdvisorResult {
    if !spec.enabled || std::env::var("TEMPERA_ADVISOR").is_ok_and(|v| v == "off") {
        return AdvisorResult::Skipped("advisor disabled".into());
    }
    let url = std::env::var("TEMPERA_LLM_URL").unwrap_or_else(|_| DEFAULT_URL.to_owned());
    advise_at(&url, spec, profile, target)
}

/// [`advise`] against an explicit endpoint.
pub fn advise_at(url: &str, spec: &AdvisorSpec, profile: &Profile, target: Option<&str>) -> AdvisorResult {
    if !spec.remote && !is_local(url) {
        return AdvisorResult::Skipped(format!(
            "{url} is not local; set audit.advisor.remote: true to send column names and samples there"
        ));
    }
    let base = url.trim_end_matches('/');
    let Some(model) = resolve_model(base, &spec.model) else {
        return AdvisorResult::Skipped(format!("no model available at {base} (wanted `{}`)", spec.model));
    };
    match ask(base, &model, profile, target) {
        Ok(columns) => AdvisorResult::Advice { model, columns },
        Err(e) => AdvisorResult::Skipped(e),
    }
}

/// The configured model if the server has it; otherwise a Qwen3 model it has.
fn resolve_model(base: &str, wanted: &str) -> Option<String> {
    let v: Value = agent(Duration::from_secs(3))
        .get(&format!("{base}/models"))
        .call()
        .ok()?
        .into_body()
        .read_json()
        .ok()?;
    let ids: Vec<String> = v["data"]
        .as_array()?
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_owned))
        .collect();
    if ids.iter().any(|i| i == wanted) {
        return Some(wanted.to_owned());
    }
    let family = wanted.split([':', '-']).next().unwrap_or("qwen3");
    ids.iter()
        .find(|i| i.starts_with(family))
        .or_else(|| ids.iter().find(|i| i.contains("qwen3")))
        .cloned()
}

fn prompt(profile: &Profile, target: Option<&str>) -> String {
    let mut lines = Vec::new();
    for c in profile.columns.iter().take(MAX_COLUMNS) {
        let samples: Vec<String> = c
            .top_values
            .iter()
            .take(4)
            .map(|v| format!("{:?}", truncate(&v.value, 40)))
            .collect();
        let range = c
            .numeric
            .as_ref()
            .map(|n| format!(" range {:.4}..{:.4}", n.min, n.max))
            .unwrap_or_default();
        lines.push(format!(
            "- {} ({}, {} distinct, {:.0}% null{range}){}",
            c.name,
            c.dtype,
            c.n_unique,
            c.null_fraction * 100.0,
            if samples.is_empty() {
                String::new()
            } else {
                format!(" e.g. {}", samples.join(", "))
            }
        ));
    }
    format!(
        "A dataset will be used to train a model that predicts `{}`.\n\
         Columns:\n{}\n\n\
         For EVERY column except the target, give its role:\n\
         - post_outcome: only known AFTER the outcome happens (it would leak the answer), e.g. refund_amount when predicting churn\n\
         - identifier: a row/entity id with no meaning\n\
         - timestamp: a date/time\n\
         - sensitive: a protected attribute (gender, race, ethnicity, religion, age, disability, nationality)\n\
         - personal_data: names, e-mails, phone numbers, documents, addresses\n\
         - free_text: long free text\n\
         - feature: anything else (normal predictive input)\n\
         Be conservative: use `feature` unless the name or values make the role clear. \
         Give a confidence between 0 and 1 and a one-sentence reason.",
        target.unwrap_or("(no target: clustering)"),
        lines.join("\n")
    )
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

fn ask(
    base: &str,
    model: &str,
    profile: &Profile,
    target: Option<&str>,
) -> Result<Vec<ColumnAdvice>, String> {
    let schema = json!({
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": { "type": "string" },
                        "role": { "type": "string", "enum": ROLES },
                        "confidence": { "type": "number" },
                        "reason": { "type": "string" }
                    },
                    "required": ["column", "role", "confidence", "reason"]
                }
            }
        },
        "required": ["columns"]
    });
    let body = json!({
        "model": model,
        "temperature": 0,
        "messages": [
            { "role": "system", "content": "You are a careful data auditor. Answer only with JSON that matches the schema." },
            { "role": "user", "content": prompt(profile, target) }
        ],
        "response_format": { "type": "json_schema", "json_schema": { "name": "column_roles", "schema": schema } }
    });
    let v: Value = agent(Duration::from_secs(120))
        .post(&format!("{base}/chat/completions"))
        .send_json(&body)
        .map_err(|e| format!("advisor request failed: {e}"))?
        .into_body()
        .read_json()
        .map_err(|e| format!("advisor response is not JSON: {e}"))?;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("advisor returned no content")?;
    let advice: Advice =
        serde_json::from_str(content).map_err(|e| format!("advisor output rejected (schema): {e}"))?;
    // Keep only known columns and roles; clamp confidence.
    let known: std::collections::HashSet<&str> = profile.columns.iter().map(|c| c.name.as_str()).collect();
    Ok(advice
        .columns
        .into_iter()
        .filter(|a| {
            known.contains(a.column.as_str())
                && ROLES.contains(&a.role.as_str())
                && Some(a.column.as_str()) != target
        })
        .map(|mut a| {
            a.confidence = a.confidence.clamp(0.0, 1.0);
            a
        })
        .collect())
}

/// Turns advice into non-blocking findings, skipping columns the statistical
/// checks already flagged and roles that need no action.
pub fn findings(
    advice: &[ColumnAdvice],
    profile: &Profile,
    already_flagged: &[&str],
    sensitive: &[String],
) -> Vec<Finding> {
    let mut out = Vec::new();
    for a in advice {
        if a.confidence < MIN_CONFIDENCE || already_flagged.contains(&a.column.as_str()) {
            continue;
        }
        let kind = profile.column(&a.column).map(|c| c.kind);
        let (title, suggestion) = match a.role.as_str() {
            "post_outcome" => (
                "Advisor: may only be known after the outcome",
                format!(
                    "If `{}` is filled after the outcome, add it to `data.drop` (it would leak).",
                    a.column
                ),
            ),
            "sensitive" if !sensitive.contains(&a.column) => (
                "Advisor: sensitive attribute",
                format!(
                    "Add `{}` to `data.sensitive` to get the fairness report and proxy checks.",
                    a.column
                ),
            ),
            "personal_data" => (
                "Advisor: personal data",
                format!(
                    "Drop or mask `{}` (LGPD/GDPR); it is rarely a useful feature.",
                    a.column
                ),
            ),
            "identifier" if !matches!(kind, Some(ColumnKind::HighCardinality | ColumnKind::Constant)) => (
                "Advisor: identifier",
                format!(
                    "If `{}` is just an id, add it to `data.drop`; if rows share it, consider `data.group_column`.",
                    a.column
                ),
            ),
            _ => continue,
        };
        out.push(Finding {
            id: format!("SLM:{}", a.column),
            check: "SLM".into(),
            severity: Severity::Yellow,
            column: Some(a.column.clone()),
            title: title.into(),
            evidence: format!("{} (confidence {:.2})", a.reason, a.confidence),
            suggestion,
            resolution: Resolution::Auto {
                action: "suggestion only; statistics decide".into(),
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_local_urls_without_opt_in() {
        assert!(is_local("http://127.0.0.1:11434/v1"));
        assert!(is_local("http://localhost:8080"));
        assert!(is_local("http://[::1]:11434/v1"));
        assert!(!is_local("https://api.example.com/v1"));
        let spec = AdvisorSpec {
            enabled: true,
            model: "x".into(),
            remote: false,
        };
        let p = Profile {
            version: 1,
            dataset_id: String::new(),
            rows: 0,
            columns: vec![],
            sample: vec![],
            sampled_rows: None,
        };
        assert!(matches!(
            advise_at("https://api.example.com/v1", &spec, &p, None),
            AdvisorResult::Skipped(s) if s.contains("not local")
        ));
    }

    #[test]
    fn low_confidence_and_flagged_columns_are_ignored() {
        let advice = vec![
            ColumnAdvice {
                column: "refund".into(),
                role: "post_outcome".into(),
                confidence: 0.9,
                reason: "after churn".into(),
            },
            ColumnAdvice {
                column: "x".into(),
                role: "post_outcome".into(),
                confidence: 0.3,
                reason: "?".into(),
            },
            ColumnAdvice {
                column: "email".into(),
                role: "personal_data".into(),
                confidence: 0.95,
                reason: "e-mail".into(),
            },
            ColumnAdvice {
                column: "plan".into(),
                role: "feature".into(),
                confidence: 1.0,
                reason: "ok".into(),
            },
        ];
        let p = Profile {
            version: 1,
            dataset_id: String::new(),
            rows: 0,
            columns: vec![],
            sample: vec![],
            sampled_rows: None,
        };
        let f = findings(&advice, &p, &["email"], &[]);
        assert_eq!(
            f.iter().map(|f| f.id.as_str()).collect::<Vec<_>>(),
            ["SLM:refund"]
        );
        assert!(!f[0].blocks());
    }
}

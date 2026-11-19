//! A starter `tempera.yaml` for a dataset (used by `tempera init` and the API).

use tempera_core::JobSpec;
use tempera_data::{ColumnKind, Profile};

/// Builds a commented spec. `source` is written as given (relative or absolute).
pub fn draft_yaml(
    profile: &Profile,
    source: &str,
    target: Option<&str>,
    name: &str,
) -> anyhow::Result<String> {
    let columns: Vec<&str> = profile.columns.iter().map(|c| c.name.as_str()).collect();
    if let Some(t) = target
        && !columns.contains(&t)
    {
        anyhow::bail!("`{t}` is not a column (columns: {})", columns.join(", "));
    }
    let name: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let time_hint = profile
        .columns
        .iter()
        .find(|c| c.kind == ColumnKind::Datetime && Some(c.name.as_str()) != target)
        .map(|c| {
            format!(
                "  # time_column: {}   # uncomment if you will predict future rows\n",
                c.name
            )
        })
        .unwrap_or_default();
    let target_line = match target {
        Some(t) => format!("  target: {t}\n"),
        None => "  # no target: the job clusters the rows (set task.type: anomaly_detection for outliers)\n"
            .into(),
    };
    let yaml = format!(
        "# Têmpera job spec. Reference: docs/wiki/Contracts.md · schema: `tempera schema`\nversion: 1\nname: {name}\ndata:\n  source: {source}\n{target_line}{time_hint}  # group_column: customer_id   # rows of the same entity never split across train/test\n  # drop: []                    # columns to ignore (e.g. audit findings)\n# audit:\n#   acknowledge: []             # finding ids you reviewed and accept\nsplit:\n  test_size: 0.2\n  seed: 42\n"
    );
    JobSpec::from_yaml(&yaml).map_err(|e| anyhow::anyhow!("generated spec is invalid (bug): {e}"))?;
    Ok(yaml)
}

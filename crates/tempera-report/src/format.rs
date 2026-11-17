//! Number and label formatting shared by the HTML sections and the charts.
//!
//! Every function here is total: non-finite inputs become an em dash, so the
//! report never shows `NaN` or `inf`.

use tempera_audit::{Resolution, Severity};
use tempera_core::spec::{Metric, SplitStrategy, TaskType};
use tempera_data::split::FeatureKind;

/// Placeholder for missing or non-finite values.
pub const DASH: &str = "—";

/// A metric value: 4 decimals below 1, 3 decimals below 1000, then grouped with 1 decimal.
pub fn metric(v: f64) -> String {
    if !v.is_finite() {
        return DASH.to_owned();
    }
    let a = v.abs();
    if a < 1.0 {
        format!("{v:.4}")
    } else if a < 1000.0 {
        format!("{v:.3}")
    } else {
        grouped_decimal(v, 1)
    }
}

/// An optional metric value.
pub fn metric_opt(v: Option<f64>) -> String {
    v.map_or_else(|| DASH.to_owned(), metric)
}

/// A signed difference with 3 decimals, e.g. `+0.214` (uses a real minus sign).
pub fn signed(v: f64) -> String {
    if !v.is_finite() {
        return DASH.to_owned();
    }
    let body = if v.abs() < 1000.0 {
        format!("{:.3}", v.abs())
    } else {
        grouped_decimal(v.abs(), 1)
    };
    if v < 0.0 {
        format!("\u{2212}{body}")
    } else {
        format!("+{body}")
    }
}

/// A count with thousands separators: `1234567` → `1,234,567`.
pub fn count(n: u64) -> String {
    group_digits(&n.to_string())
}

/// A share as a percentage with one decimal.
pub fn percent(share: f64) -> String {
    if share.is_finite() {
        format!("{:.1}%", share * 100.0)
    } else {
        DASH.to_owned()
    }
}

/// A duration: `0.4s`, `12s`, `1m 23s`, `2h 05m`.
pub fn duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return DASH.to_owned();
    }
    if seconds < 10.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// Cost in US dollars; zero is shown as a local (free) run.
pub fn cost(usd: f64) -> String {
    if !usd.is_finite() {
        DASH.to_owned()
    } else if usd.abs() < 0.005 {
        "US$ 0.00 (local)".to_owned()
    } else {
        format!("US$ {}", grouped_decimal(usd, 2))
    }
}

fn grouped_decimal(v: f64, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, v.abs());
    let (int, frac) = s.split_once('.').unwrap_or((s.as_str(), ""));
    let sign = if v < 0.0 { "\u{2212}" } else { "" };
    if frac.is_empty() {
        format!("{sign}{}", group_digits(int))
    } else {
        format!("{sign}{}.{frac}", group_digits(int))
    }
}

fn group_digits(digits: &str) -> String {
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Human label of a metric.
pub fn metric_label(m: Metric) -> &'static str {
    match m {
        Metric::RocAuc => "ROC-AUC",
        Metric::PrAuc => "PR-AUC",
        Metric::F1 => "F1",
        Metric::BalancedAccuracy => "Balanced accuracy",
        Metric::Accuracy => "Accuracy",
        Metric::LogLoss => "Log loss",
        Metric::Rmse => "RMSE",
        Metric::Mae => "MAE",
        Metric::Mape => "MAPE",
        Metric::Mase => "MASE",
        Metric::R2 => "R²",
        Metric::Silhouette => "Silhouette",
        Metric::Perplexity => "Perplexity",
    }
}

/// Human label of a metric given its snake_case name (falls back to the name itself).
pub fn metric_key_label(key: &str) -> String {
    match serde_json::from_value::<Metric>(serde_json::Value::String(key.to_owned())) {
        Ok(m) => metric_label(m).to_owned(),
        Err(_) => humanize(key),
    }
}

/// `some_snake_name` → `Some snake name`.
pub fn humanize(s: &str) -> String {
    let spaced = s.replace(['_', '-'], " ");
    let mut chars = spaced.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

pub fn task_label(t: TaskType) -> &'static str {
    match t {
        TaskType::BinaryClassification => "Binary classification",
        TaskType::MulticlassClassification => "Multiclass classification",
        TaskType::Regression => "Regression",
        TaskType::TimeSeriesForecast => "Time-series forecast",
        TaskType::Clustering => "Clustering",
        TaskType::AnomalyDetection => "Anomaly detection",
        TaskType::ImageClassification => "Image classification",
        TaskType::TextClassification => "Text classification",
        TaskType::LlmFineTune => "LLM fine-tune",
    }
}

pub fn strategy_label(s: SplitStrategy) -> &'static str {
    match s {
        SplitStrategy::Auto => "Auto",
        SplitStrategy::Random => "Random",
        SplitStrategy::Stratified => "Stratified",
        SplitStrategy::Group => "Group (no entity in two partitions)",
        SplitStrategy::Temporal => "Temporal (tested on the future)",
    }
}

pub fn feature_kind_label(k: FeatureKind) -> &'static str {
    match k {
        FeatureKind::Numeric => "numeric",
        FeatureKind::Categorical => "categorical",
        FeatureKind::Boolean => "boolean",
    }
}

/// `(css class, badge text)` for a severity.
pub fn severity_badge(s: Severity) -> (&'static str, &'static str) {
    match s {
        Severity::Red => ("red", "Blocking"),
        Severity::Yellow => ("yellow", "Warning"),
        Severity::Green => ("green", "OK"),
    }
}

/// `(css class, label, optional detail)` for a resolution.
pub fn resolution_parts(r: &Resolution) -> (&'static str, &'static str, Option<&str>) {
    match r {
        Resolution::Unresolved => ("unresolved", "Unresolved", None),
        Resolution::Auto { action } => ("auto", "Auto action", Some(action.as_str())),
        Resolution::Acknowledged => ("ack", "Acknowledged", None),
    }
}

/// Truncates to `max` characters, adding an ellipsis.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_owned()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(count(0), "0");
        assert_eq!(count(1_234_567), "1,234,567");
        assert_eq!(count(999), "999");
        assert_eq!(metric(0.91234), "0.9123");
        assert_eq!(metric(12.3456), "12.346");
        assert_eq!(metric(12345.67), "12,345.7");
        assert_eq!(metric(f64::NAN), DASH);
        assert_eq!(metric(f64::INFINITY), DASH);
        assert_eq!(signed(0.2144), "+0.214");
        assert_eq!(signed(-0.5), "\u{2212}0.500");
        assert_eq!(percent(0.123), "12.3%");
    }

    #[test]
    fn durations_and_cost() {
        assert_eq!(duration(0.42), "0.4s");
        assert_eq!(duration(42.0), "42s");
        assert_eq!(duration(83.0), "1m 23s");
        assert_eq!(duration(7500.0), "2h 05m");
        assert_eq!(duration(f64::NAN), DASH);
        assert_eq!(cost(0.0), "US$ 0.00 (local)");
        assert_eq!(cost(1234.5), "US$ 1,234.50");
    }

    #[test]
    fn labels() {
        assert_eq!(metric_key_label("roc_auc"), "ROC-AUC");
        assert_eq!(metric_key_label("custom_score"), "Custom score");
        assert_eq!(truncate("abcdef", 4), "abc…");
    }
}

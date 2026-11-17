//! Terminal output. Plain text with a few symbols; colors only on a TTY.

use std::io::IsTerminal;
use std::path::Path;

use serde::Serialize;
use tempera_audit::{AuditReport, Finding, Resolution, Severity};
use tempera_core::metrics::higher_is_better;
use tempera_core::spec::Metric;
use tempera_data::eval::metric_name;
use tempera_report::{RunSummary, Winner};

fn color() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

fn paint(code: &str, s: &str) -> String {
    if color() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_owned()
    }
}

pub fn header(s: &str) {
    println!("\n{}", paint("1", &format!("▌ têmpera · {s}")));
}

pub fn step(s: &str) {
    println!("  {} {s}", paint("36", "●"));
}

pub fn detail(s: &str) {
    println!("      {}", paint("2", s));
}

pub fn finding(f: &Finding) {
    let badge = match f.severity {
        Severity::Red => paint("31;1", "RED   "),
        Severity::Yellow => paint("33", "YELLOW"),
        Severity::Green => paint("32", "GREEN "),
    };
    let res = match &f.resolution {
        Resolution::Unresolved => paint("31", "needs a decision"),
        Resolution::Auto { action } => paint("2", &format!("auto: {action}")),
        Resolution::Acknowledged => paint("2", "acknowledged"),
    };
    println!("      {badge} {:<22} {} · {res}", f.id, f.title);
    println!("             {}", paint("2", &f.evidence));
}

pub fn blocked(a: &AuditReport) {
    println!(
        "\n  {} Training blocked by the audit (policy: block_on_red).",
        paint("31;1", "✗")
    );
    for f in a.blocking() {
        println!("    • {}: {}", f.id, f.suggestion);
    }
    println!("    Fix the data or edit tempera.yaml (`data.drop` / `audit.acknowledge`), then run again.");
}

pub fn run_line(s: &RunSummary, metric: Metric) {
    let name = format!("{} ({})", s.algorithm, s.backend);
    let line = match s.status.as_str() {
        "succeeded" => format!(
            "{} {:<34} {} {:>8}  {}",
            paint("32", "✓"),
            name,
            metric_name(metric),
            s.valid_score.map_or("—".into(), |v| format!("{v:.4}")),
            paint("2", &secs(s.duration_s))
        ),
        "skipped" => format!(
            "{} {:<34} {}",
            paint("33", "–"),
            name,
            paint("2", &format!("skipped: {}", s.error.as_deref().unwrap_or("")))
        ),
        _ => format!(
            "{} {:<34} {}",
            paint("31", "✗"),
            name,
            paint("31", s.error.as_deref().unwrap_or("failed"))
        ),
    };
    println!("      {line}");
}

pub fn verdict(w: &Winner, metric: Metric, board: &[RunSummary]) {
    let test = w
        .test
        .metrics
        .get(&metric_name(metric))
        .copied()
        .unwrap_or(f64::NAN);
    let base = board
        .iter()
        .filter(|r| r.is_baseline)
        .filter_map(|r| r.valid_score)
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| {
                if higher_is_better(metric) {
                    a.max(v)
                } else {
                    a.min(v)
                }
            }))
        });
    let best_valid = board
        .iter()
        .find(|r| r.run_id == w.run_id)
        .and_then(|r| r.valid_score);
    step(&format!(
        "Winner    {} · test {} = {test:.4} (evaluated once)",
        w.algorithm,
        metric_name(metric)
    ));
    if let (Some(b), Some(v)) = (base, best_valid) {
        let name = metric_name(metric);
        if higher_is_better(metric) {
            detail(&format!(
                "{:+.4} {name} over the trivial baseline on validation",
                v - b
            ));
        } else {
            let pct = if b != 0.0 {
                format!(" ({:.0}% lower)", (b - v) / b.abs() * 100.0)
            } else {
                String::new()
            };
            detail(&format!(
                "{name} {v:.4} vs {b:.4} for the trivial baseline on validation{pct}"
            ));
        }
    }
}

pub fn done(report: &Path, seconds: f64) {
    println!(
        "\n  {} Done in {}. Report: {}\n",
        paint("32;1", "✓"),
        secs(seconds),
        report.display()
    );
}

/// Backend logs: warnings and errors always, the rest with TEMPERA_VERBOSE=1.
pub fn backend_log(level: i32, msg: &str) {
    use tempera_proto::v1::LogLevel;
    let verbose = std::env::var_os("TEMPERA_VERBOSE").is_some();
    match LogLevel::try_from(level) {
        Ok(LogLevel::Warn) => println!("        {}", paint("33", &format!("warn: {msg}"))),
        Ok(LogLevel::Error) => println!("        {}", paint("31", &format!("error: {msg}"))),
        _ if verbose => println!("        {}", paint("2", msg)),
        _ => {}
    }
}

pub fn count(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn secs(s: f64) -> String {
    if s < 60.0 {
        format!("{s:.1}s")
    } else {
        format!("{}m {:02}s", (s / 60.0) as u64, (s % 60.0) as u64)
    }
}

pub fn snake<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

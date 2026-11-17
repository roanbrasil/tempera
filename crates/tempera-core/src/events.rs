//! Backend → agent events, one JSON object per line on stdout (JSONL).
//!
//! Backends (Python or Rust) never speak gRPC: they print events and the agent
//! forwards them. Anything on stdout that is not a valid event is reported as
//! an invalid line instead of stopping the job.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::spec::ExportFormat;

/// Maximum accepted line length (bytes). Longer lines are reported as invalid.
pub const MAX_LINE_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Event {
    /// A scalar metric at a training step.
    Metric {
        name: String,
        value: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        epoch: Option<u32>,
        /// Unix timestamp (seconds) when the metric was produced.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ts: Option<f64>,
    },
    Log {
        #[serde(default)]
        level: LogLevel,
        msg: String,
    },
    /// Progress for ETA estimation.
    Progress {
        step: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_steps: Option<u64>,
    },
    /// A checkpoint was written and can be uploaded.
    Checkpoint { path: PathBuf, step: u64 },
    /// A final artifact (model, report, ...).
    Artifact {
        kind: ArtifactKind,
        path: PathBuf,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<ExportFormat>,
    },
    /// The backend finished successfully.
    Done {
        #[serde(default)]
        summary: BTreeMap<String, f64>,
    },
    /// The backend failed in a way it could describe.
    Error {
        msg: String,
        #[serde(default)]
        retryable: bool,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Model,
    Report,
    Predictions,
    Other,
}

impl Event {
    /// Parses one line. Surrounding whitespace is ignored.
    pub fn parse_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line.trim())
    }

    /// Serializes to a single JSONL line (without the trailing newline).
    pub fn to_line(&self) -> String {
        // Serializing these plain data types cannot fail.
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// The outcome of reading one line of backend output.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    Event(Event),
    /// Not a valid event; kept so it can be forwarded as a raw log.
    Invalid {
        line_no: u64,
        raw: String,
        error: String,
    },
}

/// Reads backend stdout as a stream of [`Line`]s. Blank lines are skipped and
/// invalid lines never end the stream.
#[derive(Debug)]
pub struct EventReader<R> {
    inner: R,
    line_no: u64,
    buf: Vec<u8>,
}

impl<R: BufRead> EventReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            line_no: 0,
            buf: Vec::with_capacity(256),
        }
    }
}

impl<R: BufRead> Iterator for EventReader<R> {
    type Item = std::io::Result<Line>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.buf.clear();
            match self.inner.read_until(b'\n', &mut self.buf) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(e)),
            }
            self.line_no += 1;
            let raw = String::from_utf8_lossy(&self.buf);
            let text = raw.trim();
            if text.is_empty() {
                continue;
            }
            if self.buf.len() > MAX_LINE_BYTES {
                return Some(Ok(Line::Invalid {
                    line_no: self.line_no,
                    raw: text.chars().take(200).collect(),
                    error: format!("line longer than {MAX_LINE_BYTES} bytes"),
                }));
            }
            return Some(Ok(match Event::parse_line(text) {
                Ok(ev) => Line::Event(ev),
                Err(e) => Line::Invalid {
                    line_no: self.line_no,
                    raw: text.to_owned(),
                    error: e.to_string(),
                },
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_documented_examples() {
        let lines = [
            r#"{"t":"metric","step":120,"epoch":2,"name":"val/pr_auc","value":0.731,"ts":1763251200.1}"#,
            r#"{"t":"log","level":"info","msg":"epoch 2 done"}"#,
            r#"{"t":"progress","step":120,"total_steps":2000}"#,
            r#"{"t":"checkpoint","path":"/work/ckpt/step-120","step":120}"#,
            r#"{"t":"artifact","kind":"model","format":"onnx","path":"/work/out/model.onnx"}"#,
            r#"{"t":"done","summary":{"val/pr_auc":0.74}}"#,
            r#"{"t":"error","msg":"CUDA out of memory","retryable":true}"#,
        ];
        for l in lines {
            let ev = Event::parse_line(l).unwrap();
            assert_eq!(Event::parse_line(&ev.to_line()).unwrap(), ev);
        }
    }

    #[test]
    fn reader_tolerates_garbage() {
        let input =
            "\n{\"t\":\"log\",\"msg\":\"hi\"}\nnot json\n{\"t\":\"nope\"}\n{\"t\":\"progress\",\"step\":1}";
        let out: Vec<_> = EventReader::new(Cursor::new(input)).map(Result::unwrap).collect();
        assert_eq!(out.len(), 4);
        assert!(matches!(out[0], Line::Event(Event::Log { .. })));
        assert!(matches!(out[1], Line::Invalid { line_no: 3, .. }));
        assert!(matches!(out[2], Line::Invalid { line_no: 4, .. }));
        assert!(matches!(out[3], Line::Event(Event::Progress { step: 1, .. })));
    }

    #[test]
    fn reader_rejects_huge_lines() {
        let input = format!("{}\n", "x".repeat(MAX_LINE_BYTES + 1));
        let out: Vec<_> = EventReader::new(Cursor::new(input)).map(Result::unwrap).collect();
        assert!(matches!(out[0], Line::Invalid { .. }));
    }

    #[test]
    fn throughput_100k_events() {
        let line = Event::Metric {
            name: "loss".into(),
            value: 0.5,
            step: Some(1),
            epoch: None,
            ts: None,
        }
        .to_line();
        let input = format!("{line}\n").repeat(100_000);
        let start = std::time::Instant::now();
        let n = EventReader::new(Cursor::new(input))
            .filter(|l| matches!(l, Ok(Line::Event(_))))
            .count();
        assert_eq!(n, 100_000);
        // Generous bound so debug builds on slow CI machines pass.
        assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
    }
}

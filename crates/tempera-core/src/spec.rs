//! `tempera.yaml`: the job specification.
//!
//! This module is the single source of truth for the spec format. The JSON
//! Schema in `schemas/tempera.schema.json` is generated from these types
//! (see the `schema` module) and checked in CI.

use std::borrow::Cow;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};

use crate::error::{SpecError, ValidationIssue};

/// Current spec format version.
pub const SPEC_VERSION: u32 = 1;

/// A complete job specification (`tempera.yaml`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    /// Spec format version. Must be `1`.
    pub version: u32,
    /// Human-friendly job name (letters, digits, `-`, `_`).
    pub name: String,
    pub data: DataSpec,
    #[serde(default)]
    pub task: TaskSpec,
    #[serde(default)]
    pub audit: AuditSpec,
    #[serde(default)]
    pub split: SplitSpec,
    #[serde(default)]
    pub strategy: StrategySpec,
    #[serde(default)]
    pub budget: BudgetSpec,
    #[serde(default)]
    pub compute: ComputeSpec,
    #[serde(default)]
    pub output: OutputSpec,
}

/// Where the data comes from and which columns play special roles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataSpec {
    /// Path or URI (`./file.parquet`, `s3://...`, `postgres://...`, a folder).
    pub source: String,
    /// Column to predict. Optional for unsupervised tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Time column; enables temporal splits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_column: Option<String>,
    /// Entity column (e.g. `customer_id`); enables group splits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_column: Option<String>,
    /// Sensitive attributes used for the fairness report.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sensitive: Vec<String>,
    /// Columns to drop (usually decisions taken during the audit).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    /// Task type. Inferred from the data when omitted.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<TaskType>,
    /// Main metric. Recommended automatically when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<Metric>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    BinaryClassification,
    MulticlassClassification,
    Regression,
    TimeSeriesForecast,
    Clustering,
    AnomalyDetection,
    ImageClassification,
    TextClassification,
    LlmFineTune,
}

impl TaskType {
    /// Whether the task needs a target column.
    pub fn is_supervised(self) -> bool {
        !matches!(self, Self::Clustering | Self::AnomalyDetection)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    RocAuc,
    PrAuc,
    F1,
    BalancedAccuracy,
    Accuracy,
    LogLoss,
    Rmse,
    Mae,
    Mape,
    Mase,
    R2,
    Silhouette,
    Perplexity,
}

impl Metric {
    /// Task types for which this metric makes sense.
    pub fn fits(self, task: TaskType) -> bool {
        use Metric::*;
        use TaskType::*;
        match self {
            RocAuc | PrAuc => matches!(task, BinaryClassification | MulticlassClassification),
            F1 | BalancedAccuracy | Accuracy | LogLoss => matches!(
                task,
                BinaryClassification | MulticlassClassification | ImageClassification | TextClassification
            ),
            Rmse | Mae | R2 => matches!(task, Regression | TimeSeriesForecast),
            Mape | Mase => matches!(task, Regression | TimeSeriesForecast),
            Silhouette => matches!(task, Clustering),
            Perplexity => matches!(task, LlmFineTune),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AuditSpec {
    #[serde(default)]
    pub policy: AuditPolicy,
    #[serde(default)]
    pub advisor: AdvisorSpec,
    /// Finding ids (e.g. `A4`, `A1:refund_amount`) the user reviewed and accepts.
    /// Acknowledged red findings no longer block training; the decision is recorded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acknowledge: Vec<String>,
}

impl Default for AuditSpec {
    fn default() -> Self {
        Self {
            policy: AuditPolicy::BlockOnRed,
            advisor: AdvisorSpec::default(),
            acknowledge: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AuditPolicy {
    /// Red findings block training until the user decides.
    #[default]
    BlockOnRed,
    /// Findings never block; they are only reported.
    WarnOnly,
}

/// The local SLM that explains audit findings. It advises; it never decides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AdvisorSpec {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_advisor_model")]
    pub model: String,
    /// Allow a remote LLM API. Off by default for privacy.
    #[serde(default)]
    pub remote: bool,
}

impl Default for AdvisorSpec {
    fn default() -> Self {
        Self {
            enabled: true,
            model: default_advisor_model(),
            remote: false,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_advisor_model() -> String {
    "qwen3-4b-q4".to_owned()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SplitSpec {
    #[serde(default)]
    pub strategy: SplitStrategy,
    /// Fraction held out as the final test set, in (0, 0.5].
    #[serde(default = "default_test_size")]
    pub test_size: f64,
    /// Cross-validation folds (>= 2).
    #[serde(default = "default_folds")]
    pub folds: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for SplitSpec {
    fn default() -> Self {
        Self {
            strategy: SplitStrategy::Auto,
            test_size: default_test_size(),
            folds: default_folds(),
            seed: default_seed(),
        }
    }
}

fn default_test_size() -> f64 {
    0.2
}
fn default_folds() -> u32 {
    5
}
fn default_seed() -> u64 {
    42
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SplitStrategy {
    /// Picked from the audit: temporal > group > stratified > random.
    #[default]
    Auto,
    Random,
    Stratified,
    Group,
    Temporal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StrategySpec {
    #[serde(default)]
    pub init: InitMode,
    /// `auto` or an explicit list of algorithm names.
    #[serde(default)]
    pub candidates: Candidates,
    /// Always train cheap baselines first.
    #[serde(default = "default_true")]
    pub baseline: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hpo: Option<HpoSpec>,
}

impl Default for StrategySpec {
    fn default() -> Self {
        Self {
            init: InitMode::Auto,
            candidates: Candidates::default(),
            baseline: true,
            hpo: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InitMode {
    #[default]
    Auto,
    Scratch,
    Pretrained,
}

/// Either the literal `auto` or a list of algorithm names.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Candidates {
    #[default]
    Auto,
    List(Vec<String>),
}

impl Serialize for Candidates {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Auto => s.serialize_str("auto"),
            Self::List(list) => list.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for Candidates {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Word(String),
            List(Vec<String>),
        }
        match Raw::deserialize(d)? {
            Raw::Word(w) if w == "auto" => Ok(Self::Auto),
            Raw::Word(w) => Err(serde::de::Error::custom(format!(
                "expected `auto` or a list of algorithms, got `{w}`"
            ))),
            Raw::List(list) => Ok(Self::List(list)),
        }
    }
}

impl JsonSchema for Candidates {
    fn schema_name() -> Cow<'static, str> {
        "Candidates".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "description": "`auto` or a list of algorithm names.",
            "oneOf": [
                { "const": "auto" },
                { "type": "array", "items": { "type": "string" } }
            ]
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HpoSpec {
    #[serde(default)]
    pub method: HpoMethod,
    #[serde(default = "default_trials")]
    pub max_trials: u32,
}

fn default_trials() -> u32 {
    30
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HpoMethod {
    /// Asynchronous successive halving: kills bad configs early.
    #[default]
    Asha,
    Random,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost: Option<Money>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_time: Option<HumanDuration>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Money {
    pub amount: f64,
    /// ISO 4217 code, e.g. `USD`, `BRL`, `EUR`.
    pub currency: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComputeSpec {
    #[serde(default)]
    pub objective: ComputeObjective,
    /// Required when `objective` is `cheapest_under`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<HumanDuration>,
    /// Allowed machines/providers. Empty means local only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    /// Prefer interruptible (spot) instances on cloud providers.
    #[serde(default = "default_true")]
    pub spot: bool,
    #[serde(default = "default_restarts")]
    pub max_restarts: u32,
}

impl Default for ComputeSpec {
    fn default() -> Self {
        Self {
            objective: ComputeObjective::Cheapest,
            deadline: None,
            allow: Vec::new(),
            spot: true,
            max_restarts: default_restarts(),
        }
    }
}

fn default_restarts() -> u32 {
    3
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComputeObjective {
    #[default]
    Cheapest,
    Fastest,
    CheapestUnder,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    #[serde(default = "default_formats")]
    pub formats: Vec<ExportFormat>,
    #[serde(default = "default_true")]
    pub report: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            formats: default_formats(),
            report: true,
            dir: None,
        }
    }
}

fn default_formats() -> Vec<ExportFormat> {
    vec![ExportFormat::Onnx]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    Onnx,
    Safetensors,
    Native,
}

/// A duration written as `<n><unit>`: `90s`, `30m`, `2h`, `1d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HumanDuration(pub Duration);

impl HumanDuration {
    pub fn as_duration(self) -> Duration {
        self.0
    }
}

impl FromStr for HumanDuration {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        let (num, unit) = s.split_at(split);
        let n: u64 = num
            .parse()
            .map_err(|_| format!("invalid duration `{s}` (use e.g. 30m, 2h)"))?;
        let secs = match unit {
            "s" => n,
            "m" => n * 60,
            "h" => n * 3600,
            "d" => n * 86_400,
            _ => return Err(format!("invalid duration unit in `{s}` (use s, m, h or d)")),
        };
        if secs == 0 {
            return Err("duration must be greater than zero".to_owned());
        }
        Ok(Self(Duration::from_secs(secs)))
    }
}

impl fmt::Display for HumanDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0.as_secs();
        match s {
            _ if s.is_multiple_of(86_400) => write!(f, "{}d", s / 86_400),
            _ if s.is_multiple_of(3600) => write!(f, "{}h", s / 3600),
            _ if s.is_multiple_of(60) => write!(f, "{}m", s / 60),
            _ => write!(f, "{s}s"),
        }
    }
}

impl Serialize for HumanDuration {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for HumanDuration {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for HumanDuration {
    fn schema_name() -> Cow<'static, str> {
        "HumanDuration".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "string",
            "pattern": "^[0-9]+(s|m|h|d)$",
            "description": "Duration such as `90s`, `30m`, `2h`, `1d`."
        })
    }
}

impl JobSpec {
    /// Parses and validates a spec from YAML text.
    ///
    /// Errors point at the offending field path (e.g. `split.test_size`).
    pub fn from_yaml(text: &str) -> Result<Self, SpecError> {
        // Deserialize straight into the typed spec (so a column named `y` or
        // `no` stays a string, not a YAML 1.1 boolean) and record the field
        // path of the first error, e.g. `split.folds`.
        let mut path = None;
        let spec: JobSpec = serde_saphyr::with_deserializer_from_str(text, |de| {
            serde_path_to_error::deserialize(de).map_err(|e| {
                path = Some(e.path().to_string());
                e.into_inner()
            })
        })
        .map_err(|e| SpecError::Parse {
            path: path
                .clone()
                .filter(|p| p != ".")
                .unwrap_or_else(|| "<yaml>".to_owned()),
            message: e.to_string(),
        })?;
        spec.validate()?;
        Ok(spec)
    }

    /// Serializes the spec back to YAML.
    pub fn to_yaml(&self) -> Result<String, SpecError> {
        serde_saphyr::to_string(self).map_err(|e| SpecError::Serialize(e.to_string()))
    }

    /// Semantic checks that the type system cannot express.
    pub fn validate(&self) -> Result<(), SpecError> {
        let mut issues = Vec::new();
        let mut push = |path: &str, message: String| {
            issues.push(ValidationIssue {
                path: path.to_owned(),
                message,
            });
        };

        if self.version != SPEC_VERSION {
            push(
                "version",
                format!("unsupported version {}; expected {SPEC_VERSION}", self.version),
            );
        }
        if self.name.is_empty()
            || !self
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            push("name", "use only letters, digits, `-` and `_`".to_owned());
        }
        if self.data.source.trim().is_empty() {
            push("data.source", "must not be empty".to_owned());
        }
        if let Some(kind) = self.task.kind {
            if kind.is_supervised() && self.data.target.is_none() {
                push("data.target", format!("required for task `{}`", enum_name(&kind)));
            }
            if let Some(metric) = self.task.metric
                && !metric.fits(kind)
            {
                push(
                    "task.metric",
                    format!(
                        "`{}` does not apply to `{}`",
                        enum_name(&metric),
                        enum_name(&kind)
                    ),
                );
            }
        }
        if let Some(target) = &self.data.target
            && self.data.drop.contains(target)
        {
            push("data.drop", format!("cannot drop the target column `{target}`"));
        }
        if !(self.split.test_size > 0.0 && self.split.test_size <= 0.5) {
            push("split.test_size", "must be in (0, 0.5]".to_owned());
        }
        if self.split.folds < 2 {
            push("split.folds", "must be at least 2".to_owned());
        }
        match self.split.strategy {
            SplitStrategy::Group if self.data.group_column.is_none() => {
                push(
                    "data.group_column",
                    "required by `split.strategy: group`".to_owned(),
                );
            }
            SplitStrategy::Temporal if self.data.time_column.is_none() => {
                push(
                    "data.time_column",
                    "required by `split.strategy: temporal`".to_owned(),
                );
            }
            _ => {}
        }
        if let Candidates::List(list) = &self.strategy.candidates
            && list.is_empty()
        {
            push(
                "strategy.candidates",
                "use `auto` or list at least one algorithm".to_owned(),
            );
        }
        if let Some(hpo) = &self.strategy.hpo
            && hpo.max_trials == 0
        {
            push("strategy.hpo.max_trials", "must be at least 1".to_owned());
        }
        if let Some(money) = &self.budget.max_cost {
            if money.amount <= 0.0 || !money.amount.is_finite() {
                push("budget.max_cost.amount", "must be a positive number".to_owned());
            }
            let c = &money.currency;
            if c.len() != 3 || !c.chars().all(|ch| ch.is_ascii_uppercase()) {
                push(
                    "budget.max_cost.currency",
                    format!("`{c}` is not an ISO 4217 code (e.g. USD)"),
                );
            }
        }
        if self.compute.objective == ComputeObjective::CheapestUnder && self.compute.deadline.is_none() {
            push(
                "compute.deadline",
                "required when `objective` is `cheapest_under`".to_owned(),
            );
        }
        if self.output.formats.is_empty() {
            push("output.formats", "list at least one export format".to_owned());
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(SpecError::Invalid(issues))
        }
    }
}

/// The serialized (snake_case) name of a unit enum variant.
fn enum_name<T: Serialize>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const FULL: &str = r#"
version: 1
name: customer-churn
data:
  source: ./data/customers.parquet
  target: churned
  time_column: reference_date
  group_column: customer_id
  sensitive: [gender, age_group]
  drop: [refunded_amount]
task:
  type: binary_classification
  metric: pr_auc
audit:
  policy: block_on_red
  advisor:
    model: qwen3-4b-q4
    remote: false
split:
  strategy: auto
  test_size: 0.2
  folds: 5
  seed: 42
strategy:
  init: auto
  candidates: auto
  baseline: true
  hpo:
    method: asha
    max_trials: 30
budget:
  max_cost: { amount: 2, currency: USD }
  max_time: 2h
compute:
  objective: cheapest_under
  deadline: 2h
  allow: [local, living-room-pc, runpod]
  spot: true
  max_restarts: 3
output:
  formats: [onnx, safetensors]
  report: true
"#;

    #[test]
    fn parses_full_spec() {
        let spec = JobSpec::from_yaml(FULL).unwrap();
        assert_eq!(spec.task.kind, Some(TaskType::BinaryClassification));
        assert_eq!(
            spec.budget.max_time.unwrap().as_duration(),
            Duration::from_secs(7200)
        );
        assert_eq!(spec.strategy.candidates, Candidates::Auto);
    }

    #[test]
    fn minimal_spec_gets_defaults() {
        let spec = JobSpec::from_yaml("version: 1\nname: x\ndata: { source: a.csv }\n").unwrap();
        assert_eq!(spec.split, SplitSpec::default());
        assert!(spec.strategy.baseline);
        assert_eq!(spec.audit.policy, AuditPolicy::BlockOnRed);
    }

    #[test]
    fn yaml_round_trip() {
        let spec = JobSpec::from_yaml(FULL).unwrap();
        let again = JobSpec::from_yaml(&spec.to_yaml().unwrap()).unwrap();
        assert_eq!(spec, again);
    }

    #[test]
    fn candidate_list() {
        let y = FULL.replace("candidates: auto", "candidates: [lightgbm, logistic_regression]");
        let spec = JobSpec::from_yaml(&y).unwrap();
        assert_eq!(
            spec.strategy.candidates,
            Candidates::List(vec!["lightgbm".into(), "logistic_regression".into()])
        );
    }

    #[test]
    fn parse_error_points_at_field() {
        let err = JobSpec::from_yaml(&FULL.replace("folds: 5", "folds: five")).unwrap_err();
        match err {
            SpecError::Parse { path, .. } => assert_eq!(path, "split.folds"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = JobSpec::from_yaml(&FULL.replace("seed: 42", "seed: 42\n  sed: 1")).unwrap_err();
        assert!(
            matches!(err, SpecError::Parse { ref path, .. } if path == "split.sed"),
            "{err:?}"
        );
    }

    #[test]
    fn semantic_errors_are_collected() {
        let y = FULL
            .replace("test_size: 0.2", "test_size: 0.9")
            .replace("metric: pr_auc", "metric: rmse")
            .replace("  deadline: 2h\n", "");
        let SpecError::Invalid(issues) = JobSpec::from_yaml(&y).unwrap_err() else {
            panic!("expected validation errors");
        };
        let paths: Vec<_> = issues.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(paths, ["task.metric", "split.test_size", "compute.deadline"]);
    }

    #[test]
    fn yaml11_boolean_words_stay_strings() {
        for name in ["y", "n", "yes", "no", "on", "off"] {
            let spec = JobSpec::from_yaml(&format!(
                "version: 1\nname: t\ndata: {{ source: x.csv, target: {name}, drop: [{name}2] }}\n"
            ))
            .unwrap();
            assert_eq!(spec.data.target.as_deref(), Some(name));
        }
    }

    #[test]
    fn human_duration() {
        assert_eq!("90s".parse::<HumanDuration>().unwrap().to_string(), "90s");
        assert_eq!("120m".parse::<HumanDuration>().unwrap().to_string(), "2h");
        assert!("2x".parse::<HumanDuration>().is_err());
        assert!("0h".parse::<HumanDuration>().is_err());
        assert!("h".parse::<HumanDuration>().is_err());
    }
}

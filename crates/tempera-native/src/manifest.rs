//! Backend run manifests (Contracts §7). Shared by the native backend, the
//! agent and the orchestrator, which write them for every backend.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tempera_core::spec::{ExportFormat, TaskType};

use crate::error::NativeError;

/// Kind of a feature column in the model-ready Parquet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureKind {
    /// Float64 / Int64 (other numeric widths are accepted and cast).
    Numeric,
    /// Utf8, nulls allowed.
    Categorical,
    /// Boolean, nulls allowed.
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSpec {
    pub name: String,
    pub kind: FeatureKind,
}

/// One `{input, output}` pair: predict `input`, write predictions to `output`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictTarget {
    pub input: PathBuf,
    pub output: PathBuf,
}

/// The `fit` manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub algorithm: String,
    pub task: TaskType,
    /// Sorted label list (classification only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
    /// Positive label (binary classification only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub positive_class: Option<String>,
    pub features: Vec<FeatureSpec>,
    /// Algorithm hyper-parameters; unknown keys are ignored.
    #[serde(default)]
    pub params: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub seed: u64,
    pub train: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid: Option<PathBuf>,
    #[serde(default)]
    pub predict: Vec<PredictTarget>,
    pub work_dir: PathBuf,
    #[serde(default)]
    pub export: Vec<ExportFormat>,
}

/// The `predict` manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictManifest {
    pub algorithm: String,
    pub task: TaskType,
    pub model_dir: PathBuf,
    pub input: PathBuf,
    pub output: PathBuf,
}

impl Manifest {
    /// Parses a fit manifest; any problem is a [`NativeError::Manifest`] (exit code 2).
    pub fn from_json(text: &str) -> Result<Self, NativeError> {
        serde_json::from_str(text).map_err(|e| NativeError::Manifest(e.to_string()))
    }

    /// Classes for classification tasks, checked for consistency.
    pub(crate) fn checked_classes(&self) -> Result<Vec<String>, NativeError> {
        let classes = self
            .classes
            .clone()
            .ok_or_else(|| NativeError::Manifest("`classes` is required for classification".into()))?;
        if classes.len() < 2 {
            return Err(NativeError::Manifest(
                "`classes` needs at least two labels".into(),
            ));
        }
        let mut dedup = classes.clone();
        dedup.sort();
        dedup.dedup();
        if dedup.len() != classes.len() {
            return Err(NativeError::Manifest("`classes` contains duplicates".into()));
        }
        if self.task == TaskType::BinaryClassification {
            if classes.len() != 2 {
                return Err(NativeError::Manifest(
                    "binary_classification needs exactly two classes".into(),
                ));
            }
            if let Some(p) = &self.positive_class
                && !classes.contains(p)
            {
                return Err(NativeError::Manifest(format!(
                    "positive_class `{p}` is not one of `classes`"
                )));
            }
        }
        Ok(classes)
    }
}

impl PredictManifest {
    /// Parses a predict manifest; any problem is a [`NativeError::Manifest`] (exit code 2).
    pub fn from_json(text: &str) -> Result<Self, NativeError> {
        serde_json::from_str(text).map_err(|e| NativeError::Manifest(e.to_string()))
    }
}

/// Whether the task is a classification task handled by this backend.
pub(crate) fn is_classification(task: TaskType) -> bool {
    matches!(
        task,
        TaskType::BinaryClassification | TaskType::MulticlassClassification
    )
}

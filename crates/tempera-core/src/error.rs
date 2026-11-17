//! Error types shared across crates.

use std::fmt;

/// One semantic problem found in a spec, anchored to a field path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Dotted field path, e.g. `split.test_size`.
    pub path: String,
    pub message: String,
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    /// The YAML could not be read into the spec types.
    #[error("{path}: {message}")]
    Parse { path: String, message: String },
    /// The spec parsed but breaks one or more rules.
    #[error("invalid spec:\n{}", .0.iter().map(|i| format!("  - {i}")).collect::<Vec<_>>().join("\n"))]
    Invalid(Vec<ValidationIssue>),
    #[error("could not serialize spec: {0}")]
    Serialize(String),
}

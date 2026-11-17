use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("file not found: {0}")]
    NotFound(PathBuf),
    #[error("unsupported file format: {0} (use .csv, .tsv, .parquet or .jsonl)")]
    UnsupportedFormat(PathBuf),
    #[error("column `{0}` not found in the dataset")]
    MissingColumn(String),
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T, E = DataError> = std::result::Result<T, E>;

use tempera_core::Event;

/// Errors of the native backend. [`NativeError::exit_code`] maps them to the
/// process exit codes of Contracts §7.
#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    #[error("bad manifest: {0}")]
    Manifest(String),
    #[error("unknown algorithm `{0}` (native backend supports: {supported})", supported = crate::SUPPORTED_ALGORITHMS.join(", "))]
    UnknownAlgorithm(String),
    #[error("algorithm `{algorithm}` does not support task `{task}`")]
    UnsupportedTask { algorithm: String, task: String },
    #[error("invalid data: {0}")]
    Data(String),
    #[error("training failed: {0}")]
    Training(String),
    #[error("model: {0}")]
    Model(String),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl NativeError {
    /// `2` for a bad manifest or an unknown/unsupported algorithm, `1` otherwise.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Manifest(_) | Self::UnknownAlgorithm(_) | Self::UnsupportedTask { .. } => 2,
            _ => 1,
        }
    }

    /// The `error` event a backend process prints before exiting.
    pub fn to_event(&self) -> Event {
        Event::Error {
            msg: self.to_string(),
            retryable: false,
        }
    }
}

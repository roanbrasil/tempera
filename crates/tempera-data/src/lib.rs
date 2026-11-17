//! Data ingestion (CSV/Parquet/JSONL → normalized Parquet), content hashing,
//! profiling and train/valid/test splits. See docs/wiki/Data-Audit.md.

pub mod error;
pub mod eval;
pub mod infer;
pub mod ingest;
pub mod profile;
pub mod split;

pub use error::{DataError, Result};
pub use infer::{TaskInference, infer_task};
pub use ingest::{Dataset, ingest};
pub use profile::{ColumnKind, Profile, profile_dataset};
pub use split::{Feature, FeatureKind, FeaturePlan, Partition, Split, make_split};

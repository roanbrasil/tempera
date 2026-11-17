//! Core domain types for Têmpera: the job spec (`tempera.yaml`), backend
//! events, and shared errors. Every other crate depends on this one.

pub mod error;
pub mod events;
pub mod schema;
pub mod spec;

pub use error::{SpecError, ValidationIssue};
pub use events::{Event, EventReader, Line};
pub use spec::JobSpec;

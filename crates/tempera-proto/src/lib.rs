//! gRPC types for the Têmpera agent protocol (`proto/tempera/agent/v1`).

/// Protocol version announced by the agent in `Register`.
pub const PROTOCOL_VERSION: u32 = 1;

#[allow(clippy::all, missing_debug_implementations)]
pub mod agent {
    pub mod v1 {
        tonic::include_proto!("tempera.agent.v1");
    }
}

pub use agent::v1;

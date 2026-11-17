//! Where jobs run: the agent control plane ([`hub::Hub`]) and executors that
//! bring agents up. M1 ships [`local::LocalExecutor`]; SSH (M4) and cloud
//! providers (M6) implement the same idea. See docs/wiki/Execution-and-Machines.md.

pub mod hub;
pub mod local;

pub use hub::Hub;
pub use local::LocalExecutor;

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("agent `{0}` is not connected")]
    NotConnected(String),
    #[error("agent `{0}` did not connect in time")]
    AgentTimeout(String),
    #[error("cannot find the tempera-agent binary (set TEMPERA_AGENT_BIN)")]
    AgentBinaryNotFound,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

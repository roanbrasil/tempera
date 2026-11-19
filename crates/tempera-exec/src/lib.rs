//! Where jobs run: the agent control plane ([`hub::Hub`]) and executors that
//! bring agents up. [`local::LocalExecutor`] runs the agent as a child process;
//! [`ssh::SshMachine`] installs and starts it on another machine over the
//! system OpenSSH client (ADR-0007). Cloud providers (M6) follow the same idea.
//! See docs/wiki/Execution-and-Machines.md.

pub mod hub;
pub mod local;
pub mod machines;
pub mod ssh;
pub mod system;

pub use hub::Hub;
pub use local::LocalExecutor;
pub use machines::{MachineConfig, MachinesFile};
pub use ssh::{CachedFile, RemoteInfo, SshMachine, SshSession, StartOpts};

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("agent `{0}` is not connected")]
    NotConnected(String),
    #[error("agent `{0}` did not connect in time")]
    AgentTimeout(String),
    #[error("cannot find the tempera-agent binary (set TEMPERA_AGENT_BIN)")]
    AgentBinaryNotFound,
    /// Anything that went wrong talking to (or on) a remote machine over SSH.
    #[error("machine `{machine}`: {message}")]
    Ssh { machine: String, message: String },
    /// Invalid machine definition or registry file.
    #[error("{0}")]
    Config(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

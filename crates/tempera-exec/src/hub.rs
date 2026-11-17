//! Agent control plane: the gRPC `AgentControl` service plus routing of agent
//! messages to whoever started each run.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use tempera_proto::v1::agent_control_server::{AgentControl, AgentControlServer};
use tempera_proto::v1::{
    AgentMessage, CancelJob, Hardware, JobFailed, Registered, ServerMessage, Shutdown, StartJob,
    agent_message, server_message,
};
use tokio::sync::{Mutex, Notify, mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::ExecError;

pub const HEARTBEAT_INTERVAL_S: u32 = 10;

type Outbound = mpsc::Sender<Result<ServerMessage, Status>>;

#[derive(Debug)]
struct AgentConn {
    tx: Outbound,
    hardware: Hardware,
    session: u64,
}

#[derive(Debug, Default)]
struct State {
    agents: HashMap<String, AgentConn>,
    /// run_id → subscriber.
    subs: HashMap<String, mpsc::UnboundedSender<AgentMessage>>,
    /// run_id → machine_id, to fail runs when their agent disconnects.
    running_on: HashMap<String, String>,
    next_session: u64,
}

/// Shared handle to connected agents.
#[derive(Debug, Clone, Default)]
pub struct Hub {
    state: Arc<Mutex<State>>,
    changed: Arc<Notify>,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Serves the gRPC endpoint on `addr` (use port 0 for an ephemeral port).
    pub async fn serve(
        &self,
        addr: SocketAddr,
    ) -> Result<(SocketAddr, tokio::task::JoinHandle<()>), ExecError> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let bound = listener.local_addr()?;
        let svc = AgentControlServer::new(HubService { hub: self.clone() });
        let handle = tokio::spawn(async move {
            if let Err(e) = tonic::transport::Server::builder()
                .add_service(svc)
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
            {
                tracing::error!(error = %e, "agent control server stopped");
            }
        });
        Ok((bound, handle))
    }

    /// Waits until `machine_id` is connected.
    pub async fn wait_for(&self, machine_id: &str, timeout: Duration) -> Result<Hardware, ExecError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.changed.notified();
            if let Some(a) = self.state.lock().await.agents.get(machine_id) {
                return Ok(a.hardware.clone());
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(ExecError::AgentTimeout(machine_id.to_owned()));
            }
        }
    }

    pub async fn connected(&self) -> Vec<String> {
        self.state.lock().await.agents.keys().cloned().collect()
    }

    /// Sends `StartJob` and returns the stream of messages for that run. The
    /// stream ends after `JobFinished`/`JobFailed`.
    pub async fn start(
        &self,
        machine_id: &str,
        start: StartJob,
    ) -> Result<mpsc::UnboundedReceiver<AgentMessage>, ExecError> {
        let (sub_tx, sub_rx) = mpsc::unbounded_channel();
        let tx = {
            let mut st = self.state.lock().await;
            let agent = st
                .agents
                .get(machine_id)
                .ok_or_else(|| ExecError::NotConnected(machine_id.to_owned()))?;
            let tx = agent.tx.clone();
            st.subs.insert(start.run_id.clone(), sub_tx);
            st.running_on.insert(start.run_id.clone(), machine_id.to_owned());
            tx
        };
        let run_id = start.run_id.clone();
        if tx
            .send(Ok(ServerMessage {
                msg: Some(server_message::Msg::Start(start)),
            }))
            .await
            .is_err()
        {
            let mut st = self.state.lock().await;
            st.subs.remove(&run_id);
            st.running_on.remove(&run_id);
            return Err(ExecError::NotConnected(machine_id.to_owned()));
        }
        Ok(sub_rx)
    }

    pub async fn cancel(
        &self,
        machine_id: &str,
        job_id: &str,
        save_checkpoint: bool,
    ) -> Result<(), ExecError> {
        self.send(
            machine_id,
            server_message::Msg::Cancel(CancelJob {
                job_id: job_id.to_owned(),
                save_checkpoint,
            }),
        )
        .await
    }

    pub async fn shutdown(&self, machine_id: &str, reason: &str) -> Result<(), ExecError> {
        self.send(
            machine_id,
            server_message::Msg::Shutdown(Shutdown {
                reason: reason.to_owned(),
                upgrade_to_version: None,
            }),
        )
        .await
    }

    async fn send(&self, machine_id: &str, m: server_message::Msg) -> Result<(), ExecError> {
        let tx = {
            let st = self.state.lock().await;
            st.agents
                .get(machine_id)
                .map(|a| a.tx.clone())
                .ok_or_else(|| ExecError::NotConnected(machine_id.to_owned()))?
        };
        tx.send(Ok(ServerMessage { msg: Some(m) }))
            .await
            .map_err(|_| ExecError::NotConnected(machine_id.to_owned()))
    }

    async fn route(&self, m: AgentMessage) {
        let Some(inner) = &m.msg else { return };
        let (run_id, terminal) = match inner {
            agent_message::Msg::Metrics(x) => (&x.run_id, false),
            agent_message::Msg::Logs(x) => (&x.run_id, false),
            agent_message::Msg::Progress(x) => (&x.run_id, false),
            agent_message::Msg::Checkpoint(x) => (&x.run_id, false),
            agent_message::Msg::Finished(x) => (&x.run_id, true),
            agent_message::Msg::Failed(x) => (&x.run_id, true),
            _ => return,
        };
        let run_id = run_id.clone();
        let mut st = self.state.lock().await;
        if let Some(sub) = st.subs.get(&run_id) {
            let _ = sub.send(m);
        }
        if terminal {
            st.subs.remove(&run_id);
            st.running_on.remove(&run_id);
        }
    }

    async fn disconnected(&self, machine_id: &str, session: u64) {
        let mut st = self.state.lock().await;
        if st.agents.get(machine_id).is_some_and(|a| a.session == session) {
            st.agents.remove(machine_id);
        }
        let orphaned: Vec<String> = st
            .running_on
            .iter()
            .filter(|(_, m)| m.as_str() == machine_id)
            .map(|(r, _)| r.clone())
            .collect();
        for run_id in orphaned {
            st.running_on.remove(&run_id);
            if let Some(sub) = st.subs.remove(&run_id) {
                let _ = sub.send(AgentMessage {
                    msg: Some(agent_message::Msg::Failed(JobFailed {
                        run_id,
                        reason: format!("agent `{machine_id}` disconnected"),
                        retryable: true,
                        ..Default::default()
                    })),
                });
            }
        }
        drop(st);
        self.changed.notify_waiters();
    }
}

#[derive(Debug)]
struct HubService {
    hub: Hub,
}

type SessionStream = Pin<Box<dyn Stream<Item = Result<ServerMessage, Status>> + Send>>;

#[tonic::async_trait]
impl AgentControl for HubService {
    type SessionStream = SessionStream;

    async fn session(
        &self,
        req: Request<Streaming<AgentMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = req.into_inner();
        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty session"))??;
        let Some(agent_message::Msg::Register(reg)) = first.msg else {
            return Err(Status::failed_precondition("first message must be Register"));
        };
        if reg.protocol_version != tempera_proto::PROTOCOL_VERSION {
            return Err(Status::failed_precondition(format!(
                "protocol version {} not supported (server speaks {})",
                reg.protocol_version,
                tempera_proto::PROTOCOL_VERSION
            )));
        }
        let machine_id = if reg.machine_id.is_empty() {
            "local".to_owned()
        } else {
            reg.machine_id.clone()
        };
        let (tx, rx) = mpsc::channel(1024);
        let session = {
            let mut st = self.hub.state.lock().await;
            st.next_session += 1;
            let session = st.next_session;
            st.agents.insert(
                machine_id.clone(),
                AgentConn {
                    tx: tx.clone(),
                    hardware: reg.hardware.unwrap_or_default(),
                    session,
                },
            );
            session
        };
        self.hub.changed.notify_waiters();
        tracing::info!(%machine_id, agent_version = %reg.agent_version, "agent registered");
        let _ = tx
            .send(Ok(ServerMessage {
                msg: Some(server_message::Msg::Registered(Registered {
                    session_token: format!("{machine_id}-{session}"),
                    heartbeat_interval_s: HEARTBEAT_INTERVAL_S,
                })),
            }))
            .await;

        let hub = self.hub.clone();
        tokio::spawn(async move {
            while let Some(Ok(m)) = inbound.next().await {
                hub.route(m).await;
            }
            tracing::info!(%machine_id, "agent disconnected");
            hub.disconnected(&machine_id, session).await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

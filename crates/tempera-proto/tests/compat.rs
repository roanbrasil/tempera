//! Protocol compatibility tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::pin::Pin;

use prost::Message;
use tempera_proto::PROTOCOL_VERSION;
use tempera_proto::v1::agent_control_client::AgentControlClient;
use tempera_proto::v1::agent_control_server::{AgentControl, AgentControlServer};
use tempera_proto::v1::{AgentMessage, Register, Registered, ServerMessage, agent_message, server_message};
use tokio_stream::{Stream, StreamExt, wrappers::TcpListenerStream};
use tonic::{Request, Response, Status, Streaming};

/// A v1 sender that knows a field the receiver does not must still decode.
#[test]
fn unknown_fields_are_ignored() {
    let reg = Register {
        protocol_version: 1,
        agent_version: "0.1.0".into(),
        ..Default::default()
    };
    let mut bytes = reg.encode_to_vec();
    // Field 99, wire type 2 (length-delimited), payload "future".
    bytes.extend_from_slice(&[0x9a, 0x06, 6]);
    bytes.extend_from_slice(b"future");
    let decoded = Register::decode(bytes.as_slice()).unwrap();
    assert_eq!(decoded.agent_version, "0.1.0");
}

struct EchoServer;

type ServerStream = Pin<Box<dyn Stream<Item = Result<ServerMessage, Status>> + Send>>;

#[tonic::async_trait]
impl AgentControl for EchoServer {
    type SessionStream = ServerStream;

    async fn session(
        &self,
        req: Request<Streaming<AgentMessage>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = req.into_inner();
        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty"))??;
        let Some(agent_message::Msg::Register(reg)) = first.msg else {
            return Err(Status::failed_precondition("first message must be Register"));
        };
        if reg.protocol_version != PROTOCOL_VERSION {
            return Err(Status::failed_precondition("unsupported protocol version"));
        }
        let reply = ServerMessage {
            msg: Some(server_message::Msg::Registered(Registered {
                session_token: format!("session-{}", reg.machine_id),
                heartbeat_interval_s: 10,
            })),
        };
        Ok(Response::new(Box::pin(tokio_stream::iter([Ok(reply)]))))
    }
}

#[tokio::test]
async fn client_and_server_handshake() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(AgentControlServer::new(EchoServer))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );

    let mut client = AgentControlClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let register = AgentMessage {
        msg: Some(agent_message::Msg::Register(Register {
            protocol_version: PROTOCOL_VERSION,
            machine_id: "m1".into(),
            ..Default::default()
        })),
    };
    let mut replies = client
        .session(tokio_stream::iter([register]))
        .await
        .unwrap()
        .into_inner();
    let first = replies.next().await.unwrap().unwrap();
    match first.msg {
        Some(server_message::Msg::Registered(r)) => assert_eq!(r.session_token, "session-m1"),
        other => panic!("unexpected {other:?}"),
    }
}

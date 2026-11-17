# Execution and Machines

## The `Executor` trait

```rust
#[async_trait]
pub trait Executor: Send + Sync {
    /// Describes hardware, price, and availability.
    async fn describe(&self) -> Result<MachineInfo>;
    /// Ensures the machine exists and is reachable (in the cloud case, rents it).
    async fn provision(&self, req: &ResourceRequest) -> Result<MachineHandle>;
    /// Installs/updates tempera-agent and starts it pointing at the server.
    async fn bootstrap_agent(&self, m: &MachineHandle, cfg: &AgentConfig) -> Result<()>;
    /// Releases the machine (in the cloud case, destroys the instance).
    async fn release(&self, m: &MachineHandle) -> Result<()>;
}
```

Implementations:

| Executor | Provisioning | Agent bootstrap | Release |
|---|---|---|---|
| `LocalExecutor` | None | Starts the agent as a child process | Kills the process |
| `SshExecutor` | None (existing machine) | `scp` of the binary (checks hash/version) + `ssh` to start it; reverse tunnel to the server | Stops the agent |
| `CloudExecutor<P: Provider>` | Provider API (creates the pod/instance) | Base image with the agent preinstalled **or** cloud-init | API: destroys the instance |

## The agent (`tempera-agent`)

A static Rust binary (musl), a few MB in size.

Responsibilities:

1. **Inventory:** CPU, RAM, GPUs (via NVML), driver/CUDA version, free disk. Sent on registration.
2. **Outbound connection** to the server via gRPC over TLS, which works behind NAT. Over SSH, the connection goes through SSH's own port forwarding, so no ports need to be opened.
3. **Environment preparation:**
   - Data: downloads from the server/S3 with caching by hash.
   - Python (only if the job asks for it): `uv` creates the venv from the worker's lockfile, cached by the lockfile hash. With a warm cache, this takes seconds.
   - Native backend: nothing to prepare.
4. **Execution:** starts the backend as a subprocess. The backend writes JSONL events to stdout (metrics, logs, checkpoint ready) and the agent forwards them.
5. **Heartbeat** every N seconds. If the server goes silent for too long, the agent keeps training and saving checkpoints locally, and syncs when the connection comes back.
6. **Checkpointing:** receives "checkpoint ready" from the backend and uploads it asynchronously.
7. **Hardware telemetry:** GPU usage, VRAM, temperature, and power, for the dashboard and to calibrate the ETA.
8. **Preemption signal:** when the provider warns that the spot instance is about to go down, it requests an emergency checkpoint.

## Machines on the local network (SSH)

Registration:

```yaml
machines:
  - name: living-room-pc
    kind: ssh
    host: 192.168.0.42
    user: bruno
    key: ~/.ssh/id_ed25519      # or the ssh-agent
    tags: [gpu, rtx3090]
```

- **Optional discovery** via mDNS: already-installed agents announce themselves on the LAN.
- Health checks and inventory are done at registration.
- Host prerequisites: Linux or macOS, plus the NVIDIA driver if there is a GPU. Python and Docker are **not** required: `uv` is downloaded on demand.
- Windows: support via WSL2 (later phase).

## Cloud (GPU via API key)

Providers in priority order (criteria: price, good API, spot):

| Provider | Why | Phase |
|---|---|---|
| **RunPod** | Simple API, cheap GPUs, spot ("community") | M6 |
| **Vast.ai** | Cheapest marketplace; requires a reliability filter | M6 |
| Lambda | Good on-demand pricing | post-M6 |
| AWS / GCP / Azure (spot) | For companies that already have accounts and credits | post-M6 |
| Any Kubernetes | Self-hosted clusters | future |

`Provider` trait:

```rust
#[async_trait]
pub trait Provider {
    async fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>>; // price, GPU, VRAM, spot?, region
    async fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<InstanceId>;
    async fn status(&self, id: &InstanceId) -> Result<InstanceStatus>;
    async fn terminate(&self, id: &InstanceId) -> Result<()>;
}
```

## Checkpointing and resumption

- Every iterative backend must support `save_checkpoint` and `resume_from` (it is a plugin requirement).
- Adaptive frequency: on spot, saves every ~5–10 min of work; on your own machine, per epoch.
- If the instance goes down, the server detects it (heartbeat or provider status), rents another compatible offer, and resumes from the last checkpoint. This is recorded in the run and shown in the UI.
- Configurable policy: `max_restarts` and `fallback_to_on_demand: true|false`.

## Cost guardrails

- **Per-job budget cap:** at 100%, saves a checkpoint and stops. Warns at 80%.
- **Automatic shutdown** of idle instances: no jobs for X minutes and the server destroys it.
- **Reaper:** a periodic job lists the open instances at each provider tagged `tempera` and terminates orphans. This protects against server crashes.
- **Cost reconciliation:** compares the cost calculated by Têmpera with what the provider reports.

## Scheduling (later phase)

- Per-machine job queue, with priority.
- Several small jobs on the same GPU (parallel HPO).
- Distributed multi-GPU training (DDP/FSDP via PyTorch) only after M7.

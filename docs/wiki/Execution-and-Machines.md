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
| `SshMachine` (SSH) | None (existing machine) | Streams the binary over the system `ssh` only when its sha256 differs, then starts it through `ssh -R` (reverse tunnel to the hub) | `Shutdown` to the agent, then the `ssh` process ends |
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

Têmpera drives the **system OpenSSH client** (`ssh`), not an embedded SSH library
([ADR-0007](../adr/0007-ssh-via-openssh.md)). Whatever works with `ssh host` works here:
`~/.ssh/config` aliases, `ProxyJump`, `ssh-agent`, hardware keys and `known_hosts`.
Every call adds `-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15`, so
nothing ever prompts. Host key checking is never disabled: connect once with plain `ssh`
to accept a new machine's key.

### Registration

`<TEMPERA_HOME>/machines.yaml` (managed by `tempera machines`, editable by hand):

```yaml
machines:
  - name: living-room-pc        # also the agent's machine id
    kind: ssh
    host: 192.168.0.42          # or a ~/.ssh/config alias
    user: bruno
    port: 22                    # optional
    identity: ~/.ssh/id_ed25519 # optional (`key` also accepted); ssh-agent/config otherwise
    remote_dir: ~/.tempera-agent  # optional, this is the default
    tags: [gpu, rtx3090]
```

```text
tempera machines add living-room-pc --host 192.168.0.42 --user bruno [--port P] [--identity F] [--tag gpu]
tempera machines list
tempera machines check living-room-pc [--worker-dir workers/python] [--install-uv]
tempera machines remove living-room-pc
```

`check` is a real round trip: inventory over `ssh`, agent install, agent started through
the tunnel against a temporary local hub, the `Register` hardware printed, agent stopped.

### What happens on the machine

Everything lives under `remote_dir` (default `~/.tempera-agent`):

| Path | Content |
|---|---|
| `bin/tempera-agent` | Uploaded only when the sha256 differs; verified remotely, then moved into place atomically |
| `cache/<sha256>.<ext>` | Content-addressed data cache: a second job with the same file transfers nothing |
| `workers/<name>/` | Synced Python worker project (`.venv`, `__pycache__`, `.pytest_cache` excluded; the remote `.venv` is kept so uv stays warm). Skipped when the content hash is unchanged |
| `runs/...` | Suggested place for run work dirs (the orchestrator chooses) |

The library API is `tempera_exec::SshMachine`:

| Method | Does |
|---|---|
| `check()` | OS, arch, libc, `uv`/`python3`, free disk, installed agent version (`RemoteInfo`) |
| `install_agent(local)` | Hash-checked upload; refuses a binary built for another OS/arch |
| `sync_worker(local_dir)` | `tar` over `ssh` with excludes, skipped when unchanged |
| `put(local)` / `put_cached(local)` | Upload into the data cache, returns the remote path (and whether it was uploaded) |
| `get(remote, local)` | Downloads a file, or a directory recursively |
| `exec(cmd)` | Runs a shell command (`sh -c`) and returns stdout |
| `start(hub, hub_addr, opts)` | All of the above, then `ssh -R 127.0.0.1:<port>:<hub>` running `tempera-agent connect --server http://127.0.0.1:<port> --machine-id <name> [--worker-dir ...]`; returns once the agent registered |

- **Tunnel:** the remote port is random in 20000..60000 and retried when busy
  (`ExitOnForwardFailure=yes`). The agent dials the hub through the SSH connection, so no
  port is opened on either side. If the connection drops, the agent's stream breaks and it
  exits; running jobs fail as retryable.
- **Python:** the agent runs Python backends with `uv run --project <remote worker dir>`.
  If `uv` is missing, `check()` reports it (`RemoteInfo::python_ready`) and `start()` with a
  worker dir fails clearly, unless `install_uv` is set: then uv is installed user-locally
  (`~/.local/bin`) with the official installer. The native backend needs nothing.
- **Running a job remotely:** `put()` the model-ready Parquet files, write the manifest
  with those remote paths and a remote `work_dir`, send `StartJob` to the machine id, then
  `get()` the work dir (model, predictions) back.
- **Optional discovery** via mDNS (#43): later.
- Host prerequisites: Linux or macOS with `sh`, `tar`, `sha256sum` or `shasum`, and sshd
  with `AllowTcpForwarding yes` (the default), plus the NVIDIA driver if there is a GPU.
  Python and Docker are **not** required: `uv` is downloaded on demand. The agent binary
  must match the machine's OS/arch (a glibc build runs on hosts with an equal or newer
  glibc; static musl builds come with #38).
- Windows: support via WSL2 (later phase).

## Cloud (GPU via API key)

> **Status (M6, preview).** The RunPod and Vast.ai clients were written from the providers' public
> API documentation and are covered by unit tests on recorded JSON shapes, but they **have not yet
> been exercised against a live account**. The full flow (plan → rent → SSH → train → terminate →
> report) is tested end to end with the `mock` provider, which boots a real sshd container.

Providers in priority order (criteria: price, good API, spot):

| Provider | Why | Status |
|---|---|---|
| **RunPod** | Simple API, cheap GPUs, spot ("community") | client implemented, unverified live |
| **Vast.ai** | Cheapest marketplace; requires a reliability filter | client implemented, unverified live |
| `mock` | CI and demos; optional Docker sshd instance, simulated preemption | tested end to end |
| Lambda | Good on-demand pricing | post-M6 |
| AWS / GCP / Azure (spot) | For companies that already have accounts and credits | post-M6 |
| Any Kubernetes | Self-hosted clusters | future |

### Using it

```bash
export RUNPOD_API_KEY=...        # or VAST_API_KEY; read from the environment only, never printed
tempera cloud offers --provider runpod --gpu 4090
tempera cloud estimate tempera.yaml   # compares machines, prints the savings line, rents nothing
```

```yaml
compute:
  allow: [runpod, local]   # `local` lets this machine compete (it costs $0)
  objective: cheapest      # cheapest | fastest | cheapest_under (with deadline)
  spot: true
budget:
  max_cost: { amount: 2, currency: USD }   # cloud caps are in USD
```

`tempera run` then:

1. **Plans:** lists offers (saved as a pricing snapshot), estimates ETA and cost on each machine,
   ranks them by `compute.objective` and prints the comparison with a one-line rationale. If this
   machine wins, nothing is rented. If the best estimate exceeds `budget.max_cost`, the job is
   refused before anything is rented.
2. **Rents:** writes a lease, launches the offer with a throwaway SSH key (named `tempera-<job>`),
   waits until SSH answers.
3. **Runs over SSH:** exactly the LAN-machine path (agent install, reverse tunnel, content-addressed
   data cache, results pulled back).
4. **Watches:** every 15 s the spend is metered (alert at 80% of the budget, cancel at 100%); every
   minute the lease heartbeat is refreshed and a vanished (preempted) instance cancels the job.
5. **Always terminates:** on success, on error, on cancel and on Ctrl-C. A killed process cannot
   clean up, so the **reaper** (`tempera cloud reap`, and every 4 minutes under `tempera up`)
   terminates any `tempera-*` instance without a fresh lease (TTL 10 min).

`Provider` trait (synchronous; called from `spawn_blocking`):

```rust
pub trait Provider: Send + Sync + Debug {
    fn name(&self) -> &str;
    fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>>; // price, GPU, VRAM, spot?, region
    fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<String>;
    fn status(&self, id: &str) -> Result<Instance>;
    fn instances(&self) -> Result<Vec<Instance>>;                       // for the reaper
    fn terminate(&self, id: &str) -> Result<()>;                         // idempotent
}
```

Not done yet: resuming on a new instance after preemption (#56, today the job is cancelled
cleanly), billing reconciliation with the provider, idle-instance shutdown for reused instances.

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

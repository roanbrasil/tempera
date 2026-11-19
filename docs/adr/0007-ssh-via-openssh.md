# ADR-0007: SSH through the system OpenSSH client instead of russh

- **Status:** Accepted
- **Date:** 2026-11-18

## Context
M4 lets any machine on the LAN act as a worker (issues #37, #38, #40, #41). The
server has to reach the machine, install or update `tempera-agent`, ship data and
the Python worker project, start the agent and give it a channel back to the hub
without opening ports. The original plan (#37) was to embed an SSH client with the
`russh` crate.

Users who have SSH access to their machines already have it configured:
`~/.ssh/config` host aliases, `ProxyJump` bastions, `ssh-agent` (often backed by a
hardware key or 1Password/KeePassXC), `known_hosts` with verified host keys,
`Match` blocks, certificates and per-host `IdentityFile`s. An embedded client would
have to re-implement or ignore each of these, and every gap shows up as "it works
with `ssh` but not with Têmpera".

## Decision
Use the system OpenSSH client, spawned with `tokio::process`:

- `ssh` for commands, file streaming (`cat` / `tar` over stdin/stdout) and the
  reverse tunnel (`ssh -R 127.0.0.1:<port>:127.0.0.1:<hub port>` running
  `tempera-agent connect --server http://127.0.0.1:<port>`).
- Every call passes `-o BatchMode=yes` (never prompt: the server is not
  interactive), `-o ConnectTimeout=10` and `-o ServerAliveInterval=15`. Command-line
  options win over `ssh_config`, so these cannot be weakened by configuration.
- Host key checking is **never** disabled by Têmpera. Unknown hosts fail with
  OpenSSH's own message; the user connects once with `ssh` to accept the key.
  Extra `-o` options per machine exist (used by the tests for a temporary
  `known_hosts`), but nothing sets `StrictHostKeyChecking=no` by default.
- Remote commands always run as `sh -c <script>` with every interpolated value
  shell-quoted, so the user's login shell (bash, zsh, fish) does not matter.
- The tunnel uses a random remote port in 20000..60000 with
  `ExitOnForwardFailure=yes`, retried when the port is taken (`-R 0` is not
  portable across OpenSSH versions for this use).
- Directory transfers use `tar` piped over `ssh` rather than `rsync`/`scp`: it needs
  nothing beyond POSIX tools on the remote side, and change detection is done by a
  content hash so unchanged trees are not re-sent.

## Alternatives considered
- **russh (embedded client):** one static binary with no external dependency,
  fine-grained control of channels and errors, and testable without a `ssh`
  binary. But it does not read `~/.ssh/config`, so it misses aliases, `ProxyJump`,
  `Match` and certificates unless we re-implement them; agent forwarding and
  hardware-backed keys need extra work; host key verification and `known_hosts`
  hashing must be re-implemented correctly (a security-sensitive surface we would
  own). It also adds a sizeable crypto dependency tree to audit.
- **libssh2 / ssh2 crate:** C dependency, same config-file gaps as russh, weaker
  async story.
- **`scp`/`rsync` for every transfer:** `scp` switched to the SFTP protocol in
  OpenSSH 9 with different path semantics, and `rsync` must also be installed on
  the remote side and needs its own `-e` quoting of the ssh options. Streaming
  through the same `ssh` invocation avoids both.

## Consequences
- Positive: zero SSH configuration inside Têmpera; whatever works with `ssh host`
  works here, including bastions and hardware keys. No crypto code of our own.
  Errors are OpenSSH's, which users already know how to read.
- Negative: requires an `ssh` client on the server host (present on Linux and
  macOS; Windows 10+ ships OpenSSH). Behaviour depends on the installed OpenSSH
  version; error handling parses stderr in a few places (port-forwarding failure).
  Each operation is a new connection; users can enable `ControlMaster` in their
  config to multiplex.
- Security: the remote end of the tunnel listens on the worker's loopback, so other
  local users on that machine could reach the hub; per-agent tokens (#44) and TLS
  close that gap. The agent binary is verified by sha256 after upload; minisign
  signatures (#38) remain to be added.
- Necessary: an integration test against a real `sshd` (Docker) in CI, and docs
  telling users to accept host keys with a manual `ssh` first.

# Security and Privacy

## Threat model (summary)

| Asset | Threat | Mitigation |
|---|---|---|
| User data | Leakage to third parties | Local-first; the SLM runs locally; remote API use only with explicit opt-in and a warning |
| Cloud API keys | Theft, misuse | OS keyring or encrypted file (age); never in logs or in `tempera.yaml` |
| SSH keys | Compromise of LAN machines | Uses `ssh-agent`; never copies the private key; prefers a dedicated key with restricted access |
| Agent↔server channel | MITM | TLS (rustls) with the server certificate pinned at bootstrap; single-use agent token that becomes a session credential |
| Downloaded pretrained models | Malicious code (pickle) | Only **safetensors**/ONNX; `.bin`/pickle blocked by default |
| Plugin code | Arbitrary execution | Plugins are trusted code (installed by the user); warn on installation |
| Cloud instances | Residual data | Destroy the instance at the end; ephemeral disk; option to encrypt data in transit and at rest |

## Secrets

- `tempera secrets set runpod.api_key`: stores it in the OS keyring (`keyring` crate).
- Fallback on servers without a keyring: a file encrypted with `age`, whose key comes from an environment variable.
- Secrets are injected into the agent only when needed, as environment variables of the backend process, and are never written to disk.
- Logs go through automatic redaction of secret patterns.

## LGPD / GDPR / personal data

- Audit check A11 detects personal data (CPF, SSN, email, phone, names) and suggests masking it.
- The final report lists which personal columns were used, removed, or masked, which serves as evidence of data minimization.
- **"Nothing leaves here"** mode: blocks cloud executors and remote SLMs for an entire project.
- Retention: a command to delete the dataset, splits, checkpoints, and derived artifacts (right to erasure).

## Authentication (team server mode)

- Phase 1: single-user, bound to `localhost`.
- Phase 2: local users (argon2) + API tokens; optional OIDC.
- Roles: admin (machines and secrets) and member (jobs).

## Supply chain

- `cargo deny` (licenses and advisories) and `cargo audit` in CI.
- Python worker lockfile with hashes (`uv lock`).
- Signed agent binary (minisign); the executor verifies the signature before installing it on the worker.

# Security and Privacy

This page separates what Têmpera **does today** from what is **planned**. Each planned item names
its issue. When a control moves from planned to implemented, the row links to the code or test that
proves it.

Status legend: ✅ implemented · 🟡 partial · ⏳ planned.

## What is protected today

| Asset | Threat | Today | Status |
|---|---|---|---|
| User data | Leaves the machine | Local-first: data is read and trained on where it is. Nothing is sent anywhere unless the spec rents a cloud machine or points the advisor at a remote endpoint. | 🟡 |
| User data → SLM advisor | Column names and sample values sent to a remote LLM | Refused unless `audit.advisor.remote: true`. The endpoint URL is parsed (WHATWG rules); userinfo is rejected; only loopback counts as local; no redirects (`crates/tempera-audit/src/advisor.rs`, tests `only_local_urls_without_opt_in`). Sample values that look like personal data are replaced by their kind, even locally (`prompt_never_contains_personal_values`). | ✅ |
| User data → cloud machine | Personal data uploaded to a rented machine, possibly abroad | **Not gated.** A personal-data finding (A11) is yellow and does not stop a cloud job. | ⏳ #75 |
| HTTP API (`tempera up`) | Another local process or a web page drives the API | Binds to `127.0.0.1` by default. **No token, no Host/Origin check**, so DNS rebinding and cross-site requests are possible. `--host 0.0.0.0` exposes it without a warning. | ⏳ #83 |
| Agent ↔ server channel | Impersonation, eavesdropping | Plain gRPC on loopback; on SSH machines it rides an `ssh -R` tunnel to a loopback port on the remote host. **The agent token is not checked**, and any local user on either end can connect. No TLS. | ⏳ #83 |
| Cloud API keys | Theft, misuse | Read from environment variables only (`RUNPOD_API_KEY`, `VAST_API_KEY`), never written to `tempera.yaml`. **They are inherited by the local agent and by training code.** | ⏳ #44, #95 |
| SSH keys | Compromise of LAN machines | System OpenSSH with `BatchMode`; Têmpera never copies a private key. Rented machines get a throwaway per-job key, which is **not deleted** afterwards. | 🟡 #95 |
| Rented machines | Left billing; killed by mistake | Lease before launch, heartbeat, reaper scoped to this installation's owner id, fail-closed ledger (`crates/tempera-cloud/src/guard.rs`). Not guaranteed when the process is killed or the terminal closed. | 🟡 #88 |
| Python worker code | Running code from an untrusted folder | **The worker project is also searched in the current directory's ancestors**, so a cloned folder containing `workers/python` would be executed. | ⏳ #94 |
| Downloaded pretrained models | Malicious pickle | Fine-tuning loads Hugging Face models with `safetensors` when available; there is no block on pickle formats yet. | ⏳ #51 |
| Workspace files | Other users on the machine read cached data | The workspace follows the umask; no permissions are set. Profiles store the most frequent raw values of every column. | ⏳ #76 |

## Secrets

- Today: provider API keys come from environment variables. They are not logged by Têmpera, but
  there is no redaction layer.
- Planned (#44): OS keyring with an `age`-encrypted file as fallback, `tempera secrets set`.
- Planned (#95): an environment allow-list for the agent and the backend, so training code never
  sees provider keys.

## Personal data (PIPEDA, LGPD, GDPR)

- ✅ A11 flags columns where at least 30% of the first 2,000 values look like e-mail, phone, CPF,
  CNPJ, US SSN, card number or IP address. It never echoes the values.
  - ⏳ #77: a random sample instead of the first rows, values inside free text, Canadian SIN and
    postal codes, names, addresses and birth dates.
- ✅ T4 flags personal data inside agent traces before fine-tuning.
- ✅ Sensitive attributes (`data.sensitive`) are used for the fairness report and are **not** model
  features unless listed in `data.sensitive_as_feature`, which is recorded as a finding.
- ⏳ #75: dataset classification, a project policy (`egress: local_only | lan | cloud`, allowed
  regions), a gate on every path that moves data off the machine, and an egress log as
  accountability evidence. Cross-border transfer is where PIPEDA (accountability, safeguards),
  Quebec Law 25, LGPD art. 33 and GDPR chapter V require a decision and a record.
- ⏳ #76: `tempera forget` (erasure of everything derived from a dataset), retention, workspace
  permissions.
- ⏳ #81: a model card that lists the personal-data classification and the egress records.

Têmpera provides controls and evidence. It does not give legal advice, manage consent or write
privacy impact assessments.

## Authentication

- Today: single user, loopback only, no authentication (see the table above).
- ⏳ #83: a local token, Host/Origin validation, and a refusal to bind to a non-loopback address
  without TLS or an explicit flag.
- ⏳ #60: team mode with users, roles and optional OIDC.

## Supply chain

- ✅ `cargo deny` in CI: licenses, advisories, bans, sources (two documented ignores).
- ✅ Python worker locked with `uv.lock`.
- ⏳ #98: release workflow with tests as a gate, actions pinned by SHA, least-privilege tokens and
  build provenance.
- ⏳ #38: a signed agent binary verified before it is installed on a worker.
- ⏳ #97: `pip-audit` and `npm audit` in CI.

## Reporting a vulnerability

Open a [GitHub security advisory](https://github.com/roanbrasil/tempera/security/advisories/new)
rather than a public issue.

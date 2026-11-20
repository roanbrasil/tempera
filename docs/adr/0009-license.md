# ADR 0009: License: MIT OR Apache-2.0

- Status: accepted
- Date: 2026-11-20
- Issue: #9

## Context

The repository and the v0.1.0 binaries are public, but without a license all rights were reserved:
nobody could legally use, modify or redistribute Têmpera. Options considered in #9:

| Option | Adoption | Protection against a hosted competitor | Fit |
|---|---|---|---|
| MIT OR Apache-2.0 | Highest; the Rust ecosystem default; companies approve it without review | None | Local-first CLI, agents, contributors |
| Apache-2.0 only | High; explicit patent grant | None | Same, slightly less compatible with GPLv2 projects |
| AGPL-3.0 | Lower; many companies forbid it | Strong for network use | Weak: Têmpera runs mostly on the user's machine, where AGPL adds friction and no protection |
| BSL/FSL (source-available) | Lower; not OSI open source; deters contributors | Strong for a period | Premature: no hosted product exists yet |

Têmpera's value today is adoption: people running it on their own data and machines, agents
calling it over MCP, and contributors adding checks and algorithms. Nothing is hosted, so there is
nothing a copyleft or source-available license would protect yet.

## Decision

Dual license **MIT OR Apache-2.0**, at the user's option, for all code in this repository
(Rust crates, Python worker, web UI, examples). This is the Rust ecosystem convention: MIT for
maximum compatibility, Apache-2.0 for its explicit patent grant.

Contributions are accepted under the same dual license (inbound = outbound, stated in
CONTRIBUTING); no CLA.

## Business model (initial)

Open core. The local product (CLI, web UI, audit, training, SSH machines, cloud rental with the
user's own keys, MCP server) stays open under this license. Paid offerings, if any, will be
separate products that do not restrict the open core: a team server (auth, roles, shared
history; M7), managed GPU capacity billed at a margin, and support. Should that later need
protection, it would live in a separate repository under its own license; this ADR does not
allow relicensing what is already released under MIT OR Apache-2.0.

## Consequences

- `LICENSE-MIT` and `LICENSE-APACHE` at the root; `license = "MIT OR Apache-2.0"` in the
  workspace, the Python worker and the UI; release archives ship both files.
- cargo-deny's `[licenses.private] ignore` is no longer needed for unpublished workspace crates,
  but stays because they are not published to crates.io.
- Third-party dependencies remain under their own licenses (checked by cargo-deny).

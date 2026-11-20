# Contributing

## Workflow

1. Pick an issue from the current milestone (see the [Roadmap](Roadmap.md)) and assign it to yourself.
2. Create the branch: `feat/<number>-short-description` or `fix/<number>-...`.
3. Commits follow **Conventional Commits** (`feat(exec): agent ssh bootstrap`).
4. Open the PR referencing the issue (`Closes #N`) and fill in the template.
5. CI must be green: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, `cargo deny`, `uv run pytest`, `ruff`.

## Standards

### Rust
- `thiserror` in library crates and `anyhow` only in binaries.
- No `unwrap()` outside of tests.
- Every public crate function has a doc comment.
- Async with `tokio`; no blocking on the runtime (use `spawn_blocking` for Polars or heavy work).

### Python
- `ruff` (lint + format) and type checking with `pyright` in basic mode.
- One adapter per algorithm, following the `Algorithm` protocol ([Contracts](Contracts.md)).
- No global dependencies: everything goes through `uv` optional groups.

### Tests
- Unit tests in each crate/package.
- **Contract test** for every algorithm: small fixed dataset → minimum metric, checkpoint/resumption, and export.
- SSH integration via an `sshd` container in CI.
- Audit and ETA benchmarks run nightly.

## Adding an algorithm

1. Create the manifest at `algorithms/<name>/algorithm.yaml`.
2. Implement the adapter (Python or Rust).
3. Add the contract test.
4. Update the [Catalog](Algorithm-Catalog.md).

## Adding a cloud provider

1. Implement the `Provider` trait in `tempera-exec/src/cloud/<provider>.rs`.
2. Implement price syncing.
3. Implement instance tagging for the reaper.
4. Add the mock for CI and document the manual testing instructions.

## Decisions

Changed something structural? Write an ADR in `docs/adr/` (see the [template](../adr/template.md)).

## License of contributions

Têmpera is dual licensed MIT OR Apache-2.0 ([ADR 0009](../adr/0009-license.md)). Unless you
state otherwise, any contribution you submit is licensed the same way, without additional terms.
There is no CLA.

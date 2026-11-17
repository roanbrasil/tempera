# ADR-0005: SQLite + local FS first, behind traits

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
The first audience is single-user/local; team mode comes later.

## Decision
- Metadata and metrics in SQLite (`sqlx`); artifacts via `object_store` (local FS by default).
- Access goes through traits (`RunStore`, `ArtifactStore`) so they can be swapped for Postgres/S3/R2.

## Consequences
- (+) Zero infrastructure dependencies to get started.
- (−) High-frequency metrics require batching; re-evaluate DuckDB/ClickHouse if it grows.

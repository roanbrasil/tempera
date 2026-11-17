# ADR-0005: SQLite + FS local primeiro, atrás de traits

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
O primeiro público é single-user/local; o modo de time vem depois.

## Decisão
- Metadados e métricas em SQLite (`sqlx`); artefatos via `object_store` (FS local por padrão).
- O acesso passa por traits (`RunStore`, `ArtifactStore`) para trocar por Postgres/S3/R2.

## Consequências
- (+) Zero dependência de infra para começar.
- (−) Métricas de alta frequência exigem batching; reavaliar DuckDB/ClickHouse se crescer.

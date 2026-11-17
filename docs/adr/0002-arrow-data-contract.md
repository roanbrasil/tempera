# ADR-0002: Apache Arrow/Parquet as the data contract

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Data crosses between Rust (Polars) and Python (PyArrow, pandas, PyTorch), and serializing CSV/JSON across that boundary is slow and loses types.

## Decision
Every dataset is normalized to Parquet on disk and Arrow in memory. Splits are separate Parquet files, addressed by hash.

## Alternatives considered
- CSV: loses types and is slow.
- Custom format: no ecosystem.
- Lance: interesting for multimodal data; re-evaluate after M5.

## Consequences
- (+) Zero-copy/low-copy, preserved types, compression.
- (−) Blobs (images) live outside Parquet, referenced by hash.

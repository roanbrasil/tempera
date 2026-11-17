# ADR-0002: Apache Arrow/Parquet como contrato de dados

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
Os dados atravessam Rust (Polars) e Python (PyArrow, pandas, PyTorch), e serializar CSV/JSON nessa fronteira é lento e perde tipos.

## Decisão
Todo dataset é normalizado para Parquet em disco, e Arrow em memória. Os splits são arquivos Parquet separados, endereçados por hash.

## Alternativas consideradas
- CSV: perde tipos e é lento.
- Formato próprio: sem ecossistema.
- Lance: interessante para dados multimodais; reavaliar depois do M5.

## Consequências
- (+) Zero-copy/baixa cópia, tipos preservados, compressão.
- (−) Blobs (imagens) ficam fora do Parquet, referenciados por hash.

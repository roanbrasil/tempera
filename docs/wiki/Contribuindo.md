# Contribuindo

## Fluxo de trabalho

1. Pegue uma issue do milestone atual (ver o [Roadmap](Roadmap.md)) e atribua a você.
2. Crie a branch: `feat/<numero>-descricao-curta` ou `fix/<numero>-...`.
3. Commits no padrão **Conventional Commits** (`feat(exec): ssh bootstrap do agente`).
4. Abra o PR referenciando a issue (`Closes #N`) e preencha o template.
5. A CI precisa estar verde: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, `cargo deny`, `uv run pytest`, `ruff`.

## Padrões

### Rust
- `thiserror` em crates de biblioteca e `anyhow` só em binários.
- Nada de `unwrap()` fora de testes.
- Toda função pública de crate com doc comment.
- Async com `tokio`; nada de bloqueio no runtime (use `spawn_blocking` para Polars ou trabalho pesado).

### Python
- `ruff` (lint + format) e tipagem com `pyright` em modo básico.
- Um adaptador por algoritmo, seguindo o protocolo `Algorithm` ([Contratos](Contratos.md)).
- Sem dependências globais: tudo via grupos opcionais do `uv`.

### Testes
- Unidade em cada crate/pacote.
- **Teste de contrato** para todo algoritmo: dataset pequeno fixo → métrica mínima, checkpoint/retomada e export.
- Integração SSH via container `sshd` na CI.
- Benchmarks de auditoria e de ETA rodam toda noite.

## Adicionando um algoritmo

1. Crie o manifesto em `algorithms/<nome>/algorithm.yaml`.
2. Implemente o adaptador (Python ou Rust).
3. Adicione o teste de contrato.
4. Atualize o [Catálogo](Catalogo-de-Algoritmos.md).

## Adicionando um provedor de nuvem

1. Implemente a trait `Provider` em `tempera-exec/src/cloud/<provedor>.rs`.
2. Implemente a sincronização de preços.
3. Implemente a marcação (tag) de instâncias para o reaper.
4. Adicione o mock para a CI e documente as instruções de teste manual.

## Decisões

Mudou algo estrutural? Escreva um ADR em `docs/adr/` (ver o [modelo](../adr/template.md)).

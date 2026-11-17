# ADR-0006: UI em SvelteKit estático embutido no binário

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
Queremos uma UI rica (gráficos ao vivo, formulários guiados) sem perder o binário único.

## Decisão
SvelteKit com `adapter-static`; o build é embutido no `tempera-server` via `rust-embed`. Gráficos com uPlot (ao vivo) e ECharts (relatórios).

## Alternativas consideradas
- Leptos/Dioxus (Rust): uma linguagem só, mas ecossistema de componentes menor e iteração mais lenta.
- Tauri desktop: possível no futuro **reaproveitando** a mesma UI.
- React: viável; Svelte é mais leve para este escopo.

## Consequências
- (+) Iteração rápida de UI e binário único mantido.
- (−) Uma terceira linguagem (TypeScript) no repositório.

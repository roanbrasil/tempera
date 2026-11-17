# ADR-0006: Static SvelteKit UI embedded in the binary

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
We want a rich UI (live charts, guided forms) without giving up the single binary.

## Decision
SvelteKit with `adapter-static`; the build is embedded in `tempera-server` via `rust-embed`. Charts with uPlot (live) and ECharts (reports).

## Alternatives considered
- Leptos/Dioxus (Rust): a single language, but a smaller component ecosystem and slower iteration.
- Tauri desktop: possible in the future, **reusing** the same UI.
- React: viable; Svelte is lighter for this scope.

## Consequences
- (+) Fast UI iteration and the single binary is preserved.
- (−) A third language (TypeScript) in the repository.

# Têmpera web UI

SvelteKit (Svelte 5 runes) + TypeScript, built as a static SPA (`@sveltejs/adapter-static`, fallback
`index.html`) that `tempera-server` embeds from `ui/build/` with `rust-embed` and serves from `/`.
Live curves use uPlot. The HTTP/WebSocket contract is `docs/wiki/Contracts.md` §9.

Requires Node ≥ 22.17 (`.nvmrc`).

```sh
npm install
npm run dev:mock   # simulated API + live WebSocket stream (no Rust server needed)
npm run dev        # proxies /api (and the WebSocket) to http://127.0.0.1:7070 (`tempera up`)
                   # override with TEMPERA_API=http://host:port
npm run build      # → build/ (static; embedded by the Rust binary)
npm run check      # svelte-check
npm run lint       # prettier + eslint
npm test           # vitest (event reducer, reconnect replay, YAML edit helpers)
```

`MOCK_SPEED=3 npm run dev:mock` speeds the simulated training up.

## Layout

| Path                                  | What                                                                             |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| `src/routes/+page.svelte`             | `/` job list                                                                     |
| `src/routes/new/+page.svelte`         | `/new` guided journey: data → goal → audit → strategy → machine → review/train   |
| `src/routes/jobs/[id]/+page.svelte`   | `/jobs/{id}` live job: stages, runs, curves, logs, cancel, winner, report        |
| `src/lib/api.ts`, `src/lib/types.ts`  | typed client and the serde shapes of the Rust types                              |
| `src/lib/events.ts`                   | live event types, the reducer and the reconnect/replay logic                     |
| `src/lib/job-stream.svelte.ts`        | WebSocket with backoff; one UI update per animation frame                        |
| `src/lib/spec-yaml.ts`                | comment-preserving edits of `tempera.yaml` (`data.drop`, `audit.acknowledge`, …) |
| `src/lib/journey/`                    | the journey steps and their state                                                |
| `src/lib/job/`, `src/lib/components/` | job screen panels, chart, badges                                                 |
| `mock/`                               | dev-only Vite plugin simulating the API (CSV profiler, audit, training jobs)     |

Screenshots: `docs/`.

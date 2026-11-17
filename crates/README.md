# Crates

```mermaid
flowchart BT
  core[tempera-core<br/>spec, events, errors]
  proto[tempera-proto<br/>gRPC types]
  data[tempera-data] --> core
  audit[tempera-audit] --> core
  audit --> data
  cost[tempera-cost] --> core
  track[tempera-track] --> core
  native[tempera-native] --> core
  serve[tempera-serve] --> core
  exec[tempera-exec] --> core
  exec --> proto
  agent[tempera-agent<br/>bin] --> core
  agent --> proto
  agent --> native
  server[tempera / tempera-server<br/>bin] --> core
  server --> proto
  server --> data & audit & cost & track & exec & serve
```

Arrows show the **planned** dependencies (`A --> B` means A depends on B). `tempera-core` is the base and
depends on no other workspace crate. Today only `core` and `proto` have real code; the others are
placeholders with their responsibility documented in `src/lib.rs`.

| Crate | Status |
|---|---|
| `tempera-core` | ✅ `tempera.yaml` types + validation, JSONL events, schema generation |
| `tempera-proto` | ✅ gRPC protocol v1 (compiled with protox, no `protoc` needed) |
| `tempera-server` (bin `tempera`) | 🟡 `tempera validate`, `tempera schema` |
| `tempera-agent` | 🟡 skeleton |
| others | ⬜ planned (see milestones) |

"""JSONL events sent to the agent on stdout.

The format is defined by the Rust types in `tempera-core` (`schemas/event.schema.json`).
Anything else a library prints goes to stderr, never stdout, so the event stream stays clean.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from typing import IO, Any, Literal

LogLevel = Literal["debug", "info", "warn", "error"]
ArtifactKind = Literal["model", "report", "predictions", "other"]
ExportFormat = Literal["onnx", "safetensors", "native"]


class EventSink:
    """Writes one JSON object per line. Thread-safe; flushes every event."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def metric(self, name: str, value: float, step: int | None = None, epoch: int | None = None) -> None:
        if not math.isfinite(value):
            # NaN/inf are not valid JSON; report them as a warning instead.
            self.log(f"non-finite metric {name}={value} at step {step}", level="warn")
            return
        ev: dict[str, Any] = {"t": "metric", "name": name, "value": float(value), "ts": time.time()}
        if step is not None:
            ev["step"] = step
        if epoch is not None:
            ev["epoch"] = epoch
        self.emit(ev)

    def log(self, msg: str, level: LogLevel = "info") -> None:
        self.emit({"t": "log", "level": level, "msg": msg})

    def progress(self, step: int, total_steps: int | None = None) -> None:
        ev: dict[str, Any] = {"t": "progress", "step": step}
        if total_steps is not None:
            ev["total_steps"] = total_steps
        self.emit(ev)

    def checkpoint(self, path: str, step: int) -> None:
        self.emit({"t": "checkpoint", "path": path, "step": step})

    def artifact(self, kind: ArtifactKind, path: str, fmt: ExportFormat | None = None) -> None:
        ev: dict[str, Any] = {"t": "artifact", "kind": kind, "path": path}
        if fmt is not None:
            ev["format"] = fmt
        self.emit(ev)

    def done(self, summary: dict[str, float] | None = None) -> None:
        clean = {k: float(v) for k, v in (summary or {}).items() if math.isfinite(v)}
        self.emit({"t": "done", "summary": clean})

    def error(self, msg: str, retryable: bool = False) -> None:
        self.emit({"t": "error", "msg": msg, "retryable": retryable})

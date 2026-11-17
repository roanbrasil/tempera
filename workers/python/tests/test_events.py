"""Contract tests: Python events must validate against the schema generated from Rust."""

from __future__ import annotations

import io

from helpers import validate_lines

from tempera_worker.events import EventSink


def test_every_event_kind_matches_schema() -> None:
    buf = io.StringIO()
    sink = EventSink(buf)
    sink.metric("loss", 0.5, step=1, epoch=0)
    sink.log("hello", level="warn")
    sink.progress(1, 10)
    sink.checkpoint("/tmp/ckpt", 1)
    sink.artifact("model", "/tmp/m.onnx", "onnx")
    sink.done({"val/auc": 0.9})
    sink.error("boom", retryable=True)
    assert [e["t"] for e in validate_lines(buf.getvalue())] == [
        "metric",
        "log",
        "progress",
        "checkpoint",
        "artifact",
        "done",
        "error",
    ]


def test_non_finite_metric_becomes_warning() -> None:
    buf = io.StringIO()
    EventSink(buf).metric("loss", float("nan"), step=3)
    (ev,) = validate_lines(buf.getvalue())
    assert ev["t"] == "log" and ev["level"] == "warn"

"""Contract tests: Python events must validate against the schema generated from Rust."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from tempera_worker.events import EventSink

SCHEMA = json.loads((Path(__file__).parents[3] / "schemas" / "event.schema.json").read_text())


def _validate_lines(text: str) -> list[dict]:
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    for ev in events:
        jsonschema.validate(ev, SCHEMA)
    return events


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
    assert [e["t"] for e in _validate_lines(buf.getvalue())] == [
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
    (ev,) = _validate_lines(buf.getvalue())
    assert ev["t"] == "log" and ev["level"] == "warn"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tempera_worker", "run", *args], capture_output=True, text=True, timeout=60
    )


def test_cli_dummy_run_emits_valid_stream(tmp_path: Path) -> None:
    proc = _run("--algorithm", "dummy", "--params", '{"steps": 5}', "--work-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    events = _validate_lines(proc.stdout)
    assert events[-1]["t"] == "done"
    assert sum(e["t"] == "metric" for e in events) == 6


def test_cli_resume_from_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "state.json").write_text('{"step": 4, "loss": 0.3}')
    proc = _run(
        "--algorithm",
        "dummy",
        "--params",
        '{"steps": 5}',
        "--resume-from",
        str(ckpt),
        "--work-dir",
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    steps = [e["step"] for e in _validate_lines(proc.stdout) if e["t"] == "progress"]
    assert steps == [5]


@pytest.mark.parametrize("algo", ["does-not-exist"])
def test_unknown_algorithm_reports_error(algo: str) -> None:
    proc = _run("--algorithm", algo)
    assert proc.returncode == 2
    assert _validate_lines(proc.stdout)[-1]["t"] == "error"

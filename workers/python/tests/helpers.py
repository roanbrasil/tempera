"""Test helpers shared by the core and tabular test modules (no numpy needed)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA = json.loads((Path(__file__).parents[3] / "schemas" / "event.schema.json").read_text())

# The CI runners are small and shared; keep native libraries from oversubscribing CPUs.
ENV = {**os.environ, "TEMPERA_NUM_THREADS": "2"}


def validate_lines(text: str) -> list[dict[str, Any]]:
    """Every stdout line must be a JSON event valid against schemas/event.schema.json."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        ev = json.loads(line)
        jsonschema.validate(ev, SCHEMA)
        events.append(ev)
    return events


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.write_text(json.dumps(manifest))
    return path


def run_cli(*args: str, timeout: float = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tempera_worker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=ENV,
    )


def run_manifest(
    command: str, manifest: dict[str, Any], path: Path
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    proc = run_cli(command, "--manifest", str(write_manifest(path, manifest)))
    return proc, validate_lines(proc.stdout)

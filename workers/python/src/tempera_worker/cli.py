"""`python -m tempera_worker run ...`: entry point the agent invokes."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from tempera_worker import algorithms
from tempera_worker.context import RunContext
from tempera_worker.events import EventSink


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tempera-worker")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train one run and stream JSONL events")
    run.add_argument("--algorithm", required=True)
    run.add_argument("--params", default="{}", help="hyperparameters as JSON")
    run.add_argument("--spec", type=Path, help="tempera.yaml (kept for reproducibility)")
    run.add_argument("--work-dir", type=Path)
    run.add_argument("--resume-from", type=Path)
    run.add_argument("--events", choices=["stdout"], default="stdout")
    sub.add_parser("list", help="list available algorithms")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for name in sorted(algorithms.REGISTRY):
            print(name)
        return 0

    # Only events go to stdout; route stray prints from libraries to stderr.
    events = EventSink(sys.stdout)
    sys.stdout = sys.stderr

    try:
        params = json.loads(args.params)
        work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="tempera-"))
        ctx = RunContext(events=events, work_dir=work_dir, params=params, resume_from=args.resume_from)
        ctx.install_signal_handlers()
        algo = algorithms.get(args.algorithm)()
        if args.resume_from:
            algo.resume_from(args.resume_from)
            events.log(f"resumed from {args.resume_from}")
        algo.fit(None, None, ctx)
        if ctx.should_stop():
            events.error("stopped before completion; checkpoint saved", retryable=True)
            return 3
        model = algo.export(work_dir / "out", "native")
        events.artifact("model", str(model), "native")
        events.done({})
        return 0
    except SystemExit as e:
        events.error(str(e))
        return 2
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        events.error(f"{type(e).__name__}: {e}")
        return 1

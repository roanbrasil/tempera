"""`python -m tempera_worker fit|predict|quality|fairness --manifest PATH`: entry point the agent invokes.

Contract: docs/wiki/Contracts.md, sections 7 (supervised) and 8 (clustering, anomaly detection);
`quality` (label issues, Data-Audit A7) and `fairness` (Data-Audit A8) need the `quality` extra;
`finetune` (LoRA/QLoRA on chat traces, section 9) needs the `llm` extra.
stdout carries only JSONL events (section 3); everything else (Python prints, native library output)
is routed to stderr.

Exit codes: 0 success (`done` emitted) · 1 unexpected error · 2 bad manifest / unknown algorithm ·
3 stopped by signal (checkpoint saved when supported).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import IO, TYPE_CHECKING

from tempera_worker import algorithms
from tempera_worker.context import RunContext
from tempera_worker.events import EventSink
from tempera_worker.manifest import (
    SUPERVISED_TASKS,
    TARGET_COLUMN,
    FairnessManifest,
    Feature,
    FitManifest,
    ManifestError,
    PredictManifest,
    QualityManifest,
    load_fairness,
    load_fit,
    load_predict,
    load_quality,
)
from tempera_worker.protocol import MODEL_DESCRIPTOR, Evaluates

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.protocol import Algorithm

EXIT_OK, EXIT_ERROR, EXIT_BAD_MANIFEST, EXIT_STOPPED = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tempera-worker")
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="train, save the model, then predict every `predict` entry")
    fit.add_argument("--manifest", type=Path, required=True)
    predict = sub.add_parser("predict", help="load a saved model and predict one input")
    predict.add_argument("--manifest", type=Path, required=True)
    quality = sub.add_parser("quality", help="find suspected label errors (Cleanlab confident learning)")
    quality.add_argument("--manifest", type=Path, required=True)
    fairness = sub.add_parser("fairness", help="per-group fairness metrics of test predictions (Fairlearn)")
    fairness.add_argument("--manifest", type=Path, required=True)
    finetune = sub.add_parser("finetune", help="LoRA/QLoRA fine-tuning of a chat LLM on agent traces")
    finetune.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("list", help="list available algorithms")
    return p


def _event_stream() -> IO[str]:
    """Reserve the real stdout for events and point fd 1 (and `sys.stdout`) at stderr.

    Native libraries (LightGBM, XGBoost) write straight to fd 1, so redirecting `sys.stdout` alone
    would not keep the event stream clean.
    """
    sys.stdout.flush()
    events_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(events_fd, "w", buffering=1, encoding="utf-8")


def read_table(path: Path, features: list[Feature], with_target: bool) -> pa.Table:
    """Read only the feature columns (and `__target__`) of a model-ready Parquet file.

    Unsupervised tasks call this with `with_target=False`: `__target__` may be absent (clustering) and,
    when present (labelled anomaly detection), it is never even loaded, so `fit` cannot train on it.
    """
    import pyarrow.parquet as pq

    try:
        available = set(pq.read_schema(path).names)
    except (OSError, ValueError) as e:
        raise ManifestError(f"cannot read Parquet {path}: {e}") from None
    columns = [f.name for f in features] + ([TARGET_COLUMN] if with_target else [])
    missing = [c for c in columns if c not in available]
    if missing:
        raise ManifestError(f"{path} is missing columns: {', '.join(missing)}")
    return pq.read_table(path, columns=columns)


def _write_predictions(
    algo: Algorithm, data: pa.Table, output: Path, events: EventSink, evaluate: bool = False
) -> None:
    """Predict, write the Parquet file, and (for clustering, `evaluate=True`) emit `eval/*` metrics."""
    import pyarrow.parquet as pq

    preds = algo.predict(data)
    if preds.num_rows != data.num_rows:
        raise RuntimeError(f"{algo.name} returned {preds.num_rows} predictions for {data.num_rows} rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(preds, output)
    events.artifact("predictions", str(output))
    if evaluate and isinstance(algo, Evaluates):
        for name, value in algo.evaluate(data, preds).items():
            events.metric(name, value)


def _algorithm_class(name: str, task: str) -> type[Algorithm]:
    cls = algorithms.get(name)
    if task not in cls.tasks:
        raise ManifestError(
            f"`{name}` does not support task `{task}` (supports: {', '.join(sorted(cls.tasks))})"
        )
    return cls


def run_fit(m: FitManifest, events: EventSink) -> int:
    cls = _algorithm_class(m.algorithm, m.task)
    m.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        events=events,
        work_dir=m.work_dir,
        params=dict(m.params),
        resume_from=m.resume_from,
        task=m.task,
        features=list(m.features),
        classes=list(m.classes),
        seed=m.seed,
    )
    ctx.install_signal_handlers()

    supervised = m.task in SUPERVISED_TASKS
    train = read_table(m.train, m.features, with_target=supervised)
    valid = read_table(m.valid, m.features, with_target=supervised) if m.valid else None
    events.log(
        f"fit {m.algorithm} ({m.task}): {train.num_rows} train rows"
        + (f", {valid.num_rows} valid rows" if valid is not None else "")
        + f", {len(m.features)} features"
    )

    algo = cls()
    if m.resume_from:
        algo.resume_from(m.resume_from)
        events.log(f"resumed from {m.resume_from}")
    algo.fit(train, valid, ctx)
    if ctx.should_stop():
        events.error("stopped before completion; checkpoint saved when supported", retryable=True)
        return EXIT_STOPPED

    exported: list[Path] = []
    if "onnx" in m.export:
        try:
            exported.append(algo.export(m.model_dir, "onnx"))
        except Exception as e:  # conversion gaps must not fail the run
            traceback.print_exc(file=sys.stderr)
            events.log(f"ONNX export failed for `{m.algorithm}`: {type(e).__name__}: {e}", level="warn")
    algo.save(m.model_dir)
    events.artifact("model", str(m.model_dir), "native")
    for path in exported:
        events.artifact("model", str(path), "onnx")

    for i, entry in enumerate(m.predict):
        data = read_table(entry.input, m.features, with_target=False)
        # Section 8: fit-time cluster metrics refer to the first `predict` entry (the validation set).
        _write_predictions(algo, data, entry.output, events, evaluate=i == 0)
    events.done({})
    return EXIT_OK


def run_predict(m: PredictManifest, events: EventSink) -> int:
    try:
        descriptor = json.loads((m.model_dir / MODEL_DESCRIPTOR).read_text())
    except FileNotFoundError:
        raise ManifestError(f"no {MODEL_DESCRIPTOR} in {m.model_dir}") from None
    if descriptor.get("algorithm") != m.algorithm or descriptor.get("task") != m.task:
        raise ManifestError(
            f"{m.model_dir} holds a `{descriptor.get('algorithm')}` / `{descriptor.get('task')}` model, "
            f"but the manifest asks for `{m.algorithm}` / `{m.task}`"
        )
    algo = _algorithm_class(m.algorithm, m.task).load(m.model_dir)
    features = [Feature(f["name"], f["kind"]) for f in descriptor["features"]]
    data = read_table(m.input, features, with_target=False)
    _write_predictions(algo, data, m.output, events, evaluate=True)
    events.done({})
    return EXIT_OK


QUALITY_HINT = (
    "install the `quality` extra: uv sync --extra quality (or pip install 'tempera-worker[quality]')"
)


def run_quality(m: QualityManifest, events: EventSink) -> int:
    try:
        from tempera_worker.quality import label_issues
    except ImportError as e:
        events.error(f"`quality` needs Cleanlab and scikit-learn ({e}); {QUALITY_HINT}")
        return EXIT_ERROR
    data = read_table(m.data, m.features, with_target=True)
    return label_issues.run(m, data, events)


def run_fairness(m: FairnessManifest, events: EventSink) -> int:
    try:
        from tempera_worker.quality import fairness
    except ImportError as e:
        events.error(f"`fairness` needs Fairlearn ({e}); {QUALITY_HINT}")
        return EXIT_ERROR
    return fairness.run(m, events)


LLM_HINT = "install the `llm` extra: uv sync --extra llm (or pip install 'tempera-worker[llm]')"


def run_finetune(manifest: Path, events: EventSink) -> int:
    try:
        from tempera_worker.llm import finetune
    except ImportError as e:
        events.error(f"`finetune` needs torch, transformers and peft ({e}); {LLM_HINT}")
        return EXIT_ERROR
    m = finetune.load_manifest(manifest)
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        events.error(f"`finetune` needs torch, transformers and peft ({e}); {LLM_HINT}")
        return EXIT_ERROR
    return finetune.run(m, events)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for name in algorithms.names():
            print(name)
        return EXIT_OK

    events = EventSink(_event_stream())
    try:
        if args.command == "fit":
            return run_fit(load_fit(args.manifest), events)
        if args.command == "quality":
            return run_quality(load_quality(args.manifest), events)
        if args.command == "finetune":
            return run_finetune(args.manifest, events)
        if args.command == "fairness":
            return run_fairness(load_fairness(args.manifest), events)
        return run_predict(load_predict(args.manifest), events)
    except ManifestError as e:
        events.error(str(e))
        return EXIT_BAD_MANIFEST
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        events.error(f"{type(e).__name__}: {e}")
        return EXIT_ERROR

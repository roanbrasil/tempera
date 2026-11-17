"""A fake algorithm that exercises the whole manifest/event/checkpoint/resume path.

Used by tests and to smoke-test agents without installing any ML library (only pyarrow). It predicts the
first class with uniform probabilities (classification) or its final "loss" (regression).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from tempera_worker.manifest import Feature, ManifestError, Task
from tempera_worker.protocol import MODEL_DESCRIPTOR

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext
    from tempera_worker.protocol import ExportFormat

STATE_FILE = "state.json"


class DummyAlgorithm:
    name: ClassVar[str] = "dummy"
    tasks: ClassVar[frozenset[str]] = frozenset(
        {"binary_classification", "multiclass_classification", "regression"}
    )

    def __init__(self) -> None:
        self.step = 0
        self.loss = 1.0
        self.task: Task = "regression"
        self.classes: list[str] = []
        self.features: list[Feature] = []

    def fit(self, train: pa.Table | None, valid: pa.Table | None, ctx: RunContext) -> None:
        self.task, self.classes, self.features = ctx.task, list(ctx.classes), list(ctx.features)
        steps = int(ctx.params.get("steps", 10))
        delay = float(ctx.params.get("step_delay_s", 0.0))
        ckpt_dir = ctx.work_dir / "ckpt"
        while self.step < steps:
            if ctx.should_stop():
                self.save_checkpoint(ckpt_dir)
                ctx.saved_checkpoint(ckpt_dir, self.step)
                return
            self.step += 1
            self.loss = math.exp(-self.step / max(steps, 1) * 3)
            ctx.log_metric("train/loss", self.loss, step=self.step)
            ctx.progress(self.step, steps)
            if ctx.checkpoint_due():
                self.save_checkpoint(ckpt_dir)
                ctx.saved_checkpoint(ckpt_dir, self.step)
            if delay:
                time.sleep(delay)
        if valid is not None:
            ctx.log_metric("valid/loss", self.loss * 1.1, step=self.step)

    def predict(self, data: pa.Table) -> pa.Table:
        import pyarrow as pa

        n = data.num_rows
        if self.task == "regression":
            return pa.table({"prediction": pa.array([self.loss] * n, type=pa.float64())})
        columns: dict[str, Any] = {"prediction": pa.array([self.classes[0]] * n, type=pa.string())}
        for label in self.classes:
            columns[f"proba__{label}"] = pa.array([1.0 / len(self.classes)] * n, type=pa.float64())
        return pa.table(columns)

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / STATE_FILE).write_text(json.dumps({"step": self.step, "loss": self.loss}))

    def resume_from(self, path: Path) -> None:
        try:
            state = json.loads((path / STATE_FILE).read_text())
        except FileNotFoundError:
            raise ManifestError(f"{path} is not a dummy checkpoint") from None
        self.step, self.loss = int(state["step"]), float(state["loss"])

    def export(self, path: Path, fmt: ExportFormat) -> Path:
        if fmt != "native":
            raise ValueError("dummy only exports `native`")
        return self.save(path)

    def save(self, model_dir: Path) -> Path:
        self.save_checkpoint(model_dir)
        descriptor = {
            "format_version": 1,
            "algorithm": self.name,
            "task": self.task,
            "classes": self.classes,
            "features": [f.to_json() for f in self.features],
            "files": {"native": STATE_FILE},
        }
        out = model_dir / MODEL_DESCRIPTOR
        out.write_text(json.dumps(descriptor, indent=2))
        return out

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        d = json.loads((model_dir / MODEL_DESCRIPTOR).read_text())
        obj = cls()
        obj.task, obj.classes = d["task"], list(d["classes"])
        obj.features = [Feature(f["name"], f["kind"]) for f in d["features"]]
        obj.resume_from(model_dir)
        return obj

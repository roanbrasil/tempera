"""A fake algorithm that exercises the whole event/checkpoint/resume path.

Used by tests and to smoke-test agents without installing any ML library.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext
    from tempera_worker.protocol import ExportFormat


class DummyAlgorithm:
    name: ClassVar[str] = "dummy"
    tasks: ClassVar[frozenset[str]] = frozenset({"binary_classification", "regression"})

    def __init__(self) -> None:
        self.step = 0
        self.loss = 1.0

    def fit(self, train: pa.Table | None, valid: pa.Table | None, ctx: RunContext) -> None:
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
        ctx.log_metric("val/loss", self.loss * 1.1, step=self.step)

    def predict(self, data: pa.Table) -> pa.Table:
        import pyarrow as pa

        return pa.table({"prediction": [self.loss] * data.num_rows})

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "state.json").write_text(json.dumps({"step": self.step, "loss": self.loss}))

    def resume_from(self, path: Path) -> None:
        state = json.loads((path / "state.json").read_text())
        self.step, self.loss = int(state["step"]), float(state["loss"])

    def export(self, path: Path, fmt: ExportFormat) -> Path:
        if fmt != "native":
            raise ValueError("dummy only exports `native`")
        out = path / "model.json"
        path.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"loss": self.loss}))
        return out

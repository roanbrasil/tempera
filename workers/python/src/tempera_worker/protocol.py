"""The algorithm protocol every Python backend implements."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext

ExportFormat = Literal["onnx", "safetensors", "native"]

MODEL_DESCRIPTOR = "model.json"


@runtime_checkable
class Algorithm(Protocol):
    """Mirrors the Rust `Algorithm` trait (see docs/wiki/Contracts.md, sections 5 and 7).

    Lifecycle of a `fit` run: `fit` -> `export(model_dir, "onnx")` (optional) -> `save(model_dir)` ->
    `predict` for every requested input. A `predict` run calls `load(model_dir)` then `predict`.

    * `save` writes the native model files plus the `model.json` descriptor into `model_dir`.
    * `predict` returns the predictions table defined in Contracts.md, section 7.
    * `export(path, "onnx")` writes an extra file into `path` and returns it; it raises if the
      format is not supported, and the caller decides whether that is fatal.
    """

    name: ClassVar[str]
    tasks: ClassVar[frozenset[str]]

    def fit(self, train: pa.Table | None, valid: pa.Table | None, ctx: RunContext) -> None: ...

    def predict(self, data: pa.Table) -> pa.Table: ...

    def save_checkpoint(self, path: Path) -> None: ...

    def resume_from(self, path: Path) -> None: ...

    def export(self, path: Path, fmt: ExportFormat) -> Path: ...

    def save(self, model_dir: Path) -> Path: ...

    @classmethod
    def load(cls, model_dir: Path) -> Self: ...


@runtime_checkable
class Evaluates(Protocol):
    """Algorithms that score their own predictions (clustering, Contracts.md section 8).

    `evaluate` returns metric name -> value (emitted as metric events without a step); metrics that are
    undefined for the data (e.g. silhouette with fewer than 2 clusters) are left out.
    """

    def evaluate(self, data: pa.Table, predictions: pa.Table) -> dict[str, float]: ...

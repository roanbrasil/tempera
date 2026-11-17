"""The algorithm protocol every Python backend implements."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext

ExportFormat = Literal["onnx", "safetensors", "native"]


@runtime_checkable
class Algorithm(Protocol):
    """Mirrors the Rust `Algorithm` trait (see docs/wiki/Contracts.md)."""

    name: ClassVar[str]
    tasks: ClassVar[frozenset[str]]

    def fit(self, train: pa.Table | None, valid: pa.Table | None, ctx: RunContext) -> None: ...

    def predict(self, data: pa.Table) -> pa.Table: ...

    def save_checkpoint(self, path: Path) -> None: ...

    def resume_from(self, path: Path) -> None: ...

    def export(self, path: Path, fmt: ExportFormat) -> Path: ...

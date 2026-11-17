"""Shared plumbing for tabular algorithms: manifest fields, model descriptor, predictions, ONNX."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import numpy as np

from tempera_worker.manifest import Feature, ManifestError, Task
from tempera_worker.protocol import MODEL_DESCRIPTOR, ExportFormat
from tempera_worker.tabular.data import classification_predictions, regression_predictions

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext

CLASSIFICATION_TASKS: frozenset[str] = frozenset({"binary_classification", "multiclass_classification"})
ALL_TASKS: frozenset[str] = CLASSIFICATION_TASKS | {"regression"}
DESCRIPTOR_VERSION = 1
ONNX_FILE = "model.onnx"
ONNX_TOLERANCE = 1e-5
PARITY_ROWS = 1000


def default_threads() -> int:
    """Worker threads for native libraries: `TEMPERA_NUM_THREADS`, else every CPU."""
    env = os.environ.get("TEMPERA_NUM_THREADS", "")
    return int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)


def pop_int(params: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    """Remove every alias in `keys` from `params` and return the first one found (or `default`)."""
    found: int | None = None
    for key in keys:
        if key in params:
            value = params.pop(key)
            if found is None:
                found = int(value)
    return default if found is None else found


def read_descriptor(model_dir: Path) -> dict[str, Any]:
    path = model_dir / MODEL_DESCRIPTOR
    try:
        return cast(dict[str, Any], json.loads(path.read_text()))
    except FileNotFoundError:
        raise ManifestError(f"no {MODEL_DESCRIPTOR} in {model_dir}") from None


class TabularAlgorithm(ABC):
    """Base class: subclasses implement `_fit`, `_raw_predict`, native (de)serialization and ONNX."""

    name: ClassVar[str]
    tasks: ClassVar[frozenset[str]] = ALL_TASKS
    native_file: ClassVar[str]

    def __init__(self) -> None:
        self.task: Task = "regression"
        self.features: list[Feature] = []
        self.classes: list[str] = []
        self.seed = 0
        self.params: dict[str, Any] = {}
        self.files: dict[str, str] = {}
        self.onnx_meta: dict[str, Any] = {}
        self._ctx: RunContext | None = None
        self._parity_sample: pa.Table | None = None

    # ---- hooks -----------------------------------------------------------------------------------------

    @abstractmethod
    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None: ...

    @abstractmethod
    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        """Probabilities `(n, len(classes))` for classification, values `(n,)` for regression."""

    @abstractmethod
    def _save_native(self, path: Path) -> None: ...

    @abstractmethod
    def _load_native(self, path: Path) -> None: ...

    @abstractmethod
    def _to_onnx(self) -> bytes:
        """Serialized ONNX model; raise if the model cannot be converted."""

    @abstractmethod
    def onnx_feed(self, data: pa.Table) -> dict[str, np.ndarray]:
        """ONNX input tensors for `data` (also documented in `model.json`)."""

    @abstractmethod
    def feature_importance(self) -> dict[str, float]: ...

    def _extra_descriptor(self) -> dict[str, Any]:
        return {}

    def _load_extra(self, descriptor: dict[str, Any]) -> None:  # noqa: B027 - optional hook
        pass

    # ---- protocol --------------------------------------------------------------------------------------

    def score_onnx(self, onnx_path: Path, data: pa.Table) -> np.ndarray:
        """Score `data` with an exported ONNX model: probabilities `(n, K)` or values `(n,)`."""
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = self.onnx_feed(data)
        if self.is_classification:
            (proba,) = session.run(["probabilities"], feed)
            return np.asarray(proba, dtype=np.float64)
        (values,) = session.run([session.get_outputs()[0].name], feed)
        return np.asarray(values, dtype=np.float64).ravel()

    def _check_onnx_parity(self, onnx_path: Path) -> None:
        """Compare ONNX and native outputs on (a sample of) the training rows; warn when they drift.

        ONNX runtimes evaluate tree splits in float32, so rows within float32 rounding of a threshold can
        land in a different leaf than in the native float64 model.
        """
        if self._parity_sample is None or self._parity_sample.num_rows == 0:
            return
        native = self._raw_predict(self._parity_sample)
        diff = float(np.max(np.abs(self.score_onnx(onnx_path, self._parity_sample) - native)))
        # Relative to the output scale: probabilities are in [0, 1], but a regression target in the
        # hundreds of thousands loses ~1e-7 relative precision in float32, which is not a real drift.
        scale = max(1.0, float(np.max(np.abs(native)))) if native.size else 1.0
        tolerance = ONNX_TOLERANCE * scale
        self.onnx_meta["parity_max_abs_diff"] = diff
        self.onnx_meta["parity_tolerance"] = tolerance
        if diff > tolerance and self._ctx is not None:
            self._ctx.events.log(
                f"ONNX export of `{self.name}` differs from the native model by up to {diff:.3g} "
                f"on training rows (tolerance {tolerance:.3g})",
                level="warn",
            )

    @property
    def is_classification(self) -> bool:
        return self.task in CLASSIFICATION_TASKS

    def fit(self, train: pa.Table | None, valid: pa.Table | None, ctx: RunContext) -> None:
        if train is None:
            raise ManifestError(f"`{self.name}` needs a training set")
        if ctx.task not in self.tasks:
            raise ManifestError(f"`{self.name}` does not support task `{ctx.task}`")
        self.task, self.features, self.classes = ctx.task, list(ctx.features), list(ctx.classes)
        self.seed, self.params = ctx.seed, dict(ctx.params)
        self._ctx, self._parity_sample = ctx, train.slice(0, PARITY_ROWS)
        np.random.seed(self.seed % 2**32)
        self._fit(train, valid, ctx)
        if ctx.should_stop():
            return
        for feature, value in self.feature_importance().items():
            ctx.log_metric(f"importance/{feature}", float(value))

    def predict(self, data: pa.Table) -> pa.Table:
        raw = self._raw_predict(data)
        if self.is_classification:
            return classification_predictions(raw, self.classes)
        return regression_predictions(raw)

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError(f"`{self.name}` does not support checkpoints")

    def resume_from(self, path: Path) -> None:
        raise ManifestError(f"`{self.name}` does not support resuming from a checkpoint")

    def export(self, path: Path, fmt: ExportFormat) -> Path:
        if fmt == "native":
            return self.save(path)
        if fmt == "onnx":
            data = self._to_onnx()
            path.mkdir(parents=True, exist_ok=True)
            out = path / ONNX_FILE
            out.write_bytes(data)
            self.files["onnx"] = ONNX_FILE
            self._check_onnx_parity(out)
            return out
        raise ValueError(f"`{self.name}` cannot export `{fmt}`")

    def save(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        self._save_native(model_dir / self.native_file)
        self.files["native"] = self.native_file
        descriptor: dict[str, Any] = {
            "format_version": DESCRIPTOR_VERSION,
            "algorithm": self.name,
            "task": self.task,
            "classes": self.classes,
            "features": [f.to_json() for f in self.features],
            "seed": self.seed,
            "params": self.params,
            "files": dict(sorted(self.files.items())),
            **self._extra_descriptor(),
        }
        if "onnx" in self.files:
            descriptor["onnx"] = self.onnx_meta
        out = model_dir / MODEL_DESCRIPTOR
        out.write_text(json.dumps(descriptor, indent=2, default=str))
        return out

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        d = read_descriptor(model_dir)
        if d.get("algorithm") != cls.name:
            raise ManifestError(f"{model_dir} holds a `{d.get('algorithm')}` model, not `{cls.name}`")
        obj = cls()
        obj.task = d["task"]
        obj.features = [Feature(f["name"], f["kind"]) for f in d["features"]]
        obj.classes = list(d.get("classes", []))
        obj.seed = int(d.get("seed", 0))
        obj.params = dict(d.get("params", {}))
        obj.files = dict(d["files"])
        obj.onnx_meta = dict(d.get("onnx", {}))
        obj._load_extra(d)
        obj._load_native(model_dir / obj.files["native"])
        return obj


def run_onnx(model_dir: Path, data: pa.Table) -> np.ndarray:
    """Score `data` with the ONNX model saved in `model_dir` (see the `onnx` entry of `model.json`)."""
    from tempera_worker import algorithms

    algo = algorithms.get(read_descriptor(model_dir)["algorithm"]).load(model_dir)
    if not isinstance(algo, TabularAlgorithm) or "onnx" not in algo.files:
        raise ValueError(f"{model_dir} has no ONNX export")
    return algo.score_onnx(model_dir / algo.files["onnx"], data)

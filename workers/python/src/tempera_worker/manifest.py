"""Backend run manifests (docs/wiki/Contracts.md, sections 7 and 8).

The agent hands every input to a backend in a JSON manifest. Anything wrong with it (missing field, unknown
task, missing file...) raises `ManifestError`, which the CLI turns into exit code 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

Task = Literal[
    "binary_classification", "multiclass_classification", "regression", "clustering", "anomaly_detection"
]
FeatureKind = Literal["numeric", "categorical", "boolean"]

CLASSIFICATION_TASKS: frozenset[str] = frozenset({"binary_classification", "multiclass_classification"})
SUPERVISED_TASKS: frozenset[str] = CLASSIFICATION_TASKS | {"regression"}
# Section 8: these never train on `__target__` (absent for clustering, evaluation-only for anomalies).
UNSUPERVISED_TASKS: frozenset[str] = frozenset({"clustering", "anomaly_detection"})
TASKS: frozenset[str] = SUPERVISED_TASKS | UNSUPERVISED_TASKS
FEATURE_KINDS: frozenset[str] = frozenset({"numeric", "categorical", "boolean"})
EXPORT_FORMATS: frozenset[str] = frozenset({"onnx", "native"})

TARGET_COLUMN = "__target__"
RESERVED_COLUMNS: frozenset[str] = frozenset({"__target__", "__weight__", "__group__", "__time__"})


class ManifestError(ValueError):
    """The manifest (or the data it points to) is invalid. Exit code 2."""


@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind}


@dataclass(frozen=True)
class PredictTarget:
    input: Path
    output: Path


@dataclass(frozen=True)
class FitManifest:
    algorithm: str
    task: Task
    features: list[Feature]
    train: Path
    work_dir: Path
    classes: list[str] = field(default_factory=list)
    positive_class: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    valid: Path | None = None
    predict: list[PredictTarget] = field(default_factory=list)
    export: list[str] = field(default_factory=lambda: ["native"])
    resume_from: Path | None = None

    @property
    def is_classification(self) -> bool:
        return self.task in CLASSIFICATION_TASKS

    @property
    def is_supervised(self) -> bool:
        return self.task in SUPERVISED_TASKS

    @property
    def model_dir(self) -> Path:
        return self.work_dir / "model"


@dataclass(frozen=True)
class PredictManifest:
    algorithm: str
    task: Task
    model_dir: Path
    input: Path
    output: Path


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except (OSError, json.JSONDecodeError) as e:
        raise ManifestError(f"cannot read manifest {path}: {e}") from None
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    return cast(dict[str, Any], raw)


def _str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"`{key}` must be a non-empty string")
    return value


def _path(raw: dict[str, Any], key: str, must_exist: bool = False) -> Path:
    p = Path(_str(raw, key))
    if must_exist and not p.exists():
        raise ManifestError(f"`{key}` does not exist: {p}")
    return p


def _task(raw: dict[str, Any]) -> Task:
    task = _str(raw, "task")
    if task not in TASKS:
        raise ManifestError(f"unknown task `{task}` (known: {', '.join(sorted(TASKS))})")
    return cast(Task, task)


def _features(raw: dict[str, Any]) -> list[Feature]:
    items = raw.get("features")
    if not isinstance(items, list) or not items:
        raise ManifestError("`features` must be a non-empty list")
    out: list[Feature] = []
    seen: set[str] = set()
    for item in cast(list[Any], items):
        if not isinstance(item, dict):
            raise ManifestError("every feature must be an object with `name` and `kind`")
        item = cast(dict[str, Any], item)
        name, kind = _str(item, "name"), _str(item, "kind")
        if kind not in FEATURE_KINDS:
            raise ManifestError(f"feature `{name}`: unknown kind `{kind}`")
        if name in RESERVED_COLUMNS:
            raise ManifestError(f"`{name}` is a reserved column and cannot be a feature")
        if name in seen:
            raise ManifestError(f"duplicate feature `{name}`")
        seen.add(name)
        out.append(Feature(name, cast(FeatureKind, kind)))
    return out


def _classes(raw: dict[str, Any], task: Task) -> tuple[list[str], str | None]:
    if task in UNSUPERVISED_TASKS:
        # `classes` is omitted; anomaly detection with labels may name the anomaly label (section 8).
        positive = raw.get("positive_class")
        if positive is not None and not isinstance(positive, str):
            raise ManifestError("`positive_class` must be a string")
        return [], positive
    if task == "regression":
        return [], None
    classes = raw.get("classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in cast(list[Any], classes)):
        raise ManifestError("`classes` must be a list of strings for classification")
    classes = cast(list[str], classes)
    if classes != sorted(set(classes)):
        raise ManifestError("`classes` must be sorted and unique")
    if task == "binary_classification" and len(classes) != 2:
        raise ManifestError("binary_classification needs exactly 2 classes")
    if task == "multiclass_classification" and len(classes) < 3:
        raise ManifestError("multiclass_classification needs at least 3 classes")
    positive = raw.get("positive_class")
    if positive is not None and positive not in classes:
        raise ManifestError(f"`positive_class` `{positive}` is not in `classes`")
    return classes, positive


def load_fit(path: Path) -> FitManifest:
    raw = _load(path)
    task = _task(raw)
    classes, positive = _classes(raw, task)

    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise ManifestError("`params` must be an object")
    seed = raw.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestError("`seed` must be a non-negative integer")

    predict: list[PredictTarget] = []
    entries = raw.get("predict", [])
    if not isinstance(entries, list):
        raise ManifestError("`predict` must be a list")
    for entry in cast(list[Any], entries):
        if not isinstance(entry, dict):
            raise ManifestError("every `predict` entry must be an object with `input` and `output`")
        entry = cast(dict[str, Any], entry)
        predict.append(PredictTarget(_path(entry, "input", must_exist=True), _path(entry, "output")))

    export = raw.get("export", ["native"])
    if not isinstance(export, list) or not all(e in EXPORT_FORMATS for e in cast(list[Any], export)):
        raise ManifestError(f"`export` must be a list of: {', '.join(sorted(EXPORT_FORMATS))}")

    return FitManifest(
        algorithm=_str(raw, "algorithm"),
        task=task,
        features=_features(raw),
        train=_path(raw, "train", must_exist=True),
        work_dir=_path(raw, "work_dir"),
        classes=classes,
        positive_class=positive,
        params=cast(dict[str, Any], params),
        seed=seed,
        valid=_path(raw, "valid", must_exist=True) if raw.get("valid") is not None else None,
        predict=predict,
        export=list(dict.fromkeys(cast(list[str], export))),
        resume_from=_path(raw, "resume_from", True) if raw.get("resume_from") is not None else None,
    )


def load_predict(path: Path) -> PredictManifest:
    raw = _load(path)
    return PredictManifest(
        algorithm=_str(raw, "algorithm"),
        task=_task(raw),
        model_dir=_path(raw, "model_dir", must_exist=True),
        input=_path(raw, "input", must_exist=True),
        output=_path(raw, "output"),
    )

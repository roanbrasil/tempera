"""CLI contract (Contracts.md, section 7) exercised with the dummy algorithm: no ML extra needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers import run_cli, run_manifest, validate_lines

FEATURES = [
    {"name": "x", "kind": "numeric"},
    {"name": "color", "kind": "categorical"},
    {"name": "flag", "kind": "boolean"},
]


def _dataset(path: Path, n: int = 8, target: list[Any] | None = None) -> Path:
    table = pa.table(
        {
            "x": pa.array([float(i) for i in range(n)]),
            "color": pa.array(["red", "blue", None, "red"] * (n // 4)),
            "flag": pa.array([True, False] * (n // 2)),
            "__target__": pa.array(target if target is not None else ["no", "yes"] * (n // 2)),
            "__group__": pa.array(list(range(n))),
        }
    )
    pq.write_table(table, path)
    return path


def _fit_manifest(tmp: Path, **overrides: Any) -> dict[str, Any]:
    data = _dataset(tmp / "train.parquet")
    manifest: dict[str, Any] = {
        "algorithm": "dummy",
        "task": "binary_classification",
        "classes": ["no", "yes"],
        "positive_class": "yes",
        "features": FEATURES,
        "params": {"steps": 5},
        "seed": 42,
        "train": str(data),
        "valid": str(data),
        "predict": [{"input": str(data), "output": str(tmp / "run" / "valid_pred.parquet")}],
        "work_dir": str(tmp / "run"),
        "export": ["onnx", "native"],
    }
    manifest.update(overrides)
    return manifest


def test_fit_emits_valid_stream_and_predictions(tmp_path: Path) -> None:
    proc, events = run_manifest("fit", _fit_manifest(tmp_path), tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1]["t"] == "done"
    assert [e["step"] for e in events if e["t"] == "progress"] == [1, 2, 3, 4, 5]
    # dummy cannot export ONNX: the run warns and carries on.
    assert any(e["t"] == "log" and e["level"] == "warn" and "ONNX" in e["msg"] for e in events)
    artifacts = [e for e in events if e["t"] == "artifact"]
    assert {"kind": "model", "format": "native", "path": str(tmp_path / "run" / "model")} in [
        {k: v for k, v in a.items() if k != "t"} for a in artifacts
    ]
    assert (tmp_path / "run" / "model" / "model.json").exists()

    preds = pq.read_table(tmp_path / "run" / "valid_pred.parquet")
    assert preds.column_names == ["prediction", "proba__no", "proba__yes"]
    assert preds.schema.field("prediction").type == pa.string()
    assert preds.schema.field("proba__yes").type == pa.float64()
    assert preds.num_rows == 8


def test_predict_from_saved_model(tmp_path: Path) -> None:
    proc, _ = run_manifest("fit", _fit_manifest(tmp_path), tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "test_pred.parquet"
    manifest = {
        "algorithm": "dummy",
        "task": "binary_classification",
        "model_dir": str(tmp_path / "run" / "model"),
        "input": str(tmp_path / "train.parquet"),
        "output": str(out),
    }
    proc, events = run_manifest("predict", manifest, tmp_path / "pred.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1]["t"] == "done"
    assert pq.read_table(out).equals(pq.read_table(tmp_path / "run" / "valid_pred.parquet"))


def test_regression_predictions_schema(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "reg.parquet", target=[float(i) for i in range(8)])
    out = tmp_path / "run" / "pred.parquet"
    manifest = _fit_manifest(
        tmp_path, task="regression", classes=None, positive_class=None, train=str(data), valid=None
    )
    manifest["predict"] = [{"input": str(data), "output": str(out)}]
    proc, _ = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    preds = pq.read_table(out)
    assert preds.column_names == ["prediction"]
    assert preds.schema.field("prediction").type == pa.float64()


def test_resume_from_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "state.json").write_text('{"step": 4, "loss": 0.3}')
    proc, events = run_manifest("fit", _fit_manifest(tmp_path, resume_from=str(ckpt)), tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert [e["step"] for e in events if e["t"] == "progress"] == [5]


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"algorithm": "does-not-exist"}, "unknown algorithm"),
        ({"task": "ranking"}, "unknown task"),
        ({"classes": ["yes", "no"]}, "sorted"),
        ({"features": [{"name": "missing_col", "kind": "numeric"}]}, "missing columns"),
        ({"features": [{"name": "x", "kind": "text"}]}, "unknown kind"),
        ({"train": "/does/not/exist.parquet"}, "does not exist"),
    ],
)
def test_bad_manifest_exits_2(tmp_path: Path, overrides: dict[str, Any], fragment: str) -> None:
    proc, events = run_manifest("fit", _fit_manifest(tmp_path, **overrides), tmp_path / "run.json")
    assert proc.returncode == 2, proc.stderr
    assert events[-1]["t"] == "error"
    assert fragment in events[-1]["msg"]


def test_unreadable_manifest_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    proc = run_cli("fit", "--manifest", str(bad))
    assert proc.returncode == 2
    assert validate_lines(proc.stdout)[-1]["t"] == "error"


def test_predict_rejects_mismatched_model(tmp_path: Path) -> None:
    proc, _ = run_manifest("fit", _fit_manifest(tmp_path), tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    manifest = {
        "algorithm": "lightgbm",
        "task": "binary_classification",
        "model_dir": str(tmp_path / "run" / "model"),
        "input": str(tmp_path / "train.parquet"),
        "output": str(tmp_path / "out.parquet"),
    }
    proc, events = run_manifest("predict", manifest, tmp_path / "pred.json")
    assert proc.returncode == 2
    assert events[-1]["t"] == "error"


def test_list_shows_every_algorithm() -> None:
    proc = run_cli("list")
    assert proc.returncode == 0
    assert proc.stdout.split() == [
        "adaboost",
        "catboost",
        "dbscan",
        "dummy",
        "elastic_net",
        "extra_trees",
        "hist_gradient_boosting",
        "isolation_forest",
        "kmeans",
        "knn",
        "lasso",
        "lightgbm",
        "linear_regression",
        "logistic_regression",
        "mlp",
        "naive_bayes",
        "pca_kmeans",
        "random_forest",
        "ridge",
        "sgd",
        "svm",
        "xgboost",
    ]


@pytest.mark.parametrize("task", ["clustering", "anomaly_detection"])
def test_unsupervised_task_unsupported_by_dummy_exits_2(tmp_path: Path, task: str) -> None:
    # Unsupervised manifests omit `classes`; the task is valid but `dummy` does not implement it.
    manifest = _fit_manifest(tmp_path, task=task)
    del manifest["classes"], manifest["positive_class"]
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 2, proc.stderr
    assert "does not support task" in events[-1]["msg"]


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is not None, reason="tabular extra is installed")
def test_tabular_algorithm_without_extra_fails_cleanly(tmp_path: Path) -> None:
    proc, events = run_manifest(
        "fit", _fit_manifest(tmp_path, algorithm="logistic_regression"), tmp_path / "run.json"
    )
    assert proc.returncode == 1
    assert "tabular" in events[-1]["msg"]

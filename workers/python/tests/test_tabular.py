"""Contract tests for every tabular algorithm on small, fixed synthetic datasets.

Each case fits through the CLI subprocess, validates every stdout line against the event schema, checks
the predictions Parquet, a minimum quality bar, that `predict` from the saved model reproduces fit-time
predictions, and (when ONNX was exported) that onnxruntime matches the native model within 1e-5.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("catboost")
pytest.importorskip("onnxruntime")
pytest.importorskip("pandas")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import ENV, run_manifest, validate_lines, write_manifest
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

from tempera_worker.tabular.base import run_onnx

pytestmark = pytest.mark.tabular

FEATURES = [
    {"name": "x1", "kind": "numeric"},
    {"name": "x2", "kind": "numeric"},
    {"name": "plan", "kind": "categorical"},
    {"name": "has_discount", "kind": "boolean"},
]
CLASSES = {
    "binary_classification": ["no", "yes"],
    "multiclass_classification": ["high", "low", "mid"],
    "regression": [],
}
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 60},
    "extra_trees": {"n_estimators": 60},
    "lightgbm": {"num_boost_round": 300, "min_data_in_leaf": 10},
    "xgboost": {"num_boost_round": 300},
    "catboost": {"num_boost_round": 300},
}
ALL_SUPERVISED = ("binary_classification", "multiclass_classification", "regression")
CLASSIFICATION = ALL_SUPERVISED[:2]
TASKS_BY_ALGORITHM: dict[str, tuple[str, ...]] = {
    "logistic_regression": CLASSIFICATION,
    "linear_regression": ("regression",),
    "ridge": ("regression",),
    "lasso": ("regression",),
    "elastic_net": ("regression",),
    "sgd": ALL_SUPERVISED,
    "random_forest": ALL_SUPERVISED,
    "extra_trees": ALL_SUPERVISED,
    "hist_gradient_boosting": ALL_SUPERVISED,
    "adaboost": ALL_SUPERVISED,
    "knn": ALL_SUPERVISED,
    "svm": ALL_SUPERVISED,
    "naive_bayes": CLASSIFICATION,
    "mlp": ALL_SUPERVISED,
    "lightgbm": ALL_SUPERVISED,
    "xgboost": ALL_SUPERVISED,
    "catboost": ALL_SUPERVISED,
}
CASES = [(algorithm, task) for algorithm, tasks in TASKS_BY_ALGORITHM.items() for task in tasks]
# ONNX export must succeed for these (others may legitimately fall back to a warning).
ONNX_REQUIRED = {
    "logistic_regression",
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "extra_trees",
    "lightgbm",
}
# Models without a meaningful per-feature importance must not emit any (no made-up numbers).
NO_IMPORTANCE = {"hist_gradient_boosting", "knn", "svm", "naive_bayes", "mlp"}
# Per-iteration `valid/<loss>` metrics with a step, plus one progress event per iteration.
ITERATIVE = {"lightgbm", "xgboost", "catboost"}


def make_dataset(task: str, n: int, seed: int) -> pa.Table:
    """Easy problem: signal from two numeric features, the category and the flag; ~5% missing values."""
    rng = np.random.default_rng(seed)
    x1 = np.round(rng.normal(size=n), 2)
    x2 = np.round(rng.normal(size=n), 2)
    plan = rng.choice(["basic", "plus", "pro"], size=n)
    flag = rng.random(n) < 0.4
    signal_ = 2.0 * x1 - 1.5 * x2 + 1.5 * (plan == "pro") - 1.0 * (plan == "basic") + 1.0 * flag
    noise = rng.normal(scale=0.3, size=n)
    if task == "regression":
        target = pa.array(signal_ + noise, type=pa.float64())
    elif task == "binary_classification":
        target = pa.array(np.where(signal_ + noise > 0, "yes", "no"), type=pa.string())
    else:
        score = signal_ + noise
        target = pa.array(
            np.where(score > 1.0, "high", np.where(score < -1.0, "low", "mid")), type=pa.string()
        )
    x1_missing = rng.random(n) < 0.05
    plan_missing = rng.random(n) < 0.05
    return pa.table(
        {
            "x1": pa.array(x1, mask=x1_missing, type=pa.float64()),
            "x2": pa.array(x2, type=pa.float64()),
            "plan": pa.array(plan.tolist(), mask=plan_missing, type=pa.string()),
            "has_discount": pa.array(flag.tolist(), type=pa.bool_()),
            "__target__": target,
            "__group__": pa.array(np.arange(n) % 7, type=pa.int64()),
        }
    )


def write_splits(tmp: Path, task: str) -> tuple[Path, Path]:
    train, valid = tmp / "train.parquet", tmp / "valid.parquet"
    pq.write_table(make_dataset(task, 600, seed=1), train)
    pq.write_table(make_dataset(task, 300, seed=2), valid)
    return train, valid


def fit_manifest(tmp: Path, algorithm: str, task: str, **overrides: Any) -> dict[str, Any]:
    train, valid = write_splits(tmp, task)
    manifest: dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "features": FEATURES,
        "params": FAST_PARAMS.get(algorithm, {}),
        "seed": 42,
        "train": str(train),
        "valid": str(valid),
        "predict": [{"input": str(valid), "output": str(tmp / "run" / "valid_pred.parquet")}],
        "work_dir": str(tmp / "run"),
        "export": ["onnx", "native"],
    }
    if task != "regression":
        manifest["classes"] = CLASSES[task]
    if task == "binary_classification":
        manifest["positive_class"] = "yes"
    manifest.update(overrides)
    return manifest


# Default bars: R2 > 0.8, AUC > 0.85, accuracy > 0.6. Relaxed only where the model family justifies it.
RELAXED_BARS: dict[tuple[str, str], dict[str, float]] = {
    # AdaBoost.R2 averages 100 shallow (depth-3) trees with a weighted median: a coarse step function that
    # reaches R2 ~0.77 on this smooth, 600-row additive target, where other ensembles reach ~0.9.
    ("adaboost", "regression"): {"r2": 0.7},
}


def check_quality(algorithm: str, task: str, valid: pa.Table, preds: pa.Table) -> None:
    bars = {"r2": 0.8, "auc": 0.85, "accuracy": 0.6} | RELAXED_BARS.get((algorithm, task), {})
    truth = valid.column("__target__").to_pylist()
    if task == "regression":
        r2 = r2_score(truth, preds.column("prediction").to_numpy())
        print(f"QUALITY {algorithm} {task} r2={r2:.3f}")
        assert r2 > bars["r2"], f"R2 {r2:.3f}"
        return
    classes = CLASSES[task]
    proba = np.column_stack([preds.column(f"proba__{c}").to_numpy() for c in classes])
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    if task == "binary_classification":
        auc = roc_auc_score([t == "yes" for t in truth], proba[:, 1])
    else:
        auc = roc_auc_score(truth, proba, multi_class="ovr", labels=classes)
    accuracy = accuracy_score(truth, preds.column("prediction").to_pylist())
    print(f"QUALITY {algorithm} {task} auc={auc:.3f} acc={accuracy:.3f}")
    assert auc > bars["auc"], f"AUC {auc:.3f}"
    assert accuracy > bars["accuracy"], f"accuracy {accuracy:.3f}"


@pytest.mark.parametrize(("algorithm", "task"), CASES)
def test_contract(tmp_path: Path, algorithm: str, task: str) -> None:
    manifest = fit_manifest(tmp_path, algorithm, task)
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"t": "done", "summary": {}}

    # Feature importance for every feature, without a step (or none at all when the model has none).
    importance = {e["name"]: e for e in events if e["t"] == "metric" and e["name"].startswith("importance/")}
    if algorithm in NO_IMPORTANCE:
        assert not importance
    else:
        assert set(importance) == {f"importance/{f['name']}" for f in FEATURES}
    assert all("step" not in e for e in importance.values())
    if algorithm in ITERATIVE:
        losses = [e for e in events if e["t"] == "metric" and e["name"].startswith("valid/")]
        assert losses and all(isinstance(e.get("step"), int) for e in losses)
        assert sum(e["t"] == "progress" for e in events) == len({e["step"] for e in losses})

    model_dir = tmp_path / "run" / "model"
    artifacts = [e for e in events if e["t"] == "artifact"]
    assert {"t": "artifact", "kind": "model", "path": str(model_dir), "format": "native"} in artifacts
    onnx_events = [a for a in artifacts if a.get("format") == "onnx"]
    if algorithm in ONNX_REQUIRED:
        assert onnx_events == [
            {"t": "artifact", "kind": "model", "path": str(model_dir / "model.onnx"), "format": "onnx"}
        ]
    elif not onnx_events:
        assert any(e["t"] == "log" and e["level"] == "warn" and "ONNX" in e["msg"] for e in events)
    assert (model_dir / "model.json").exists()

    # Predictions Parquet schema and row count.
    valid = pq.read_table(manifest["valid"])
    preds = pq.read_table(tmp_path / "run" / "valid_pred.parquet")
    assert preds.num_rows == valid.num_rows
    expected = ["prediction"] + [f"proba__{c}" for c in CLASSES[task]]
    assert preds.column_names == expected
    want_type = pa.float64() if task == "regression" else pa.string()
    assert preds.schema.field("prediction").type == want_type
    assert all(preds.schema.field(c).type == pa.float64() for c in expected[1:])
    check_quality(algorithm, task, valid, preds)

    # `predict` from the saved model reproduces fit-time predictions.
    out = tmp_path / "again.parquet"
    pred_manifest = {
        "algorithm": algorithm,
        "task": task,
        "model_dir": str(model_dir),
        "input": manifest["valid"],
        "output": str(out),
    }
    proc, events = run_manifest("predict", pred_manifest, tmp_path / "pred.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1]["t"] == "done"
    again = pq.read_table(out)
    assert again.column_names == preds.column_names
    for name in preds.column_names:
        if name == "prediction" and task != "regression":
            assert again.column(name).to_pylist() == preds.column(name).to_pylist()
        else:
            np.testing.assert_allclose(
                again.column(name).to_numpy(), preds.column(name).to_numpy(), atol=1e-9
            )

    # ONNX reproduces the native predictions.
    if onnx_events:
        onnx_out = run_onnx(model_dir, valid)
        if algorithm == "knn":
            # Features are rounded to 2 decimals, so equidistant neighbors are common, and float32 distances
            # in onnxruntime break those ties differently from float64 scikit-learn on a few rows.
            native = (
                preds.column("prediction").to_numpy()
                if task == "regression"
                else np.column_stack([preds.column(f"proba__{c}").to_numpy() for c in CLASSES[task]])
            )
            close = np.isclose(onnx_out, native, rtol=1e-5, atol=1e-5)
            assert (close.all(axis=1) if close.ndim == 2 else close).mean() >= 0.98
            return
        if task == "regression":
            native = preds.column("prediction").to_numpy()
            np.testing.assert_allclose(onnx_out, native, rtol=1e-5, atol=1e-5)
        else:
            native = np.column_stack([preds.column(f"proba__{c}").to_numpy() for c in CLASSES[task]])
            np.testing.assert_allclose(onnx_out, native, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["xgboost", "catboost"])
def test_onnx_without_categorical_features(tmp_path: Path, algorithm: str) -> None:
    # Both converters reject categorical splits; numeric-only models must export and match.
    numeric_only = [f for f in FEATURES if f["kind"] != "categorical"]
    manifest = fit_manifest(tmp_path, algorithm, "binary_classification", features=numeric_only)
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert any(e["t"] == "artifact" and e.get("format") == "onnx" for e in events)
    valid = pq.read_table(manifest["valid"])
    preds = pq.read_table(tmp_path / "run" / "valid_pred.parquet")
    native = np.column_stack(
        [preds.column(f"proba__{c}").to_numpy() for c in CLASSES["binary_classification"]]
    )
    np.testing.assert_allclose(run_onnx(tmp_path / "run" / "model", valid), native, atol=1e-5)


def test_svm_subsamples_large_training_sets(tmp_path: Path) -> None:
    # `max_train_rows` lowers the 20,000-row cap so the test stays fast.
    manifest = fit_manifest(
        tmp_path, "svm", "binary_classification", params={"max_train_rows": 200}, export=["native"]
    )
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    warns = [e["msg"] for e in events if e["t"] == "log" and e["level"] == "warn"]
    assert any("from 600 to 200 rows" in w for w in warns), warns


def test_unseen_category_and_missing_values_at_predict(tmp_path: Path) -> None:
    manifest = fit_manifest(tmp_path, "lightgbm", "binary_classification", predict=[])
    proc, _ = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    weird = pa.table(
        {
            "x1": pa.array([None, 0.5], type=pa.float64()),
            "x2": pa.array([0.1, None], type=pa.float64()),
            "plan": pa.array(["enterprise", None], type=pa.string()),
            "has_discount": pa.array([None, True], type=pa.bool_()),
        }
    )
    pq.write_table(weird, tmp_path / "weird.parquet")
    for algorithm in ("lightgbm",):
        pred_manifest = {
            "algorithm": algorithm,
            "task": "binary_classification",
            "model_dir": str(tmp_path / "run" / "model"),
            "input": str(tmp_path / "weird.parquet"),
            "output": str(tmp_path / "weird_pred.parquet"),
        }
        proc, _ = run_manifest("predict", pred_manifest, tmp_path / "pred.json")
        assert proc.returncode == 0, proc.stderr
        assert pq.read_table(tmp_path / "weird_pred.parquet").num_rows == 2


def test_label_not_in_classes_exits_2(tmp_path: Path) -> None:
    manifest = fit_manifest(
        tmp_path, "logistic_regression", "multiclass_classification", classes=["high", "low", "zzz"]
    )
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 2
    assert "not in `classes`" in events[-1]["msg"]


def test_unsupported_task_exits_2(tmp_path: Path) -> None:
    proc, events = run_manifest(
        "fit", fit_manifest(tmp_path, "linear_regression", "binary_classification"), tmp_path / "run.json"
    )
    assert proc.returncode == 2
    assert "does not support task" in events[-1]["msg"]


@pytest.mark.parametrize("algorithm", ["lightgbm", "xgboost"])
def test_stop_signal_checkpoints_and_resume(tmp_path: Path, algorithm: str) -> None:
    # Many slow rounds and no validation set: the run is still training when SIGTERM arrives.
    params = {"num_boost_round": 1_000_000, ("eta" if algorithm == "xgboost" else "learning_rate"): 0.001}
    manifest = fit_manifest(
        tmp_path, algorithm, "binary_classification", valid=None, predict=[], params=params
    )
    path = write_manifest(tmp_path / "run.json", manifest)
    proc = subprocess.Popen(
        [sys.executable, "-m", "tempera_worker", "fit", "--manifest", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENV,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        if '"t":"progress"' in line:
            proc.send_signal(signal.SIGTERM)
            break
    rest, err = proc.communicate(timeout=120)
    events = validate_lines("".join(lines) + rest)
    assert proc.returncode == 3, err
    assert events[-1]["t"] == "error" and events[-1]["retryable"] is True
    (ckpt,) = [e for e in events if e["t"] == "checkpoint"]
    assert Path(ckpt["path"], "state.json").exists()

    step = ckpt["step"]
    resumed = fit_manifest(
        tmp_path,
        algorithm,
        "binary_classification",
        valid=None,
        params={"num_boost_round": step + 3},
        resume_from=ckpt["path"],
        work_dir=str(tmp_path / "run2"),
        predict=[{"input": manifest["train"], "output": str(tmp_path / "run2" / "pred.parquet")}],
    )
    done, events = run_manifest("fit", resumed, tmp_path / "run2.json")
    assert done.returncode == 0, done.stderr
    assert [e["step"] for e in events if e["t"] == "progress"] == [step + 1, step + 2, step + 3]

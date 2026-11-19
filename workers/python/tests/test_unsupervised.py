"""Contract tests for the unsupervised algorithms (Contracts.md, section 8) on a small blobs dataset.

Three well-separated clusters plus a few planted outliers far away from all of them. Each case fits
through the CLI subprocess, validates every stdout line against the event schema, checks the predictions
Parquet, the `eval/*` metrics, that `predict` from the saved model reproduces fit-time predictions, and
that `__target__` never influences training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("onnxruntime")
pytest.importorskip("pandas")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import run_manifest
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from tempera_worker.tabular.base import run_onnx

pytestmark = pytest.mark.tabular

FEATURES = [
    {"name": "x1", "kind": "numeric"},
    {"name": "x2", "kind": "numeric"},
    {"name": "segment", "kind": "categorical"},
    {"name": "is_top", "kind": "boolean"},
]
CENTERS = np.array([[-6.0, 0.0], [6.0, 0.0], [0.0, 9.0]])
OUTLIERS = np.array([[18.0, 18.0], [-18.0, 16.0], [0.0, -16.0], [20.0, -8.0], [-20.0, -10.0], [14.0, 26.0]])
CLUSTERING = ["kmeans", "dbscan", "pca_kmeans"]
EVAL_METRICS = {"eval/silhouette", "eval/davies_bouldin", "eval/n_clusters", "eval/noise_fraction"}


def make_blobs(per_cluster: int, seed: int, target: str | None = None) -> tuple[pa.Table, np.ndarray]:
    """Blobs + outliers (last rows). Returns the table and the true blob index (-1 for outliers).

    `target`: None (no `__target__`), "labels" (anomaly/normal) or "garbage" (random strings).
    """
    rng = np.random.default_rng(seed)
    blob = np.repeat(np.arange(len(CENTERS)), per_cluster)
    xy = CENTERS[blob] + rng.normal(scale=0.7, size=(len(blob), 2))
    xy = np.vstack([xy, OUTLIERS])
    truth = np.concatenate([blob, np.full(len(OUTLIERS), -1)])
    n = len(truth)
    segment = np.where(truth >= 0, np.array(["a", "b", "c"])[np.maximum(truth, 0)], "b")
    columns: dict[str, Any] = {
        "x1": pa.array(np.round(xy[:, 0], 3), type=pa.float64()),
        "x2": pa.array(np.round(xy[:, 1], 3), type=pa.float64()),
        "segment": pa.array(segment.tolist(), type=pa.string()),
        "is_top": pa.array((truth == 2).tolist(), type=pa.bool_()),
        "__group__": pa.array(np.arange(n) % 5, type=pa.int64()),
    }
    if target == "labels":
        columns["__target__"] = pa.array(np.where(truth < 0, "anomaly", "normal").tolist(), type=pa.string())
    elif target == "garbage":
        columns["__target__"] = pa.array(
            [f"junk-{v}" for v in rng.integers(0, 1000, size=n)], type=pa.string()
        )
    return pa.table(columns), truth


def write_blobs(tmp: Path, target: str | None = None) -> tuple[Path, Path, np.ndarray]:
    train, valid = tmp / "train.parquet", tmp / "valid.parquet"
    pq.write_table(make_blobs(150, seed=1, target=target)[0], train)
    table, truth = make_blobs(80, seed=2, target=target)
    pq.write_table(table, valid)
    return train, valid, truth


def fit_manifest(
    tmp: Path, algorithm: str, task: str, target: str | None = None, **overrides: Any
) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    train, valid, _ = write_blobs(tmp, target)
    manifest: dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "features": FEATURES,
        "params": {},
        "seed": 7,
        "train": str(train),
        "valid": str(valid),
        "predict": [{"input": str(valid), "output": str(tmp / "run" / "valid_pred.parquet")}],
        "work_dir": str(tmp / "run"),
        "export": ["onnx", "native"],
    }
    manifest.update(overrides)
    return manifest


def eval_metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    found = [e for e in events if e["t"] == "metric" and e["name"].startswith("eval/")]
    assert all("step" not in e for e in found)
    return {e["name"]: e["value"] for e in found}


def predict_again(tmp: Path, algorithm: str, task: str, input_: str) -> tuple[pa.Table, list[dict[str, Any]]]:
    out = tmp / "again.parquet"
    manifest = {
        "algorithm": algorithm,
        "task": task,
        "model_dir": str(tmp / "run" / "model"),
        "input": input_,
        "output": str(out),
    }
    proc, events = run_manifest("predict", manifest, tmp / "pred.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"t": "done", "summary": {}}
    return pq.read_table(out), events


@pytest.mark.parametrize("algorithm", CLUSTERING)
def test_clustering_contract(tmp_path: Path, algorithm: str) -> None:
    manifest = fit_manifest(tmp_path, algorithm, "clustering")
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"t": "done", "summary": {}}
    assert not [e for e in events if e["t"] == "metric" and e["name"].startswith("importance/")]

    truth = make_blobs(80, seed=2)[1]
    preds = pq.read_table(tmp_path / "run" / "valid_pred.parquet")
    assert preds.column_names == ["cluster"]
    assert preds.schema.field("cluster").type == pa.int64()
    assert preds.num_rows == len(truth)
    cluster = preds.column("cluster").to_numpy()

    # All three blobs found, cleanly separated.
    inliers = truth >= 0
    assert adjusted_rand_score(truth[inliers], cluster[inliers]) > 0.95
    metrics = eval_metrics(events)
    assert set(metrics) == EVAL_METRICS
    assert metrics["eval/n_clusters"] == 3
    assert metrics["eval/silhouette"] > 0.5
    assert metrics["eval/davies_bouldin"] < 1.0
    if algorithm == "dbscan":
        assert set(cluster[~inliers]) == {-1}, "planted outliers must be noise"
        assert (cluster[inliers] == -1).mean() < 0.05
        assert metrics["eval/noise_fraction"] == pytest.approx((cluster == -1).mean())
    else:
        assert metrics["eval/noise_fraction"] == 0.0
        selection = [e for e in events if e["t"] == "metric" and e["name"] == "select/silhouette"]
        assert [e["step"] for e in selection] == list(range(2, 11))
    if algorithm == "pca_kmeans":
        pca = {e["name"]: e for e in events if e["t"] == "metric" and e["name"].startswith("pca/")}
        assert set(pca) == {"pca/explained_variance", "pca/components"}
        assert pca["pca/explained_variance"]["value"] >= 0.95
        assert 1 <= pca["pca/components"]["value"] <= 6

    # `predict` from the saved model reproduces the fit-time clusters and emits the same metrics.
    again, pred_events = predict_again(tmp_path, algorithm, "clustering", manifest["valid"])
    assert again.column("cluster").to_pylist() == preds.column("cluster").to_pylist()
    assert eval_metrics(pred_events) == pytest.approx(metrics)

    onnx = [e for e in events if e["t"] == "artifact" and e.get("format") == "onnx"]
    if onnx:
        valid = pq.read_table(manifest["valid"])
        np.testing.assert_array_equal(run_onnx(tmp_path / "run" / "model", valid), cluster)
    else:
        assert any(e["t"] == "log" and e["level"] == "warn" and "ONNX" in e["msg"] for e in events)
    if algorithm in {"kmeans", "pca_kmeans"}:
        assert onnx, f"{algorithm} must export to ONNX"


def test_kmeans_fixed_n_clusters(tmp_path: Path) -> None:
    manifest = fit_manifest(tmp_path, "kmeans", "clustering", params={"n_clusters": 4}, export=["native"])
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert eval_metrics(events)["eval/n_clusters"] == 4
    assert not [e for e in events if e["t"] == "metric" and e["name"] == "select/silhouette"]


def test_isolation_forest_contract(tmp_path: Path) -> None:
    manifest = fit_manifest(tmp_path, "isolation_forest", "anomaly_detection", target="labels")
    manifest["positive_class"] = "anomaly"
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    assert events[-1] == {"t": "done", "summary": {}}
    assert not eval_metrics(events), "anomaly detection is scored in Rust, not by the backend"

    truth = make_blobs(80, seed=2)[1]
    preds = pq.read_table(tmp_path / "run" / "valid_pred.parquet")
    assert preds.column_names == ["anomaly_score", "prediction"]
    assert preds.schema.field("anomaly_score").type == pa.float64()
    assert preds.schema.field("prediction").type == pa.string()
    assert preds.num_rows == len(truth)
    score = preds.column("anomaly_score").to_numpy()
    labels = np.asarray(preds.column("prediction").to_pylist())
    assert set(labels) <= {"anomaly", "normal"}

    # Planted outliers rank on top (higher score = more anomalous) and are predicted `anomaly`.
    is_outlier = truth < 0
    assert roc_auc_score(is_outlier, score) > 0.9
    assert set(labels[is_outlier]) == {"anomaly"}
    assert (labels == "anomaly").mean() < 0.2
    assert np.all((labels == "anomaly") == (score > score[labels == "normal"].max()))

    again, _ = predict_again(tmp_path, "isolation_forest", "anomaly_detection", manifest["valid"])
    np.testing.assert_allclose(again.column("anomaly_score").to_numpy(), score, atol=1e-12)
    assert again.column("prediction").to_pylist() == labels.tolist()

    onnx = [e for e in events if e["t"] == "artifact" and e.get("format") == "onnx"]
    assert onnx, "isolation_forest must export to ONNX"
    valid = pq.read_table(manifest["valid"])
    np.testing.assert_allclose(run_onnx(tmp_path / "run" / "model", valid), score, atol=1e-4)


def test_isolation_forest_contamination(tmp_path: Path) -> None:
    manifest = fit_manifest(
        tmp_path, "isolation_forest", "anomaly_detection", params={"contamination": 0.1}, export=["native"]
    )
    proc, _ = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 0, proc.stderr
    labels = pq.read_table(tmp_path / "run" / "valid_pred.parquet").column("prediction").to_pylist()
    assert 0.03 < labels.count("anomaly") / len(labels) < 0.2


@pytest.mark.parametrize(
    ("algorithm", "task"),
    [("kmeans", "clustering"), ("dbscan", "clustering"), ("isolation_forest", "anomaly_detection")],
)
def test_fit_never_reads_target(tmp_path: Path, algorithm: str, task: str) -> None:
    """Results are identical with no `__target__`, real labels, or garbage in it."""
    outputs: list[pa.Table] = []
    for target in (None, "labels", "garbage"):
        tmp = tmp_path / str(target)
        manifest = fit_manifest(tmp, algorithm, task, target=target, export=["native"])
        proc, _ = run_manifest("fit", manifest, tmp / "run.json")
        assert proc.returncode == 0, proc.stderr
        outputs.append(pq.read_table(tmp / "run" / "valid_pred.parquet"))
    assert outputs[0].equals(outputs[1])
    assert outputs[0].equals(outputs[2])


@pytest.mark.parametrize(
    ("algorithm", "task"),
    [
        ("kmeans", "regression"),
        ("isolation_forest", "clustering"),
        ("dbscan", "anomaly_detection"),
        ("lightgbm", "clustering"),
    ],
)
def test_unsupported_task_exits_2(tmp_path: Path, algorithm: str, task: str) -> None:
    manifest = fit_manifest(tmp_path, algorithm, task, target="labels")
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 2, proc.stderr
    assert "does not support task" in events[-1]["msg"]


def test_supervised_still_requires_target(tmp_path: Path) -> None:
    manifest = fit_manifest(tmp_path, "logistic_regression", "binary_classification", classes=["a", "b"])
    proc, events = run_manifest("fit", manifest, tmp_path / "run.json")
    assert proc.returncode == 2
    assert "__target__" in events[-1]["msg"]

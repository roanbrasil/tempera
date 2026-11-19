"""`quality` (label issues, Data-Audit A7) and `fairness` (A8) commands through the CLI subprocess.

Synthetic data with planted problems: ~5% flipped labels must surface in the label-issue report, and a group
with a clearly lower selection rate must show up in the fairness summary.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("cleanlab")
pytest.importorskip("fairlearn")
pytest.importorskip("pandas")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from helpers import run_manifest, validate_lines, write_manifest

from tempera_worker.cli import run_quality
from tempera_worker.events import EventSink
from tempera_worker.manifest import load_quality
from tempera_worker.quality import label_issues

pytestmark = pytest.mark.quality

FEATURES = [
    {"name": "x1", "kind": "numeric"},
    {"name": "x2", "kind": "numeric"},
    {"name": "plan", "kind": "categorical"},
    {"name": "flag", "kind": "boolean"},
]
FLIP_RATE = 0.05


def _noisy_dataset(path: Path, task: str, n: int = 1500, seed: int = 7) -> tuple[list[str], set[int]]:
    """Well-separated classes, then FLIP_RATE of the labels moved to another class. Returns flipped rows."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    plan = rng.choice(["basic", "pro", "team"], size=n)
    flag = rng.random(n) < 0.5
    score = 2.0 * x1 + x2 + np.where(plan == "pro", 0.5, 0.0)
    if task == "binary_classification":
        classes = ["no", "yes"]
        label = (score > 0).astype(np.int64)
    else:
        classes = ["high", "low", "mid"]
        label = np.where(score > 1.0, 0, np.where(score < -1.0, 1, 2))
    flipped = rng.choice(n, size=int(n * FLIP_RATE), replace=False)
    shift = rng.integers(1, len(classes), size=len(flipped))
    label[flipped] = (label[flipped] + shift) % len(classes)
    target = np.asarray(classes, dtype=object)[label]
    table = pa.table(
        {
            "x1": pa.array(x1),
            "x2": pa.array(x2),
            "plan": pa.array(plan.tolist()),
            "flag": pa.array(flag.tolist()),
            "__target__": pa.array(target.tolist(), type=pa.string()),
        }
    )
    pq.write_table(table, path)
    return classes, {int(i) for i in flipped}


def _quality_manifest(tmp: Path, task: str, **overrides: Any) -> tuple[dict[str, Any], set[int]]:
    classes, flipped = _noisy_dataset(tmp / "train.parquet", task)
    manifest: dict[str, Any] = {
        "task": task,
        "classes": classes,
        "features": FEATURES,
        "data": str(tmp / "train.parquet"),
        "output": str(tmp / "out" / "label_issues.json"),
        "folds": 5,
        "seed": 42,
        "max_rows": 50_000,
        "top_n": 200,
    }
    manifest.update(overrides)
    return manifest, flipped


@pytest.mark.parametrize("task", ["binary_classification", "multiclass_classification"])
def test_quality_finds_planted_label_errors(tmp_path: Path, task: str) -> None:
    manifest, flipped = _quality_manifest(tmp_path, task)
    proc, events = run_manifest("quality", manifest, tmp_path / "m.json")
    assert proc.returncode == 0, proc.stderr
    assert [(e["step"], e["total_steps"]) for e in events if e["t"] == "progress"] == [
        (i, 5) for i in range(1, 6)
    ]
    artifact = next(e for e in events if e["t"] == "artifact")
    assert artifact == {"t": "artifact", "kind": "report", "path": manifest["output"]}
    done = events[-1]
    assert done["t"] == "done"

    report = json.loads(Path(manifest["output"]).read_text())
    assert report["rows_checked"] == 1500
    assert report["subsampled"] is False
    assert done["summary"] == {
        "issues_found": report["issues_found"],
        "estimated_noise_rate": report["estimated_noise_rate"],
    }
    assert 0.02 <= report["estimated_noise_rate"] <= 0.12
    assert set(report["per_class"]) == set(manifest["classes"])
    assert sum(c["rows"] for c in report["per_class"].values()) == 1500
    assert sum(c["issues"] for c in report["per_class"].values()) == report["issues_found"]

    top = report["top"]
    assert len(top) == min(report["issues_found"], manifest["top_n"])
    qualities = [t["quality"] for t in top]
    assert qualities == sorted(qualities)
    assert all(t["given"] in manifest["classes"] and t["suggested"] in manifest["classes"] for t in top)
    found = {t["row"] for t in top} & flipped
    assert len(found) >= 0.6 * len(flipped), f"{len(found)}/{len(flipped)} flipped rows in top"


def test_quality_subsample_keeps_original_row_indices(tmp_path: Path) -> None:
    manifest, flipped = _quality_manifest(
        tmp_path, "binary_classification", max_rows=1000, top_n=1000, folds=3
    )
    proc, _ = run_manifest("quality", manifest, tmp_path / "m.json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(Path(manifest["output"]).read_text())
    assert (report["rows_checked"], report["rows_total"], report["subsampled"]) == (1000, 1500, True)
    rows = [t["row"] for t in report["top"]]
    assert all(0 <= r < 1500 for r in rows) and len(set(rows)) == len(rows)
    # Indices refer to the original file: the given labels must match it.
    target = pq.read_table(manifest["data"]).column("__target__").to_pylist()
    assert all(target[t["row"]] == t["given"] for t in report["top"])
    assert len(set(rows) & flipped) > 0


def test_quality_logistic_fallback_without_lightgbm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(label_issues, "_lightgbm_available", lambda: False)
    manifest, flipped = _quality_manifest(tmp_path, "binary_classification", folds=3)
    stream = io.StringIO()
    code = run_quality(load_quality(write_manifest(tmp_path / "m.json", manifest)), EventSink(stream))
    assert code == 0
    events = validate_lines(stream.getvalue())
    assert "logistic_regression" in events[0]["msg"]
    report = json.loads(Path(manifest["output"]).read_text())
    assert len({t["row"] for t in report["top"]} & flipped) >= 0.6 * len(flipped)


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "regression"},
        {"task": "clustering"},
        {"folds": 1},
        {"classes": ["yes", "no"]},
        {"data": "/nonexistent/train.parquet"},
        {"features": [{"name": "missing_column", "kind": "numeric"}]},
    ],
)
def test_quality_bad_manifest_exits_2(tmp_path: Path, overrides: dict[str, Any]) -> None:
    manifest, _ = _quality_manifest(tmp_path, "binary_classification")
    manifest.update(overrides)
    proc, events = run_manifest("quality", manifest, tmp_path / "m.json")
    assert proc.returncode == 2, proc.stderr
    assert events[-1]["t"] == "error"


# ---- fairness ---------------------------------------------------------------------------------------------


def _fairness_files(tmp: Path, task: str) -> dict[str, Any]:
    """Groups a (300 rows), b (270 rows, much lower selection rate) and c (20 rows: small)."""
    rng = np.random.default_rng(3)
    gender = np.asarray(["a"] * 300 + ["b"] * 270 + ["c"] * 20, dtype=object)
    region = rng.choice(["north", "south"], size=len(gender))
    n = len(gender)
    if task == "regression":
        truth = rng.normal(10.0, 2.0, size=n)
        pred = truth + rng.normal(0.0, np.where(gender == "b", 2.0, 0.5))
        truth_arr, pred_arr = pa.array(truth), pa.array(pred)
        classes: list[str] = []
    else:
        classes = ["no", "yes"] if task == "binary_classification" else ["high", "low", "mid"]
        rate = np.where(gender == "b", 0.15, 0.6)
        positive = rng.random(n) < rate
        if task == "binary_classification":
            pred = np.where(positive, "yes", "no")
            agree = rng.random(n) < 0.85
            truth = np.where(agree, pred, np.where(pred == "yes", "no", "yes"))
        else:
            truth = rng.choice(classes, size=n)
            agree = rng.random(n) < np.where(gender == "b", 0.5, 0.9)
            pred = np.where(agree, truth, rng.choice(classes, size=n))
        truth_arr, pred_arr = pa.array(truth.tolist()), pa.array(pred.tolist())
    pq.write_table(
        pa.table({"x": pa.array(np.arange(n, dtype=np.float64)), "__target__": truth_arr}),
        tmp / "test.parquet",
    )
    pred_cols: dict[str, pa.Array] = {"prediction": pred_arr}
    for c in classes:
        pred_cols[f"proba__{c}"] = pa.array(np.full(n, 1.0 / len(classes)))
    pq.write_table(pa.table(pred_cols), tmp / "test_pred.parquet")
    pq.write_table(
        pa.table({"gender": pa.array(gender.tolist()), "region": pa.array(region.tolist())}),
        tmp / "sensitive_test.parquet",
    )
    sensitive = str(tmp / "sensitive_test.parquet")
    return {
        "task": task,
        "classes": classes,
        "positive_class": "yes" if task == "binary_classification" else None,
        "truth": str(tmp / "test.parquet"),
        "predictions": str(tmp / "test_pred.parquet"),
        "sensitive": {"gender": sensitive, "region": sensitive},
        "output": str(tmp / "out" / "fairness.json"),
    }


def _run_fairness(tmp: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proc, events = run_manifest("fairness", manifest, tmp / "m.json")
    assert proc.returncode == 0, proc.stderr
    assert any(e == {"t": "artifact", "kind": "report", "path": manifest["output"]} for e in events)
    assert events[-1]["t"] == "done"
    return json.loads(Path(manifest["output"]).read_text()), events


def _groups(report: dict[str, Any], attr: str) -> dict[str, dict[str, Any]]:
    return {g["group"]: g for g in report["attributes"][attr]["groups"]}


def test_fairness_binary_flags_disparity_and_small_groups(tmp_path: Path) -> None:
    manifest = _fairness_files(tmp_path, "binary_classification")
    report, events = _run_fairness(tmp_path, manifest)
    gender = _groups(report, "gender")
    assert {k: (g["count"], g["small"]) for k, g in gender.items()} == {
        "a": (300, False),
        "b": (270, False),
        "c": (20, True),
    }
    assert set(gender["a"]["metrics"]) == {"selection_rate", "accuracy", "tpr", "fpr"}
    assert gender["b"]["metrics"]["selection_rate"] < gender["a"]["metrics"]["selection_rate"] - 0.3
    summary = report["attributes"]["gender"]["summary"]
    assert summary["demographic_parity_difference"] > 0.2
    assert 0.0 <= summary["demographic_parity_ratio"] < 0.5
    assert summary["equalized_odds_difference"] is not None
    assert {"equalized_odds_ratio", "equal_opportunity_difference", "accuracy_difference"} <= set(summary)
    assert report["attributes"]["region"]["summary"]["demographic_parity_difference"] < 0.15
    done = events[-1]["summary"]
    assert done["gender/demographic_parity_difference"] == summary["demographic_parity_difference"]


def test_fairness_multiclass_reports_accuracy_only(tmp_path: Path) -> None:
    report, _ = _run_fairness(tmp_path, _fairness_files(tmp_path, "multiclass_classification"))
    gender = _groups(report, "gender")
    assert all(set(g["metrics"]) == {"accuracy"} for g in gender.values())
    assert gender["c"]["small"] is True
    summary = report["attributes"]["gender"]["summary"]
    assert summary["accuracy_difference"] > 0.2
    assert set(summary) == {"accuracy_difference", "accuracy_ratio"}


def test_fairness_regression_metrics(tmp_path: Path) -> None:
    report, _ = _run_fairness(tmp_path, _fairness_files(tmp_path, "regression"))
    gender = _groups(report, "gender")
    assert set(gender["a"]["metrics"]) == {"mae", "rmse", "mean_prediction", "mean_truth"}
    assert gender["b"]["metrics"]["mae"] > 2 * gender["a"]["metrics"]["mae"]
    assert report["attributes"]["gender"]["summary"]["mae_difference"] > 0.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "clustering"},
        {"sensitive": {}},
        {"sensitive": {"ethnicity": "SENSITIVE"}},
        {"positive_class": "maybe"},
        {"truth": "/nonexistent/test.parquet"},
    ],
)
def test_fairness_bad_manifest_exits_2(tmp_path: Path, overrides: dict[str, Any]) -> None:
    manifest = _fairness_files(tmp_path, "binary_classification")
    if overrides.get("sensitive") == {"ethnicity": "SENSITIVE"}:
        # The file exists but has no `ethnicity` column.
        overrides = {"sensitive": {"ethnicity": manifest["sensitive"]["gender"]}}
    manifest.update(overrides)
    proc, events = run_manifest("fairness", manifest, tmp_path / "m.json")
    assert proc.returncode == 2, proc.stderr
    assert events[-1]["t"] == "error"

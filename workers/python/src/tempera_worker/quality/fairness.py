"""Group fairness of a model's test predictions (Data-Audit A8) with Fairlearn.

Per group of every sensitive attribute:

* binary classification: count, selection_rate (share predicted positive), accuracy, tpr, fpr; summary
  with Fairlearn's demographic parity and equalized odds (difference and ratio), plus the equal
  opportunity and accuracy differences;
* multiclass classification: count and accuracy; summary with the accuracy difference and ratio;
* regression: count, mae, rmse, mean_prediction, mean_truth; summary with the max-min differences.

Groups with fewer than `SMALL_GROUP` rows are reported with `"small": true`. Undefined values (e.g. the
TPR of a group without positives) are written as JSON `null`.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)

from tempera_worker.manifest import TARGET_COLUMN, ManifestError

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.events import EventSink
    from tempera_worker.manifest import FairnessManifest

SMALL_GROUP = 30
MISSING_GROUP = "(missing)"


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rate(mask: np.ndarray, of: np.ndarray) -> float | None:
    """Mean of `mask` over the rows where `of` is true (None when there are none)."""
    total = int(np.sum(of))
    return float(np.sum(mask & of)) / total if total else None


def _spread(values: list[float | None]) -> tuple[float | None, float | None]:
    """(max - min, min / max) over the defined values."""
    defined = [v for v in values if v is not None]
    if not defined:
        return None, None
    hi, lo = max(defined), min(defined)
    return hi - lo, (lo / hi if hi > 0 else None)


def _read(path: Any, column: str) -> pa.ChunkedArray:
    import pyarrow.parquet as pq

    try:
        names = pq.read_schema(path).names
    except (OSError, ValueError) as e:
        raise ManifestError(f"cannot read Parquet {path}: {e}") from None
    if column not in names:
        raise ManifestError(f"{path} has no column `{column}`")
    return pq.read_table(path, columns=[column]).column(column)


def _groups(col: pa.ChunkedArray) -> np.ndarray:
    import pyarrow as pa

    try:
        values = col.cast(pa.string()).to_pylist()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise ManifestError(f"sensitive column has unsupported type {col.type}: {e}") from None
    return np.asarray([MISSING_GROUP if v is None else v for v in values], dtype=object)


def _labels(col: pa.ChunkedArray, classes: list[str], what: str) -> np.ndarray:
    import pyarrow as pa

    if col.null_count:
        raise ManifestError(f"{what} has {col.null_count} null values")
    values = np.asarray(col.cast(pa.string()).to_pylist(), dtype=object)
    unknown = sorted(set(values.tolist()) - set(classes))
    if unknown:
        raise ManifestError(f"{what} has labels not in `classes`: {unknown[:5]}")
    return values


def _floats(col: pa.ChunkedArray, what: str) -> np.ndarray:
    import pyarrow as pa

    if col.null_count:
        raise ManifestError(f"{what} has {col.null_count} null values")
    try:
        return np.asarray(col.cast(pa.float64()).to_numpy(), dtype=np.float64)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise ManifestError(f"{what} must be numeric for regression: {e}") from None


GroupMetrics = Callable[[np.ndarray], dict[str, float | None]]


def _binary(y_true: np.ndarray, y_pred: np.ndarray, sf: np.ndarray) -> tuple[GroupMetrics, dict[str, Any]]:
    """Per-group metrics and the Fairlearn summaries; `y_true` / `y_pred` are 0/1 (1 = positive class)."""
    pos_true, pos_pred = y_true == 1, y_pred == 1

    def metrics(g: np.ndarray) -> dict[str, float | None]:
        return {
            "selection_rate": float(np.mean(pos_pred[g])),
            "accuracy": float(np.mean(y_true[g] == y_pred[g])),
            "tpr": _rate(pos_pred, g & pos_true),
            "fpr": _rate(pos_pred, g & ~pos_true),
        }

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        summary: dict[str, Any] = {
            "demographic_parity_difference": demographic_parity_difference(
                y_true, y_pred, sensitive_features=sf
            ),
            "demographic_parity_ratio": demographic_parity_ratio(y_true, y_pred, sensitive_features=sf),
            "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=sf),
            "equalized_odds_ratio": equalized_odds_ratio(y_true, y_pred, sensitive_features=sf),
        }
    return metrics, summary


def _report_attribute(
    sf: np.ndarray, metrics: GroupMetrics, summary: dict[str, Any], spreads: dict[str, str]
) -> dict[str, Any]:
    """Groups sorted by name; `spreads` maps a per-group metric to the summary name of its max-min gap."""
    groups: list[dict[str, Any]] = []
    for name in sorted(set(sf.tolist())):
        g = sf == name
        count = int(np.sum(g))
        groups.append(
            {
                "group": name,
                "count": count,
                "metrics": {k: _finite(v) for k, v in metrics(g).items()},
                "small": count < SMALL_GROUP,
            }
        )
    clean = {k: _finite(v) for k, v in summary.items()}
    for metric, label in spreads.items():
        diff, ratio = _spread([grp["metrics"][metric] for grp in groups])
        clean[f"{label}_difference"] = diff
        if metric == "accuracy":
            clean["accuracy_ratio"] = ratio
    return {"groups": groups, "summary": clean}


def run(m: FairnessManifest, events: EventSink) -> int:
    truth_col = _read(m.truth, TARGET_COLUMN)
    pred_col = _read(m.predictions, "prediction")
    n = len(truth_col)
    if len(pred_col) != n:
        raise ManifestError(f"{m.predictions} has {len(pred_col)} rows but {m.truth} has {n}")
    sensitive: dict[str, np.ndarray] = {}
    for attr, path in m.sensitive.items():
        col = _read(path, attr)
        if len(col) != n:
            raise ManifestError(f"sensitive `{attr}` ({path}) has {len(col)} rows but {m.truth} has {n}")
        sensitive[attr] = _groups(col)

    report: dict[str, Any] = {"task": m.task, "rows": n, "small_group_threshold": SMALL_GROUP}
    attributes: dict[str, Any] = {}
    if m.task == "regression":
        y_true, y_pred = _floats(truth_col, f"`{TARGET_COLUMN}`"), _floats(pred_col, "`prediction`")
        err = y_pred - y_true

        def regression(g: np.ndarray) -> dict[str, float | None]:
            return {
                "mae": float(np.mean(np.abs(err[g]))),
                "rmse": float(np.sqrt(np.mean(err[g] ** 2))),
                "mean_prediction": float(np.mean(y_pred[g])),
                "mean_truth": float(np.mean(y_true[g])),
            }

        spreads = {"mae": "mae", "rmse": "rmse", "mean_prediction": "mean_prediction"}
        for attr, sf in sensitive.items():
            attributes[attr] = _report_attribute(sf, regression, {}, spreads)
    else:
        labels_true = _labels(truth_col, m.classes, f"`{TARGET_COLUMN}`")
        labels_pred = _labels(pred_col, m.classes, "`prediction`")
        if m.task == "binary_classification":
            positive = m.positive_class or m.classes[1]
            if m.positive_class is None:
                events.log(f"fairness: no `positive_class`; using `{positive}`", level="warn")
            report["positive_class"] = positive
            y_true = (labels_true == positive).astype(np.int64)
            y_pred = (labels_pred == positive).astype(np.int64)
            for attr, sf in sensitive.items():
                metrics, summary = _binary(y_true, y_pred, sf)
                attributes[attr] = _report_attribute(
                    sf, metrics, summary, {"tpr": "equal_opportunity", "accuracy": "accuracy"}
                )
        else:
            correct = labels_true == labels_pred

            def multiclass(g: np.ndarray) -> dict[str, float | None]:
                return {"accuracy": float(np.mean(correct[g]))}

            for attr, sf in sensitive.items():
                attributes[attr] = _report_attribute(sf, multiclass, {}, {"accuracy": "accuracy"})
    report["attributes"] = attributes

    m.output.parent.mkdir(parents=True, exist_ok=True)
    m.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    flagged = sum(g["small"] for a in attributes.values() for g in a["groups"])
    events.log(
        f"fairness: {len(attributes)} sensitive attribute(s), {n} rows"
        + (f", {flagged} group(s) below {SMALL_GROUP} rows" if flagged else "")
    )
    events.artifact("report", str(m.output))
    events.done(
        {
            f"{attr}/{name}": value
            for attr, a in attributes.items()
            for name, value in a["summary"].items()
            if value is not None
        }
    )
    return 0

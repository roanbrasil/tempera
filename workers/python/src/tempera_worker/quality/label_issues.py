"""Label-issue detection (Data-Audit A7) with Cleanlab confident learning.

Out-of-fold predicted probabilities come from stratified K-fold cross-validation with a fast model
(LightGBM when installed, else a logistic regression pipeline); Cleanlab flags the rows whose given label
disagrees with them. Suspects are only reported for review, never removed.

Rows beyond `max_rows` are subsampled with the manifest seed; every row in the report is referenced by its
index in the original Parquet file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from cleanlab.count import estimate_joint
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.model_selection import StratifiedKFold

from tempera_worker.manifest import Feature, ManifestError
from tempera_worker.tabular.base import default_threads
from tempera_worker.tabular.data import encode_target
from tempera_worker.tabular.sklearn_models import limit_threads

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.events import EventSink
    from tempera_worker.manifest import QualityManifest


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def _design(table: pa.Table, features: list[Feature], seed: int, use_gbm: bool) -> tuple[Any, Any]:
    """Feature frame and a factory of unfitted models (LightGBM, else a logistic regression pipeline)."""
    if use_gbm:
        import lightgbm as lgb

        from tempera_worker.tabular.data import gbm_frame, learn_categories

        def gbm() -> Any:
            # Small, regularized ensemble: out-of-fold probabilities must not memorize the noisy labels.
            return lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=default_threads(),
                verbose=-1,
            )

        return gbm_frame(table, features, learn_categories(table, features)), gbm

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from tempera_worker.tabular.data import sklearn_frame
    from tempera_worker.tabular.sklearn_models import build_preprocessor

    def logistic() -> Any:
        return Pipeline(
            [
                ("preprocess", build_preprocessor(features)),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )

    return sklearn_frame(table, features), logistic


def _full_proba(proba: np.ndarray, seen: np.ndarray, n_classes: int) -> np.ndarray:
    """Probabilities over every class: a fold may not contain every label."""
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    full[:, np.asarray(seen, dtype=np.int64)] = proba
    return full


def _subsample(table: pa.Table, max_rows: int, seed: int) -> tuple[pa.Table, np.ndarray]:
    """At most `max_rows` rows (seeded, original order kept) and their original row indices."""
    n = table.num_rows
    if n <= max_rows:
        return table, np.arange(n, dtype=np.int64)
    rows = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False)).astype(np.int64)
    return table.take(rows), rows


def run(m: QualityManifest, table: pa.Table, events: EventSink) -> int:
    total = table.num_rows
    table, rows = _subsample(table, m.max_rows, m.seed)
    n, k = table.num_rows, len(m.classes)
    y = encode_target(table, m.task, m.classes)
    if n < m.folds:
        raise ManifestError(
            f"{m.data} has {n} rows; label-issue detection needs at least `folds` ({m.folds})"
        )
    if len(np.unique(y)) < 2:
        raise ManifestError(f"`__target__` in {m.data} has a single label; nothing to check")

    use_gbm = _lightgbm_available()
    model_name = "lightgbm" if use_gbm else "logistic_regression"
    events.log(
        f"label issues: {n} rows"
        + (f" (sampled from {total})" if n < total else "")
        + f", {m.folds}-fold out-of-fold probabilities with {model_name}"
    )
    # A class rarer than `folds` only makes scikit-learn warn (stderr); `_full_proba` covers missing classes.
    folds = list(StratifiedKFold(n_splits=m.folds, shuffle=True, random_state=m.seed).split(rows, y))

    x, make_model = _design(table, m.features, m.seed, use_gbm)
    pred_probs = np.zeros((n, k), dtype=np.float64)
    with limit_threads():
        for i, (train_idx, test_idx) in enumerate(folds):
            model = make_model()
            model.fit(x.iloc[train_idx], y[train_idx])
            pred_probs[test_idx] = _full_proba(model.predict_proba(x.iloc[test_idx]), model.classes_, k)
            events.progress(i + 1, len(folds))
    events.log(f"label issues: out-of-fold accuracy {float(np.mean(pred_probs.argmax(axis=1) == y)):.4f}")

    issues = np.asarray(find_label_issues(labels=y, pred_probs=pred_probs, n_jobs=1), dtype=bool)
    quality = np.asarray(get_label_quality_scores(labels=y, pred_probs=pred_probs), dtype=np.float64)
    joint = np.asarray(estimate_joint(labels=y, pred_probs=pred_probs), dtype=np.float64)
    noise_rate = float(np.clip(1.0 - np.trace(joint), 0.0, 1.0))

    flagged = np.flatnonzero(issues)
    ranked = flagged[np.argsort(quality[flagged], kind="stable")][: m.top_n]
    suggested = pred_probs.argmax(axis=1)
    report: dict[str, Any] = {
        "rows_checked": n,
        "rows_total": total,
        "subsampled": n < total,
        "issues_found": len(flagged),
        "estimated_noise_rate": noise_rate,
        "per_class": {
            label: {"issues": int(np.sum(issues & (y == c))), "rows": int(np.sum(y == c))}
            for c, label in enumerate(m.classes)
        },
        "top": [
            {
                "row": int(rows[i]),
                "given": m.classes[int(y[i])],
                "suggested": m.classes[int(suggested[i])],
                "quality": float(quality[i]),
            }
            for i in ranked
        ],
    }
    m.output.parent.mkdir(parents=True, exist_ok=True)
    m.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    events.log(f"label issues: {len(flagged)} suspects, estimated noise rate {noise_rate:.3f}")
    events.artifact("report", str(m.output))
    events.done({"issues_found": float(len(flagged)), "estimated_noise_rate": noise_rate})
    return 0

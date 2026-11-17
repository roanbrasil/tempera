"""Model-ready Parquet -> pandas/numpy, and predictions -> Arrow (Contracts.md, section 7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from tempera_worker.manifest import TARGET_COLUMN, Feature, ManifestError, Task

# Missing categorical values are encoded as the empty string for scikit-learn pipelines (and their ONNX
# export), because skl2onnx can only impute string columns against a string sentinel.
MISSING_CATEGORY = ""


def _column(table: pa.Table, name: str) -> pa.ChunkedArray:
    if name not in table.column_names:
        raise ManifestError(f"column `{name}` is missing from the input Parquet")
    return table.column(name)


def numeric_array(table: pa.Table, feature: Feature) -> np.ndarray:
    """Numeric and boolean features as float64; nulls become NaN, booleans 0/1."""
    col = _column(table, feature.name)
    try:
        floats = col.cast(pa.float64())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise ManifestError(f"feature `{feature.name}` ({feature.kind}) has type {col.type}: {e}") from None
    return np.asarray(floats.to_numpy(), dtype=np.float64)


def string_values(table: pa.Table, feature: Feature) -> list[str | None]:
    col = _column(table, feature.name)
    try:
        return col.cast(pa.string()).to_pylist()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise ManifestError(f"feature `{feature.name}` (categorical) has type {col.type}: {e}") from None


def learn_categories(table: pa.Table, features: list[Feature]) -> dict[str, list[str]]:
    """Category list per categorical feature, fixed from the training data (sorted, nulls excluded)."""
    return {
        f.name: sorted({v for v in string_values(table, f) if v is not None})
        for f in features
        if f.kind == "categorical"
    }


def sklearn_frame(table: pa.Table, features: list[Feature]) -> pd.DataFrame:
    """Frame for scikit-learn pipelines: float columns, categorical as object strings ("" = missing)."""
    data: dict[str, Any] = {}
    for f in features:
        if f.kind == "categorical":
            values = [MISSING_CATEGORY if v is None else v for v in string_values(table, f)]
            data[f.name] = pd.Series(values, dtype=object)
        else:
            data[f.name] = numeric_array(table, f)
    return pd.DataFrame(data, columns=[f.name for f in features])


def gbm_frame(table: pa.Table, features: list[Feature], categories: dict[str, list[str]]) -> pd.DataFrame:
    """Frame for LightGBM/XGBoost: float columns, categorical as pandas `category` with fixed categories.

    Values not seen in training become missing.
    """
    data: dict[str, Any] = {}
    for f in features:
        if f.kind == "categorical":
            dtype = pd.CategoricalDtype(categories=categories[f.name], ordered=False)
            data[f.name] = pd.Series(string_values(table, f), dtype=object).astype(dtype)
        else:
            data[f.name] = numeric_array(table, f)
    return pd.DataFrame(data, columns=[f.name for f in features])


def gbm_matrix(table: pa.Table, features: list[Feature], categories: dict[str, list[str]]) -> np.ndarray:
    """Dense float32 matrix used as the single ONNX input of GBM models (categorical -> code, NaN missing)."""
    frame = gbm_frame(table, features, categories)
    cols: list[np.ndarray] = []
    for f in features:
        if f.kind == "categorical":
            codes = np.asarray(frame[f.name].cat.codes, dtype=np.float64)
            cols.append(np.where(codes < 0, np.nan, codes))
        else:
            cols.append(np.asarray(frame[f.name], dtype=np.float64))
    return np.column_stack(cols).astype(np.float32)


def encode_target(table: pa.Table, task: Task, classes: list[str]) -> np.ndarray:
    """`__target__` as class indices (position in `classes`) or float64 values."""
    col = _column(table, TARGET_COLUMN)
    if col.null_count:
        raise ManifestError(f"`{TARGET_COLUMN}` has {col.null_count} null values")
    if task == "regression":
        try:
            return np.asarray(col.cast(pa.float64()).to_numpy(), dtype=np.float64)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
            raise ManifestError(f"`{TARGET_COLUMN}` must be numeric for regression: {e}") from None
    index = {label: i for i, label in enumerate(classes)}
    labels = col.cast(pa.string()).to_pylist()
    unknown = sorted({label for label in labels if label not in index})
    if unknown:
        raise ManifestError(f"`{TARGET_COLUMN}` has labels not in `classes`: {unknown[:5]}")
    return np.fromiter((index[label] for label in labels), dtype=np.int64, count=len(labels))


def classification_predictions(proba: np.ndarray, classes: list[str]) -> pa.Table:
    """`prediction` (Utf8) + `proba__<label>` (Float64) for every class, in `classes` order."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(f"expected probabilities of shape (n, {len(classes)}), got {proba.shape}")
    labels = np.asarray(classes, dtype=object)[proba.argmax(axis=1)] if len(proba) else []
    columns: dict[str, pa.Array] = {"prediction": pa.array(list(labels), type=pa.string())}
    for i, label in enumerate(classes):
        columns[f"proba__{label}"] = pa.array(proba[:, i], type=pa.float64())
    return pa.table(columns)


def regression_predictions(values: np.ndarray) -> pa.Table:
    return pa.table({"prediction": pa.array(np.asarray(values, dtype=np.float64).ravel(), type=pa.float64())})

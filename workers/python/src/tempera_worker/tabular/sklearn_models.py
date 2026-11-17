"""scikit-learn adapters: logistic/linear/ridge regression and random forest.

Every model is a `Pipeline(preprocess, estimator)`:

* numeric: median imputation + standard scaling (no scaling for random forest: trees are scale-invariant,
  and scaling in float32 would make the ONNX export drift from the native model near split thresholds);
* boolean: 0/1, most-frequent imputation;
* categorical: most-frequent imputation + one-hot (unknown categories ignored).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tempera_worker.manifest import Feature
from tempera_worker.tabular.base import CLASSIFICATION_TASKS, TabularAlgorithm, default_threads
from tempera_worker.tabular.data import MISSING_CATEGORY, encode_target, sklearn_frame

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext


def build_preprocessor(features: list[Feature], scale: bool = True) -> ColumnTransformer:
    numeric = [f.name for f in features if f.kind == "numeric"]
    boolean = [f.name for f in features if f.kind == "boolean"]
    categorical = [f.name for f in features if f.kind == "categorical"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if boolean:
        imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
        transformers.append(("boolean", imputer, boolean))
    if categorical:
        cat_steps: list[tuple[str, Any]] = [
            (
                "impute",
                SimpleImputer(
                    missing_values=cast(Any, MISSING_CATEGORY),
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), categorical))
    return ColumnTransformer(transformers, remainder="drop")


class SklearnAlgorithm(TabularAlgorithm):
    native_file: ClassVar[str] = "model.joblib"
    scale_numeric: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Pipeline | None = None

    def _estimator(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        x = sklearn_frame(train, self.features)
        y = encode_target(train, self.task, self.classes)
        if self.is_classification and len(np.unique(y)) < 2:
            raise ValueError("the training set has a single class")
        if valid is not None:
            ctx.log(f"`{self.name}` has no early stopping; the validation set is only predicted")
        ctx.progress(0, 1)
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(self.features, scale=self.scale_numeric)),
                ("model", self._estimator(dict(self.params))),
            ]
        )
        pipeline.fit(x, y)
        self.pipeline = pipeline
        ctx.progress(1, 1)

    def _model(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        return self.pipeline

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        pipeline = self._model()
        x = sklearn_frame(data, self.features)
        if not self.is_classification:
            return np.asarray(pipeline.predict(x), dtype=np.float64)
        # The estimator only knows the classes seen in training; scatter into the full class list.
        seen = np.asarray(pipeline.classes_, dtype=np.int64)
        proba = np.zeros((len(x), len(self.classes)), dtype=np.float64)
        proba[:, seen] = pipeline.predict_proba(x)
        return proba

    def _save_native(self, path: Path) -> None:
        joblib.dump(self._model(), path)

    def _load_native(self, path: Path) -> None:
        self.pipeline = joblib.load(path)

    # ---- importance ------------------------------------------------------------------------------------

    def _raw_importance(self) -> np.ndarray:
        """One value per transformed column."""
        raise NotImplementedError

    def feature_importance(self) -> dict[str, float]:
        """Aggregate per transformed column back to the original features (one-hot columns are summed)."""
        pipeline = self._model()
        pre: ColumnTransformer = pipeline.named_steps["preprocess"]
        raw = self._raw_importance()
        out: dict[str, float] = {}
        for name, _transformer, columns in pre.transformers_:
            if name == "remainder":
                continue
            block = pre.output_indices_[name]
            values = raw[block]
            if name == "categorical":
                onehot: OneHotEncoder = pre.named_transformers_["categorical"].named_steps["onehot"]
                start = 0
                for column, cats in zip(columns, onehot.categories_, strict=True):
                    out[column] = float(values[start : start + len(cats)].sum())
                    start += len(cats)
            else:
                for column, value in zip(columns, values, strict=True):
                    out[column] = float(value)
        return {f.name: out.get(f.name, 0.0) for f in self.features}

    # ---- ONNX ------------------------------------------------------------------------------------------

    def _to_onnx(self) -> bytes:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        pipeline = self._model()
        types: list[tuple[str, Any]] = [
            (f.name, StringTensorType([None, 1]) if f.kind == "categorical" else FloatTensorType([None, 1]))
            for f in self.features
        ]
        options = {id(pipeline.named_steps["model"]): {"zipmap": False}} if self.is_classification else None
        onx: Any = convert_sklearn(pipeline, initial_types=types, options=options)
        self.onnx_meta = {
            "inputs": "one tensor per feature, shape [n, 1]: float32 for numeric/boolean (NaN = missing), "
            f"string for categorical ({MISSING_CATEGORY!r} = missing)",
            "outputs": {"label": "class index", "probabilities": "float32 [n, classes]"}
            if self.is_classification
            else {"variable": "float32 [n, 1]"},
        }
        return onx.SerializeToString()

    def onnx_feed(self, data: pa.Table) -> dict[str, np.ndarray]:
        frame = sklearn_frame(data, self.features)
        feed: dict[str, np.ndarray] = {}
        for f in self.features:
            column = frame[[f.name]].to_numpy()
            feed[f.name] = column.astype(str) if f.kind == "categorical" else column.astype(np.float32)
        return feed


class LinearImportance(SklearnAlgorithm):
    def _raw_importance(self) -> np.ndarray:
        coef = np.abs(np.asarray(self._model().named_steps["model"].coef_, dtype=np.float64))
        return coef if coef.ndim == 1 else coef.mean(axis=0)


class LogisticRegressionAlgorithm(LinearImportance):
    name: ClassVar[str] = "logistic_regression"
    tasks: ClassVar[frozenset[str]] = CLASSIFICATION_TASKS

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", self.seed)
        return LogisticRegression(**params)


class LinearRegressionAlgorithm(LinearImportance):
    name: ClassVar[str] = "linear_regression"
    tasks: ClassVar[frozenset[str]] = frozenset({"regression"})

    def _estimator(self, params: dict[str, Any]) -> Any:
        return LinearRegression(**params)


class RidgeAlgorithm(LinearImportance):
    name: ClassVar[str] = "ridge"
    tasks: ClassVar[frozenset[str]] = frozenset({"regression"})

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("random_state", self.seed)
        return Ridge(**params)


class RandomForestAlgorithm(SklearnAlgorithm):
    name: ClassVar[str] = "random_forest"
    scale_numeric: ClassVar[bool] = False

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("n_estimators", 200)
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", default_threads())
        cls = RandomForestClassifier if self.is_classification else RandomForestRegressor
        return cls(**params)

    def _raw_importance(self) -> np.ndarray:
        return np.asarray(self._model().named_steps["model"].feature_importances_, dtype=np.float64)

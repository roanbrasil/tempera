"""scikit-learn adapters: linear models, tree ensembles, kNN, SVM, naive Bayes and MLP.

Every model is a `Pipeline(preprocess, estimator)`:

* numeric: median imputation + standard scaling (no scaling for tree models: trees are scale-invariant,
  and scaling in float32 would make the ONNX export drift from the native model near split thresholds);
* boolean: 0/1, most-frequent imputation;
* categorical: most-frequent imputation + one-hot (unknown categories ignored).

`hist_gradient_boosting` is the exception: it handles missing values natively and gets categorical
features ordinal-encoded so it can use its native categorical splits.

Feature importance is reported only where the model has a meaningful one (coefficients,
`feature_importances_`); kNN, kernel SVMs, naive Bayes, MLP and histogram boosting report none.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

from tempera_worker.manifest import Feature, ManifestError
from tempera_worker.tabular.base import CLASSIFICATION_TASKS, TabularAlgorithm, default_threads
from tempera_worker.tabular.data import MISSING_CATEGORY, encode_target, sklearn_frame

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext


@contextmanager
def limit_threads() -> Iterator[None]:
    """Cap OpenMP/BLAS threads (histogram boosting, k-means, linear algebra) at `TEMPERA_NUM_THREADS`."""
    with threadpool_limits(limits=default_threads()):
        yield


def build_preprocessor(features: list[Feature], scale: bool = True, dense: bool = False) -> ColumnTransformer:
    """Impute + scale + one-hot. `dense=True` always yields a dense matrix (naive Bayes, PCA, ...)."""
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0 if dense else 0.3)


# Histogram boosting bins categorical codes; cardinality must stay below its `max_bins` (255).
HGB_MAX_CATEGORIES = 255


def build_ordinal_preprocessor(features: list[Feature]) -> ColumnTransformer:
    """Numeric/boolean passthrough (NaN kept), categorical -> ordinal codes (unknown -> NaN), in that order.

    The missing-category sentinel ("") is a category of its own, which tree splits handle fine.
    """
    numeric = [f.name for f in features if f.kind != "categorical"]
    categorical = [f.name for f in features if f.kind == "categorical"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", "passthrough", numeric))
    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            max_categories=HGB_MAX_CATEGORIES,
            dtype=np.float64,
        )
        transformers.append(("categorical", encoder, categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


class SklearnAlgorithm(TabularAlgorithm):
    native_file: ClassVar[str] = "model.joblib"
    scale_numeric: ClassVar[bool] = True
    dense: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.pipeline: Pipeline | None = None

    def _estimator(self, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    def _preprocessor(self) -> ColumnTransformer:
        return build_preprocessor(self.features, scale=self.scale_numeric, dense=self.dense)

    def _training_rows(
        self, x: pd.DataFrame, y: np.ndarray, params: dict[str, Any], ctx: RunContext
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Hook to subsample the training set (may consume keys of `params`)."""
        return x, y

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        x = sklearn_frame(train, self.features)
        y = encode_target(train, self.task, self.classes)
        if self.is_classification and len(np.unique(y)) < 2:
            raise ValueError("the training set has a single class")
        if valid is not None:
            ctx.log(f"`{self.name}` does not train on the validation set; it is only predicted")
        ctx.progress(0, 1)
        params = dict(self.params)
        x, y = self._training_rows(x, y, params, ctx)
        pipeline = Pipeline([("preprocess", self._preprocessor()), ("model", self._estimator(params))])
        with limit_threads():
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
        with limit_threads():
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

    def _raw_importance(self) -> np.ndarray | None:
        """One value per transformed column, or None when the model has no meaningful importance."""
        return None

    def feature_importance(self) -> dict[str, float]:
        """Aggregate per transformed column back to the original features (one-hot columns are summed)."""
        pipeline = self._model()
        pre: ColumnTransformer = pipeline.named_steps["preprocess"]
        raw = self._raw_importance()
        if raw is None:
            return {}
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


def abs_coef_importance(coef: Any) -> np.ndarray:
    """|coef| per transformed column, averaged over classes (or class pairs) when 2-D."""
    dense = coef.toarray() if hasattr(coef, "toarray") else coef
    values = np.abs(np.asarray(dense, dtype=np.float64))
    return values if values.ndim == 1 else values.mean(axis=0)


class LinearImportance(SklearnAlgorithm):
    def _raw_importance(self) -> np.ndarray | None:
        return abs_coef_importance(self._model().named_steps["model"].coef_)


class TreeImportance(SklearnAlgorithm):
    scale_numeric: ClassVar[bool] = False

    def _raw_importance(self) -> np.ndarray | None:
        return np.asarray(self._model().named_steps["model"].feature_importances_, dtype=np.float64)


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


class LassoAlgorithm(LinearImportance):
    name: ClassVar[str] = "lasso"
    tasks: ClassVar[frozenset[str]] = frozenset({"regression"})

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("alpha", 0.01)
        params.setdefault("max_iter", 10_000)
        params.setdefault("random_state", self.seed)
        return Lasso(**params)


class ElasticNetAlgorithm(LinearImportance):
    name: ClassVar[str] = "elastic_net"
    tasks: ClassVar[frozenset[str]] = frozenset({"regression"})

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("alpha", 0.01)
        params.setdefault("l1_ratio", 0.5)
        params.setdefault("max_iter", 10_000)
        params.setdefault("random_state", self.seed)
        return ElasticNet(**params)


class SgdAlgorithm(LinearImportance):
    """Linear model trained by SGD; classification uses the log loss so probabilities exist."""

    name: ClassVar[str] = "sgd"

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("max_iter", 2000)
        params.setdefault("random_state", self.seed)
        if self.is_classification:
            params.setdefault("loss", "log_loss")
            if params["loss"] not in {"log_loss", "modified_huber"}:
                raise ManifestError("`sgd` classification needs loss `log_loss` or `modified_huber`")
            return SGDClassifier(**params)
        return SGDRegressor(**params)


class RandomForestAlgorithm(TreeImportance):
    name: ClassVar[str] = "random_forest"

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("n_estimators", 200)
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", default_threads())
        cls = RandomForestClassifier if self.is_classification else RandomForestRegressor
        return cls(**params)


class ExtraTreesAlgorithm(TreeImportance):
    name: ClassVar[str] = "extra_trees"

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("n_estimators", 200)
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", default_threads())
        cls = ExtraTreesClassifier if self.is_classification else ExtraTreesRegressor
        return cls(**params)


class AdaBoostAlgorithm(TreeImportance):
    name: ClassVar[str] = "adaboost"

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("n_estimators", 100)
        params.setdefault("random_state", self.seed)
        cls = AdaBoostClassifier if self.is_classification else AdaBoostRegressor
        return cls(**params)


class HistGradientBoostingAlgorithm(SklearnAlgorithm):
    """Histogram gradient boosting with native missing-value and categorical support.

    Early stopping uses scikit-learn's internal validation split (`early_stopping="auto"`: on above 10,000
    rows); the manifest's validation set is only predicted.
    """

    name: ClassVar[str] = "hist_gradient_boosting"

    def _preprocessor(self) -> ColumnTransformer:
        return build_ordinal_preprocessor(self.features)

    def _estimator(self, params: dict[str, Any]) -> Any:
        n_categorical = sum(f.kind == "categorical" for f in self.features)
        n_other = len(self.features) - n_categorical
        if n_categorical:
            # Ordinal codes come last (see `build_ordinal_preprocessor`).
            params.setdefault("categorical_features", list(range(n_other, n_other + n_categorical)))
        params.setdefault("random_state", self.seed)
        cls = HistGradientBoostingClassifier if self.is_classification else HistGradientBoostingRegressor
        return cls(**params)


class KnnAlgorithm(SklearnAlgorithm):
    name: ClassVar[str] = "knn"

    def _estimator(self, params: dict[str, Any]) -> Any:
        # Uniform weights (scikit-learn's default): skl2onnx's distance weighting does not match sklearn.
        params.setdefault("n_neighbors", 15)
        params.setdefault("n_jobs", default_threads())
        cls = KNeighborsClassifier if self.is_classification else KNeighborsRegressor
        return cls(**params)


# Kernel SVMs scale quadratically or worse with the number of rows.
SVM_MAX_TRAIN_ROWS = 20_000


class SvmAlgorithm(SklearnAlgorithm):
    """SVC with Platt-scaled probabilities / SVR. Training sets above 20,000 rows are subsampled."""

    name: ClassVar[str] = "svm"

    def _training_rows(
        self, x: pd.DataFrame, y: np.ndarray, params: dict[str, Any], ctx: RunContext
    ) -> tuple[pd.DataFrame, np.ndarray]:
        limit = int(params.pop("max_train_rows", SVM_MAX_TRAIN_ROWS))
        if len(x) <= limit:
            return x, y
        rng = np.random.default_rng(self.seed)
        keep = np.sort(rng.choice(len(x), size=limit, replace=False))
        ctx.events.log(
            f"`svm` subsampled the training set from {len(x)} to {limit} rows (seed {self.seed}); "
            "kernel SVMs do not scale to larger sets",
            level="warn",
        )
        return x.iloc[keep].reset_index(drop=True), y[keep]

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.pop("max_train_rows", None)
        if self.is_classification:
            params.setdefault("probability", True)
            params.setdefault("random_state", self.seed)
            if not params["probability"]:
                raise ManifestError("`svm` classification needs `probability: true`")
            return SVC(**params)
        return SVR(**params)

    def _raw_importance(self) -> np.ndarray | None:
        model = self._model().named_steps["model"]
        # Only a linear kernel has per-feature weights; an RBF/poly kernel has no honest importance.
        return abs_coef_importance(model.coef_) if model.kernel == "linear" else None


class NaiveBayesAlgorithm(SklearnAlgorithm):
    """Gaussian naive Bayes on the dense preprocessed matrix (scaled numerics + one-hot)."""

    name: ClassVar[str] = "naive_bayes"
    tasks: ClassVar[frozenset[str]] = CLASSIFICATION_TASKS
    dense: ClassVar[bool] = True

    def _estimator(self, params: dict[str, Any]) -> Any:
        return GaussianNB(**params)


class MlpAlgorithm(SklearnAlgorithm):
    """Multi-layer perceptron with early stopping on an internal validation split."""

    name: ClassVar[str] = "mlp"

    def _estimator(self, params: dict[str, Any]) -> Any:
        params.setdefault("hidden_layer_sizes", (64, 32))
        params.setdefault("early_stopping", True)
        params.setdefault("max_iter", 500)
        params.setdefault("random_state", self.seed)
        if isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        cls = MLPClassifier if self.is_classification else MLPRegressor
        return cls(**params)

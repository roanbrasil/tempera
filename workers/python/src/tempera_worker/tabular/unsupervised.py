"""Unsupervised adapters (Contracts.md, section 8): clustering and anomaly detection.

Every model is a scikit-learn `Pipeline` over the same preprocessing as the supervised adapters (impute +
scale + one-hot, always dense), so cluster quality is measured in the model's own feature space.

* `kmeans`: k in 2..10 chosen by silhouette on a sample of the validation (else training) rows, unless
  `n_clusters` is given.
* `dbscan`: `eps` from the elbow of the sorted k-distance curve (k = `min_samples`, default 5) when not
  given. New rows join the cluster of their nearest core sample within `eps`, else `-1` (noise).
* `pca_kmeans`: PCA keeping 95% of the variance, then `kmeans` in PCA space.
* `isolation_forest`: `anomaly_score` is the negated `score_samples` (higher = more anomalous); rows above
  the `contamination` threshold (default `"auto"`) are predicted `anomaly`.

None of them ever sees `__target__`: the CLI does not even load it for these tasks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyarrow as pa
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from tempera_worker.manifest import ManifestError
from tempera_worker.tabular.base import default_threads
from tempera_worker.tabular.data import sklearn_frame
from tempera_worker.tabular.sklearn_models import SklearnAlgorithm, build_preprocessor, limit_threads

if TYPE_CHECKING:
    from tempera_worker.context import RunContext

CLUSTERING: frozenset[str] = frozenset({"clustering"})
ANOMALY: frozenset[str] = frozenset({"anomaly_detection"})
NOISE = -1
K_RANGE = range(2, 11)
SELECTION_ROWS = 2000  # silhouette is O(n^2): k is chosen on a sample
EVAL_ROWS = 10_000  # section 8: silhouette on up to 10,000 rows
EPS_ROWS = 5000  # rows used to estimate DBSCAN's eps
MIN_CLUSTER_SHARE = 0.01  # k-means candidates with smaller clusters are outlier splits


def _sample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]


def cluster_metrics(x: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    """`eval/*` metrics of section 8; silhouette and Davies-Bouldin exclude noise and need >= 2 clusters."""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    clustered = labels != NOISE
    n_clusters = len(np.unique(labels[clustered]))
    out: dict[str, float] = {
        "eval/n_clusters": float(n_clusters),
        "eval/noise_fraction": float((~clustered).sum() / n) if n else 0.0,
    }
    xs, ls = x[clustered], labels[clustered]
    if n_clusters >= 2 and len(ls) > n_clusters:
        if len(ls) > EVAL_ROWS:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(ls), size=EVAL_ROWS, replace=False))
            # A sample can lose a small cluster; only score it when >= 2 clusters remain.
            xe, le = xs[keep], ls[keep]
        else:
            xe, le = xs, ls
        if 2 <= len(np.unique(le)) < len(le):
            out["eval/silhouette"] = float(silhouette_score(xe, le))
        out["eval/davies_bouldin"] = float(davies_bouldin_score(xs, ls))
    return out


def knee_point(sorted_values: np.ndarray) -> float:
    """Value at the elbow of an increasing curve: the point farthest from the first-last chord."""
    y = np.asarray(sorted_values, dtype=np.float64)
    if len(y) < 3 or y[-1] <= y[0]:
        return float(y[-1]) if len(y) else 0.0
    x = np.linspace(0.0, 1.0, len(y))
    yn = (y - y[0]) / (y[-1] - y[0])
    return float(y[int(np.argmax(x - yn))])


class UnsupervisedAlgorithm(SklearnAlgorithm):
    """Shared parts: dense preprocessing, no target, no feature importance."""

    dense: ClassVar[bool] = True

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        if train.num_rows < 2:
            raise ValueError(f"`{self.name}` needs at least 2 training rows")
        ctx.progress(0, 1)
        params = dict(self.params)
        with limit_threads():
            self._fit_unsupervised(sklearn_frame(train, self.features), valid, params, ctx)

    def _fit_unsupervised(
        self, x: Any, valid: pa.Table | None, params: dict[str, Any], ctx: RunContext
    ) -> None:
        raise NotImplementedError

    def feature_importance(self) -> dict[str, float]:
        return {}

    def _onnx_outputs(self) -> dict[str, str]:
        raise NotImplementedError

    def _target_opset(self) -> dict[str, int] | None:
        return None

    def _to_onnx(self) -> bytes:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        types: list[tuple[str, Any]] = [
            (f.name, StringTensorType([None, 1]) if f.kind == "categorical" else FloatTensorType([None, 1]))
            for f in self.features
        ]
        onx: Any = convert_sklearn(self._model(), initial_types=types, target_opset=self._target_opset())
        self.onnx_meta = {
            "inputs": "one tensor per feature, shape [n, 1]: float32 for numeric/boolean (NaN = missing), "
            "string for categorical ('' = missing)",
            "outputs": self._onnx_outputs(),
        }
        return onx.SerializeToString()


# ---- clustering ------------------------------------------------------------------------------------------


class ClusteringAlgorithm(UnsupervisedAlgorithm):
    tasks: ClassVar[frozenset[str]] = CLUSTERING

    def _space(self, data: pa.Table) -> np.ndarray:
        """Rows in the model's own feature space (everything before the clustering step)."""
        with limit_threads():
            return np.asarray(
                self._model()[:-1].transform(sklearn_frame(data, self.features)), dtype=np.float64
            )

    def _labels(self, space: np.ndarray) -> np.ndarray:
        with limit_threads():
            return np.asarray(self._model()[-1].predict(space), dtype=np.int64)

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        return self._labels(self._space(data))

    def predict(self, data: pa.Table) -> pa.Table:
        return pa.table({"cluster": pa.array(self._raw_predict(data), type=pa.int64())})

    def evaluate(self, data: pa.Table, predictions: pa.Table) -> dict[str, float]:
        labels = np.asarray(predictions.column("cluster").to_numpy(), dtype=np.int64)
        return cluster_metrics(self._space(data), labels, self.seed)

    def _choose_kmeans(
        self, x_train: np.ndarray, x_select: np.ndarray, params: dict[str, Any], ctx: RunContext
    ) -> KMeans:
        """Fit k-means; without `n_clusters`, try k in 2..10 and keep the best silhouette on `x_select`."""
        params.setdefault("n_init", 10)
        params.setdefault("random_state", self.seed)
        if "n_clusters" in params:
            model = KMeans(**params).fit(x_train)
            ctx.progress(1, 1)
            return model
        sample = _sample(x_select, SELECTION_ROWS, self.seed)
        ks = [k for k in K_RANGE if k < min(len(x_train), len(sample))]
        if not ks:
            raise ValueError(f"`{self.name}` needs more rows to choose the number of clusters")
        # Silhouette keeps rising as k-means isolates a handful of outliers into tiny clusters, so a k whose
        # solution has a cluster below 1% of the training rows (at least 2) only wins if every k does.
        min_size = max(2, math.ceil(MIN_CLUSTER_SHARE * len(x_train)))
        best: tuple[float, KMeans] | None = None
        best_any: tuple[float, KMeans] | None = None
        for i, k in enumerate(ks, start=1):
            model = KMeans(n_clusters=k, **params).fit(x_train)
            labels = model.predict(sample)
            score = float(silhouette_score(sample, labels)) if len(np.unique(labels)) >= 2 else -1.0
            ctx.log_metric("select/silhouette", score, step=k)
            ctx.progress(i, len(ks))
            if best_any is None or score > best_any[0]:
                best_any = (score, model)
            sizes = np.bincount(np.asarray(model.labels_, dtype=np.int64), minlength=k)
            if sizes.min() >= min_size and (best is None or score > best[0]):
                best = (score, model)
        chosen = best or best_any
        assert chosen is not None
        ctx.log(f"`{self.name}` chose k={chosen[1].n_clusters} (silhouette {chosen[0]:.3f})")
        return chosen[1]

    def _onnx_outputs(self) -> dict[str, str]:
        return {"label": "int64 [n] cluster index", "scores": "float32 [n, k] distance to each centroid"}

    def score_onnx(self, onnx_path: Path, data: pa.Table) -> np.ndarray:
        """Cluster labels (as float64) from an exported ONNX model."""
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        (labels,) = session.run(["label"], self.onnx_feed(data))
        return np.asarray(labels, dtype=np.float64).ravel()


class KMeansAlgorithm(ClusteringAlgorithm):
    name: ClassVar[str] = "kmeans"

    def _fit_unsupervised(
        self, x: Any, valid: pa.Table | None, params: dict[str, Any], ctx: RunContext
    ) -> None:
        pre = build_preprocessor(self.features, dense=True)
        x_train = np.asarray(pre.fit_transform(x), dtype=np.float64)
        x_select = x_train
        if valid is not None:
            x_select = np.asarray(pre.transform(sklearn_frame(valid, self.features)), dtype=np.float64)
        model = self._choose_kmeans(x_train, x_select, params, ctx)
        self.pipeline = Pipeline([("preprocess", pre), ("model", model)])


class PcaKMeansAlgorithm(ClusteringAlgorithm):
    name: ClassVar[str] = "pca_kmeans"

    def _fit_unsupervised(
        self, x: Any, valid: pa.Table | None, params: dict[str, Any], ctx: RunContext
    ) -> None:
        variance = float(params.pop("pca_variance", 0.95))
        pre = build_preprocessor(self.features, dense=True)
        pca = PCA(n_components=variance, svd_solver="full", random_state=self.seed)
        x_train = np.asarray(pca.fit_transform(pre.fit_transform(x)), dtype=np.float64)
        x_select = x_train
        if valid is not None:
            x_select = np.asarray(
                pca.transform(pre.transform(sklearn_frame(valid, self.features))), dtype=np.float64
            )
        ctx.log_metric("pca/explained_variance", float(np.sum(pca.explained_variance_ratio_)))
        ctx.log_metric("pca/components", float(pca.n_components_))
        model = self._choose_kmeans(x_train, x_select, params, ctx)
        self.pipeline = Pipeline([("preprocess", pre), ("pca", pca), ("model", model)])


class DbscanAlgorithm(ClusteringAlgorithm):
    name: ClassVar[str] = "dbscan"

    def __init__(self) -> None:
        super().__init__()
        self._core_index: NearestNeighbors | None = None

    def _fit_unsupervised(
        self, x: Any, valid: pa.Table | None, params: dict[str, Any], ctx: RunContext
    ) -> None:
        params.setdefault("min_samples", 5)
        min_samples = int(params["min_samples"])
        pre = build_preprocessor(self.features, dense=True)
        x_train = np.asarray(pre.fit_transform(x), dtype=np.float64)
        if "eps" not in params:
            params["eps"] = self._estimate_eps(x_train, min_samples)
            ctx.log(f"`dbscan` estimated eps={params['eps']:.4g} from the k-distance elbow (k={min_samples})")
        ctx.log_metric("dbscan/eps", float(params["eps"]))
        params.setdefault("n_jobs", default_threads())
        model = DBSCAN(**params).fit(x_train)
        self.pipeline = Pipeline([("preprocess", pre), ("model", model)])
        ctx.progress(1, 1)

    def _estimate_eps(self, x: np.ndarray, min_samples: int) -> float:
        sample = _sample(x, EPS_ROWS, self.seed)
        k = min(min_samples, len(sample))
        # The query rows are part of the index, so the k-th neighbor counts the row itself, like
        # DBSCAN's `min_samples`.
        distances, _ = NearestNeighbors(n_neighbors=k).fit(sample).kneighbors(sample)
        kdist = np.sort(distances[:, -1])
        eps = knee_point(kdist)
        if not np.isfinite(eps) or eps <= 0:
            positive = kdist[kdist > 0]
            eps = float(np.quantile(positive, 0.9)) if len(positive) else 1.0
        return eps

    def _labels(self, space: np.ndarray) -> np.ndarray:
        model: Any = self._model().named_steps["model"]
        core_labels = model.labels_[model.core_sample_indices_]
        if len(core_labels) == 0 or len(space) == 0:
            return np.full(len(space), NOISE, dtype=np.int64)
        if self._core_index is None:
            self._core_index = NearestNeighbors(n_neighbors=1).fit(model.components_)
        distances, index = self._core_index.kneighbors(space)  # pyright: ignore[reportOptionalMemberAccess]
        labels = core_labels[index[:, 0]]
        return np.where(distances[:, 0] <= model.eps, labels, NOISE).astype(np.int64)

    def _to_onnx(self) -> bytes:
        raise NotImplementedError(
            "DBSCAN assigns new rows by nearest core sample, which has no ONNX converter"
        )


# ---- anomaly detection -----------------------------------------------------------------------------------


class IsolationForestAlgorithm(UnsupervisedAlgorithm):
    name: ClassVar[str] = "isolation_forest"
    tasks: ClassVar[frozenset[str]] = ANOMALY
    scale_numeric: ClassVar[bool] = False  # trees are scale-invariant (and float32 ONNX stays closer)

    def _fit_unsupervised(
        self, x: Any, valid: pa.Table | None, params: dict[str, Any], ctx: RunContext
    ) -> None:
        params.setdefault("n_estimators", 200)
        params.setdefault("contamination", "auto")
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", default_threads())
        contamination = params["contamination"]
        if contamination != "auto" and not (
            isinstance(contamination, int | float) and 0 < contamination <= 0.5
        ):
            raise ManifestError('`contamination` must be "auto" or a number in (0, 0.5]')
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(self.features, scale=False, dense=True)),
                ("model", IsolationForest(**params)),
            ]
        )
        pipeline.fit(x)
        self.pipeline = pipeline
        ctx.progress(1, 1)

    def _threshold(self) -> float:
        """`anomaly_score` above which a row is an anomaly (sklearn: decision_function < 0)."""
        return -float(self._model().named_steps["model"].offset_)

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        with limit_threads():
            scores = self._model().score_samples(sklearn_frame(data, self.features))
        return -np.asarray(scores, dtype=np.float64)

    def predict(self, data: pa.Table) -> pa.Table:
        score = self._raw_predict(data)
        labels = np.where(score > self._threshold(), "anomaly", "normal")
        return pa.table(
            {
                "anomaly_score": pa.array(score, type=pa.float64()),
                "prediction": pa.array(labels.tolist(), type=pa.string()),
            }
        )

    def _onnx_outputs(self) -> dict[str, str]:
        return {
            "label": "int64 [n]: -1 anomaly, 1 normal",
            "scores": "float32 [n, 1] decision_function; anomaly_score = -(scores + offset)",
        }

    def _target_opset(self) -> dict[str, int] | None:
        # skl2onnx's IsolationForest converter only handles `ai.onnx.ml` up to version 3.
        return {"": 17, "ai.onnx.ml": 3}

    def _extra_descriptor(self) -> dict[str, Any]:
        return {"threshold": self._threshold()} if self.pipeline is not None else {}

    def score_onnx(self, onnx_path: Path, data: pa.Table) -> np.ndarray:
        """`anomaly_score` from an exported ONNX model."""
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        (scores,) = session.run(["scores"], self.onnx_feed(data))
        offset = float(self._model().named_steps["model"].offset_)
        return -(np.asarray(scores, dtype=np.float64).ravel() + offset)

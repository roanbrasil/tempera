"""Algorithm registry. Names are the `algorithm` field of a run manifest.

Tabular adapters are imported lazily so the core package (and `list`) works without the `tabular` extra.
"""

from __future__ import annotations

import importlib

from tempera_worker.algorithms.dummy import DummyAlgorithm
from tempera_worker.manifest import ManifestError
from tempera_worker.protocol import Algorithm

REGISTRY: dict[str, type[Algorithm]] = {
    DummyAlgorithm.name: DummyAlgorithm,
}

_SKLEARN = "tempera_worker.tabular.sklearn_models"
_GBM = "tempera_worker.tabular.gbm"
_UNSUPERVISED = "tempera_worker.tabular.unsupervised"
LAZY: dict[str, tuple[str, str, str]] = {
    # supervised (Contracts.md, section 7)
    "logistic_regression": (_SKLEARN, "LogisticRegressionAlgorithm", "tabular"),
    "linear_regression": (_SKLEARN, "LinearRegressionAlgorithm", "tabular"),
    "ridge": (_SKLEARN, "RidgeAlgorithm", "tabular"),
    "lasso": (_SKLEARN, "LassoAlgorithm", "tabular"),
    "elastic_net": (_SKLEARN, "ElasticNetAlgorithm", "tabular"),
    "sgd": (_SKLEARN, "SgdAlgorithm", "tabular"),
    "random_forest": (_SKLEARN, "RandomForestAlgorithm", "tabular"),
    "extra_trees": (_SKLEARN, "ExtraTreesAlgorithm", "tabular"),
    "hist_gradient_boosting": (_SKLEARN, "HistGradientBoostingAlgorithm", "tabular"),
    "adaboost": (_SKLEARN, "AdaBoostAlgorithm", "tabular"),
    "knn": (_SKLEARN, "KnnAlgorithm", "tabular"),
    "svm": (_SKLEARN, "SvmAlgorithm", "tabular"),
    "naive_bayes": (_SKLEARN, "NaiveBayesAlgorithm", "tabular"),
    "mlp": (_SKLEARN, "MlpAlgorithm", "tabular"),
    "lightgbm": (_GBM, "LightGbmAlgorithm", "tabular"),
    "xgboost": (_GBM, "XGBoostAlgorithm", "tabular"),
    "catboost": (_GBM, "CatBoostAlgorithm", "tabular"),
    # unsupervised (Contracts.md, section 8)
    "kmeans": (_UNSUPERVISED, "KMeansAlgorithm", "tabular"),
    "dbscan": (_UNSUPERVISED, "DbscanAlgorithm", "tabular"),
    "pca_kmeans": (_UNSUPERVISED, "PcaKMeansAlgorithm", "tabular"),
    "isolation_forest": (_UNSUPERVISED, "IsolationForestAlgorithm", "tabular"),
}


class MissingExtraError(RuntimeError):
    """The algorithm exists but its optional dependencies are not installed."""


def names() -> list[str]:
    return sorted({*REGISTRY, *LAZY})


def get(name: str) -> type[Algorithm]:
    if name in REGISTRY:
        return REGISTRY[name]
    if name not in LAZY:
        raise ManifestError(f"unknown algorithm `{name}` (known: {', '.join(names())})")
    module, attr, extra = LAZY[name]
    try:
        cls: type[Algorithm] = getattr(importlib.import_module(module), attr)
    except ImportError as e:
        raise MissingExtraError(
            f"algorithm `{name}` needs the `{extra}` extra (pip install 'tempera-worker[{extra}]'): {e}"
        ) from e
    REGISTRY[name] = cls
    return cls

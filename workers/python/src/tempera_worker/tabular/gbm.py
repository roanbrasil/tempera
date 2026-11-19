"""Gradient boosting adapters: LightGBM, XGBoost and CatBoost.

* Native categorical support: categorical features are pandas `category` columns whose category list is
  fixed from the training set (stored in `model.json`); unseen values become missing.
* Early stopping on `valid` when present; per-iteration loss is reported as `valid/<loss>` metrics
  (`train/<loss>` when there is no validation set) plus `progress` events.
* Stop signals and long runs write a checkpoint under `<work_dir>/ckpt/`; `resume_from` continues
  boosting from it (LightGBM and XGBoost; CatBoost stops cleanly on a signal but has no checkpoints).
* ONNX input: a single float32 tensor `input` of shape `[n, features]` in manifest order, with
  categorical values as their index in the stored category list and NaN for missing/unknown
  (CatBoost: tensor `features`, numeric features only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping, TrainingCallback

from tempera_worker.manifest import ManifestError
from tempera_worker.tabular.base import TabularAlgorithm, default_threads, pop_int
from tempera_worker.tabular.data import (
    MISSING_CATEGORY,
    encode_target,
    gbm_frame,
    gbm_matrix,
    learn_categories,
    numeric_array,
    string_values,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from tempera_worker.context import RunContext

DEFAULT_ROUNDS = 500
DEFAULT_EARLY_STOPPING = 50
CHECKPOINT_STATE = "state.json"


class GbmAlgorithm(TabularAlgorithm):
    """Common parts: categories, checkpoints, ONNX feed."""

    def __init__(self) -> None:
        super().__init__()
        self.categories: dict[str, list[str]] = {}
        self.best_iteration: int | None = None
        self._resume_model: Path | None = None
        self._resume_iteration = 0

    def _extra_descriptor(self) -> dict[str, Any]:
        return {"categories": self.categories, "best_iteration": self.best_iteration}

    def _load_extra(self, descriptor: dict[str, Any]) -> None:
        self.categories = {k: list(v) for k, v in descriptor.get("categories", {}).items()}
        self.best_iteration = descriptor.get("best_iteration")

    def _frame(self, data: pa.Table) -> pd.DataFrame:
        return gbm_frame(data, self.features, self.categories)

    def onnx_feed(self, data: pa.Table) -> dict[str, np.ndarray]:
        return {"input": gbm_matrix(data, self.features, self.categories)}

    def _onnx_meta(self) -> dict[str, Any]:
        return {
            "inputs": "single float32 tensor `input` [n, features] in `features` order; categorical = index "
            "in `categories[feature]`, NaN = missing or unseen",
            "outputs": {"label": "class index", "probabilities": "float32 [n, classes]"}
            if self.is_classification
            else {"variable": "float32 [n, 1]"},
        }

    def _split_params(self) -> tuple[dict[str, Any], int, int]:
        params = dict(self.params)
        rounds = pop_int(
            params, ("num_boost_round", "n_estimators", "num_iterations", "num_rounds"), DEFAULT_ROUNDS
        )
        early = pop_int(params, ("early_stopping_rounds", "early_stopping_round"), DEFAULT_EARLY_STOPPING)
        return params, rounds, early

    # ---- checkpoints -----------------------------------------------------------------------------------

    def _write_checkpoint(self, path: Path, iteration: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._save_booster(path / self.native_file)
        (path / CHECKPOINT_STATE).write_text(json.dumps({"algorithm": self.name, "iteration": iteration}))

    def _maybe_checkpoint(self, iteration: int) -> bool:
        """Checkpoint when due; return True when training must stop."""
        ctx = self._ctx
        if ctx is None or not ctx.checkpoint_due():
            return False
        ckpt = ctx.work_dir / "ckpt"
        self._write_checkpoint(ckpt, iteration)
        ctx.saved_checkpoint(ckpt, iteration)
        return ctx.should_stop()

    def save_checkpoint(self, path: Path) -> None:
        self._write_checkpoint(path, self._resume_iteration)

    def resume_from(self, path: Path) -> None:
        state_file = path / CHECKPOINT_STATE
        if not state_file.exists():
            raise ManifestError(f"{path} is not a `{self.name}` checkpoint (no {CHECKPOINT_STATE})")
        state = json.loads(state_file.read_text())
        if state.get("algorithm") != self.name:
            raise ManifestError(f"{path} is a `{state.get('algorithm')}` checkpoint, not `{self.name}`")
        self._resume_model = path / self.native_file
        self._resume_iteration = int(state["iteration"])

    def _save_booster(self, path: Path) -> None:
        raise NotImplementedError

    def _save_native(self, path: Path) -> None:
        self._save_booster(path)

    def _prepare(self, train: pa.Table) -> None:
        self.categories = learn_categories(train, self.features)
        y = encode_target(train, self.task, self.classes)
        if self.is_classification and len(np.unique(y)) < 2:
            raise ValueError("the training set has a single class")


class _LightGbmCallback:
    """Reports metrics/progress, checkpoints, and stops on signal."""

    order = 20  # before lightgbm's early stopping (30), so the last iteration is still reported
    before_iteration = False

    def __init__(self, algo: LightGbmAlgorithm, ctx: RunContext) -> None:
        self.algo, self.ctx = algo, ctx

    def __call__(self, env: Any) -> None:
        step = env.iteration + 1
        for data_name, eval_name, value, _higher_better, *_ in env.evaluation_result_list:
            self.ctx.log_metric(f"{data_name}/{eval_name}", float(value), step=step)
        self.ctx.progress(step, env.end_iteration)
        self.algo.booster = env.model
        if self.algo._maybe_checkpoint(step):
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)


class LightGbmAlgorithm(GbmAlgorithm):
    name: ClassVar[str] = "lightgbm"
    native_file: ClassVar[str] = "model.txt"

    def __init__(self) -> None:
        super().__init__()
        self.booster: lgb.Booster | None = None

    def _lgb_params(self, params: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "seed": self.seed,
            "deterministic": True,
            "num_threads": default_threads(),
        }
        if self.task == "binary_classification":
            base |= {"objective": "binary", "metric": "binary_logloss"}
        elif self.task == "multiclass_classification":
            base |= {"objective": "multiclass", "num_class": len(self.classes), "metric": "multi_logloss"}
        else:
            base |= {"objective": "regression", "metric": "l2"}
        return base | params

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        self._prepare(train)
        params, rounds, early = self._split_params()
        params = self._lgb_params(params)
        train_ds = lgb.Dataset(
            self._frame(train), label=encode_target(train, self.task, self.classes), free_raw_data=False
        )
        callbacks: list[Any] = [_LightGbmCallback(self, ctx)]
        if valid is not None:
            valid_ds = lgb.Dataset(
                self._frame(valid), label=encode_target(valid, self.task, self.classes), reference=train_ds
            )
            sets, names = [valid_ds], ["valid"]
            if early > 0:
                callbacks.append(lgb.early_stopping(early, first_metric_only=True, verbose=False))
        else:
            sets, names = [train_ds], ["train"]
        init_model = str(self._resume_model) if self._resume_model else None
        remaining = max(rounds - self._resume_iteration, 0)
        if init_model:
            ctx.log(f"resuming LightGBM at iteration {self._resume_iteration}; {remaining} rounds left")
        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=remaining,
            valid_sets=sets,
            valid_names=names,
            callbacks=callbacks,
            init_model=init_model,
        )
        best = booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()
        self.best_iteration = int(best)
        # Keep only the trees up to the best iteration so native, reloaded and ONNX models agree.
        self.booster = lgb.Booster(model_str=booster.model_to_string(num_iteration=best))

    def _model(self) -> lgb.Booster:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return self.booster

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        out = np.asarray(self._model().predict(self._frame(data)), dtype=np.float64)
        if self.task == "binary_classification":
            return np.column_stack([1.0 - out, out])
        return out

    def _save_booster(self, path: Path) -> None:
        self._model().save_model(str(path))

    def _load_native(self, path: Path) -> None:
        self.booster = lgb.Booster(model_file=str(path))

    def feature_importance(self) -> dict[str, float]:
        gains = self._model().feature_importance(importance_type="gain")
        return {f.name: float(g) for f, g in zip(self.features, gains, strict=True)}

    def _to_onnx(self) -> bytes:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType

        onx = convert_lightgbm(
            self._model(),
            initial_types=[("input", FloatTensorType([None, len(self.features)]))],
            zipmap=False,
        )
        self.onnx_meta = self._onnx_meta()
        return onx.SerializeToString()


class _XGBoostCallback(TrainingCallback):
    def __init__(self, algo: XGBoostAlgorithm, ctx: RunContext, offset: int, total: int) -> None:
        super().__init__()
        self.algo, self.ctx, self.offset, self.total = algo, ctx, offset, total

    def after_iteration(self, model: Any, epoch: int, evals_log: dict[str, Any]) -> bool:
        step = self.offset + epoch + 1
        for data_name, metrics in evals_log.items():
            for metric, values in metrics.items():
                last = values[-1]
                value = last[0] if isinstance(last, tuple) else last
                self.ctx.log_metric(f"{data_name}/{metric}", float(value), step=step)
        self.ctx.progress(step, self.total)
        self.algo.booster = model
        return self.algo._maybe_checkpoint(step)


class XGBoostAlgorithm(GbmAlgorithm):
    name: ClassVar[str] = "xgboost"
    native_file: ClassVar[str] = "model.ubj"

    def __init__(self) -> None:
        super().__init__()
        self.booster: xgb.Booster | None = None

    def _xgb_params(self, params: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "eta": 0.05,
            "max_depth": 6,
            "tree_method": "hist",
            "seed": self.seed,
            "nthread": default_threads(),
            "verbosity": 0,
        }
        if self.task == "binary_classification":
            base |= {"objective": "binary:logistic", "eval_metric": "logloss"}
        elif self.task == "multiclass_classification":
            base |= {"objective": "multi:softprob", "num_class": len(self.classes), "eval_metric": "mlogloss"}
        else:
            base |= {"objective": "reg:squarederror", "eval_metric": "rmse"}
        return base | params

    def _dmatrix(self, data: pa.Table, with_label: bool) -> xgb.DMatrix:
        label = encode_target(data, self.task, self.classes) if with_label else None
        return xgb.DMatrix(self._frame(data), label=label, enable_categorical=True)

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        self._prepare(train)
        params, rounds, early = self._split_params()
        params = self._xgb_params(params)
        dtrain = self._dmatrix(train, with_label=True)
        callbacks: list[Any] = [_XGBoostCallback(self, ctx, self._resume_iteration, rounds)]
        if valid is not None:
            evals = [(self._dmatrix(valid, with_label=True), "valid")]
            if early > 0:
                callbacks.append(EarlyStopping(rounds=early, data_name="valid", save_best=True))
        else:
            evals = [(dtrain, "train")]
        init_model = str(self._resume_model) if self._resume_model else None
        remaining = max(rounds - self._resume_iteration, 0)
        if init_model:
            ctx.log(f"resuming XGBoost at iteration {self._resume_iteration}; {remaining} rounds left")
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=remaining,
            evals=evals,
            callbacks=callbacks,
            xgb_model=init_model,
            verbose_eval=False,
        )
        self.booster = booster
        self.best_iteration = int(booster.num_boosted_rounds())

    def _model(self) -> xgb.Booster:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        return self.booster

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        out = np.asarray(self._model().predict(self._dmatrix(data, with_label=False)), dtype=np.float64)
        if self.task == "binary_classification":
            return np.column_stack([1.0 - out, out])
        return out

    def _save_booster(self, path: Path) -> None:
        self._model().save_model(str(path))

    def _load_native(self, path: Path) -> None:
        booster = xgb.Booster()
        booster.load_model(str(path))
        self.booster = booster

    def feature_importance(self) -> dict[str, float]:
        scores = self._model().get_score(importance_type="gain")
        out: dict[str, float] = {}
        for f in self.features:
            score = scores.get(f.name, 0.0)
            out[f.name] = float(sum(score)) if isinstance(score, list) else float(score)
        return out

    def _to_onnx(self) -> bytes:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType

        if self.categories:
            raise NotImplementedError("onnxmltools cannot convert XGBoost categorical splits")
        # onnxmltools only understands the default `f<index>` feature names.
        booster = self._model().copy()
        booster.feature_names = None
        booster.feature_types = None
        onx = convert_xgboost(booster, initial_types=[("input", FloatTensorType([None, len(self.features)]))])
        self.onnx_meta = self._onnx_meta()
        return onx.SerializeToString()


class _CatBoostCallback:
    """Reports per-iteration metrics/progress and stops on signal (CatBoost `callbacks` protocol)."""

    def __init__(self, ctx: RunContext, total: int) -> None:
        self.ctx, self.total = ctx, total

    def after_iteration(self, info: Any) -> bool:
        step = int(info.iteration)  # CatBoost counts iterations from 1 here
        metrics: dict[str, dict[str, list[float]]] = info.metrics
        data_name = "validation" if "validation" in metrics else "learn"
        prefix = "valid" if data_name == "validation" else "train"
        for metric, values in metrics.get(data_name, {}).items():
            if values:
                self.ctx.log_metric(f"{prefix}/{metric}", float(values[-1]), step=step)
        self.ctx.progress(step, self.total)
        return not self.ctx.should_stop()


class CatBoostAlgorithm(GbmAlgorithm):
    """CatBoost with native categorical features (missing and unseen values share one category)."""

    name: ClassVar[str] = "catboost"
    native_file: ClassVar[str] = "model.cbm"

    def __init__(self) -> None:
        super().__init__()
        self.model: cb.CatBoost | None = None

    def _cb_frame(self, data: pa.Table) -> pd.DataFrame:
        cols: dict[str, Any] = {}
        for f in self.features:
            if f.kind == "categorical":
                known = set(self.categories[f.name])
                values = [
                    v if v is not None and v in known else MISSING_CATEGORY for v in string_values(data, f)
                ]
                cols[f.name] = pd.Series(values, dtype=object)
            else:
                cols[f.name] = numeric_array(data, f)
        return pd.DataFrame(cols, columns=[f.name for f in self.features])

    def _cat_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "categorical"]

    def _fit(self, train: pa.Table, valid: pa.Table | None, ctx: RunContext) -> None:
        if self._resume_model is not None:
            raise ManifestError("`catboost` does not support resuming from a checkpoint")
        self._prepare(train)
        params, rounds, early = self._split_params()
        if self.task == "binary_classification":
            loss = "Logloss"
        elif self.task == "multiclass_classification":
            loss = "MultiClass"
        else:
            loss = "RMSE"
        base: dict[str, Any] = {
            "iterations": rounds,
            "loss_function": loss,
            "random_seed": self.seed,
            "thread_count": default_threads(),
            "allow_writing_files": False,
            "verbose": False,
        }
        params = base | params
        cls = cb.CatBoostClassifier if self.is_classification else cb.CatBoostRegressor
        model = cls(**params)
        y = encode_target(train, self.task, self.classes)
        fit_kwargs: dict[str, Any] = {}
        if valid is not None:
            fit_kwargs["eval_set"] = cb.Pool(
                self._cb_frame(valid),
                label=encode_target(valid, self.task, self.classes),
                cat_features=self._cat_features(),
            )
            fit_kwargs["use_best_model"] = True
            if early > 0:
                fit_kwargs["early_stopping_rounds"] = early
        model.fit(
            cb.Pool(self._cb_frame(train), label=y, cat_features=self._cat_features()),
            callbacks=[_CatBoostCallback(ctx, int(params["iterations"]))],
            **fit_kwargs,
        )
        self.model = model
        self.best_iteration = int(model.tree_count_ or 0)

    def resume_from(self, path: Path) -> None:
        raise ManifestError("`catboost` does not support resuming from a checkpoint")

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError("`catboost` does not support checkpoints")

    def _model(self) -> cb.CatBoost:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model

    def _scatter(self, proba: np.ndarray, seen: list[int]) -> np.ndarray:
        """The model only knows the classes seen in training; scatter into the full class list."""
        out = np.zeros((proba.shape[0], len(self.classes)), dtype=np.float64)
        out[:, seen] = proba
        return out

    def _seen_classes(self) -> list[int]:
        classes: Any = self._model().classes_
        return [int(c) for c in classes]

    def _raw_predict(self, data: pa.Table) -> np.ndarray:
        model = self._model()
        frame = self._cb_frame(data)
        if not self.is_classification:
            return np.asarray(model.predict(frame), dtype=np.float64)
        proba = np.asarray(model.predict(frame, prediction_type="Probability"), dtype=np.float64)
        return self._scatter(proba, self._seen_classes())

    def _save_booster(self, path: Path) -> None:
        self._model().save_model(str(path), format="cbm")

    def _load_native(self, path: Path) -> None:
        model = cb.CatBoostClassifier() if self.is_classification else cb.CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        self.model = model

    def feature_importance(self) -> dict[str, float]:
        values: Any = self._model().get_feature_importance()
        return {f.name: float(v) for f, v in zip(self.features, values, strict=True)}

    def _to_onnx(self) -> bytes:
        import tempfile

        if self._cat_features():
            raise NotImplementedError("CatBoost cannot export categorical features to ONNX")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "model.onnx"
            self._model().save_model(str(out), format="onnx")
            data = out.read_bytes()
        self.onnx_meta = {
            "inputs": "single float32 tensor `features` [n, features] in `features` order, NaN = missing",
            "outputs": {"label": "class index", "probabilities": "sequence of {class index: probability}"}
            if self.is_classification
            else {"predictions": "float32 [n]"},
        }
        return data

    def onnx_feed(self, data: pa.Table) -> dict[str, np.ndarray]:
        cols = [numeric_array(data, f) for f in self.features]
        return {"features": np.column_stack(cols).astype(np.float32)}

    def score_onnx(self, onnx_path: Path, data: pa.Table) -> np.ndarray:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feed = self.onnx_feed(data)
        if not self.is_classification:
            (values,) = session.run(["predictions"], feed)
            return np.asarray(values, dtype=np.float64).ravel()
        maps: Any = session.run(["probabilities"], feed)[0]
        seen = self._seen_classes()
        proba = np.asarray([[float(m[c]) for c in seen] for m in maps], dtype=np.float64)
        return self._scatter(proba.reshape(len(maps), len(seen)), seen)

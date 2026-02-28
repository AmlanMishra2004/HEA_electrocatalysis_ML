import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def make_splits(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val : n_train + n_val + n_test]
    return idx_train, idx_val, idx_test


def build_split_data(X: np.ndarray, y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray) -> SplitData:
    return SplitData(
        X_train=X[idx_train],
        y_train=y[idx_train],
        X_val=X[idx_val],
        y_val=y[idx_val],
        X_test=X[idx_test],
        y_test=y[idx_test],
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


class RidgeNumpy:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.mean_: np.ndarray = np.array([])
        self.std_: np.ndarray = np.array([])
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeNumpy":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        Xs = (X - self.mean_) / self.std_
        y_center = y - y.mean()
        XtX = Xs.T @ Xs
        reg = self.alpha * np.eye(Xs.shape[1])
        self.coef_ = np.linalg.solve(XtX + reg, Xs.T @ y_center)
        self.intercept_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.coef_ + self.intercept_


def train_model(
    split: SplitData,
    feature_names: List[str],
    artifacts_dir: str,
) -> Dict[str, object]:
    model_type = "ridge_numpy"
    model_obj = None
    importance = np.zeros(len(feature_names), dtype=float)

    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_val, split.y_val)],
            verbose=False,
        )
        model_type = "xgboost"
        model_obj = model

        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        for i in range(len(feature_names)):
            importance[i] = float(gain.get(f"f{i}", 0.0))

        model_path = os.path.join(artifacts_dir, "model_xgb.json")
        model.save_model(model_path)

    except Exception:
        model = RidgeNumpy(alpha=1.0).fit(split.X_train, split.y_train)
        model_obj = model
        importance = np.abs(model.coef_)
        model_path = os.path.join(artifacts_dir, "model_ridge.npz")
        np.savez(
            model_path,
            mean=model.mean_,
            std=model.std_,
            coef=model.coef_,
            intercept=model.intercept_,
            alpha=model.alpha,
        )

    yhat_train = model_obj.predict(split.X_train)
    yhat_val = model_obj.predict(split.X_val)
    yhat_test = model_obj.predict(split.X_test)

    metrics = {
        "model_type": model_type,
        "train": eval_metrics(split.y_train, yhat_train),
        "val": eval_metrics(split.y_val, yhat_val),
        "test": eval_metrics(split.y_test, yhat_test),
    }

    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    importance_order = np.argsort(importance)[::-1]
    importance_rows = [
        {"feature": feature_names[i], "importance": float(importance[i])}
        for i in importance_order
    ]

    return {
        "model": model_obj,
        "model_type": model_type,
        "model_path": model_path,
        "metrics": metrics,
        "importance_rows": importance_rows,
        "pred_train": yhat_train,
        "pred_val": yhat_val,
        "pred_test": yhat_test,
    }

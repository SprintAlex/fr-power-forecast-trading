"""LightGBM quantile regression -> probabilistic day-ahead forecast P10/P50/P90.

One LightGBM per quantile (objective='quantile', alpha=q). Single model across
all 24 hours (hour is a feature) — gradient boosting handles the hour x driver
interactions that a per-hour linear model can't.

Quantile crossing (P10 > P50 etc., which boosting can produce) is repaired by
sorting the three predictions row-wise: cheap and standard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import config

DEFAULT_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)


class QuantileLGBM:
    def __init__(self, quantiles: list[float] | None = None, params: dict | None = None):
        self.quantiles = quantiles or config.QUANTILES
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.models: dict[float, LGBMRegressor] = {}
        self.feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_cols = list(X.columns)
        for q in self.quantiles:
            m = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols is None:
            raise RuntimeError("QuantileLGBM not fitted.")
        X = X[self.feature_cols]
        preds = {q: self.models[q].predict(X) for q in self.quantiles}
        arr = np.sort(np.column_stack([preds[q] for q in self.quantiles]), axis=1)  # de-cross
        cols = [f"p{int(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(arr, columns=cols, index=X.index)

    def feature_importance(self) -> pd.Series:
        """Gain importance from the median (P50) model."""
        q50 = min(self.quantiles, key=lambda q: abs(q - 0.5))
        m = self.models[q50]
        return pd.Series(m.feature_importances_, index=self.feature_cols).sort_values(ascending=False)

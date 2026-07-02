"""LEAR — LASSO-Estimated AutoRegressive model (Lago et al. 2021), in-house.

The EPF reference linear model. Implemented with scikit-learn (no epftoolbox
dependency -> avoids its pinned TensorFlow, which breaks on Python 3.13).

Faithful to the LEAR recipe:
  - one independent LASSO per hour of the day (24 models),
  - asinh variance-stabilising transform on the target (handles negative prices
    and spikes far better than log; standard in modern EPF),
  - standardised features, LASSO penalty chosen by BIC (LassoLarsIC).

Point forecast only (the EPF benchmark is a point model). Probabilistic P10/P50/
P90 come from the LightGBM quantile model; LEAR is the credibility baseline P50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LEAR:
    def __init__(self, criterion: str = "bic"):
        self.criterion = criterion
        self.models: dict[int, object] = {}
        self.feature_cols: list[str] | None = None

    def fit(self, feat: pd.DataFrame, feature_cols: list[str]):
        self.feature_cols = feature_cols
        for h in range(24):
            sub = feat[feat["hour"] == h]
            if len(sub) == 0:
                continue
            X = sub[feature_cols].to_numpy()
            y = np.arcsinh(sub["target"].to_numpy())      # variance-stabilise
            model = make_pipeline(
                StandardScaler(),
                LassoLarsIC(criterion=self.criterion, max_iter=2000),
            )
            model.fit(X, y)
            self.models[h] = model
        return self

    def predict(self, feat: pd.DataFrame) -> pd.Series:
        if self.feature_cols is None:
            raise RuntimeError("LEAR not fitted.")
        pred = pd.Series(index=feat.index, dtype=float, name="lear")
        for h, model in self.models.items():
            mask = feat["hour"] == h
            if mask.sum() == 0:
                continue
            X = feat.loc[mask, self.feature_cols].to_numpy()
            pred.loc[mask] = np.sinh(model.predict(X))    # invert transform
        return pred

"""Cheap point baselines. The bar every fancy model must clear.

naive D-7 is the standard EPF reference (Lago et al.): same hour, one week ago —
captures the weekly + daily seasonality for free. If LightGBM can't beat it, the
features add nothing.
"""
from __future__ import annotations

import pandas as pd


def naive_d7(feat: pd.DataFrame) -> pd.Series:
    """Same hour, 7 days ago (price_lag_168)."""
    return feat["price_lag_168"].rename("naive_d7")


def naive_d1(feat: pd.DataFrame) -> pd.Series:
    """Same hour, 1 day ago (price_lag_24)."""
    return feat["price_lag_24"].rename("naive_d1")

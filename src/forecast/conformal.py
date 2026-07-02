"""Conformalized Quantile Regression (Romano et al. 2019) — coverage repair.

LightGBM quantiles under-cover (our P10-P90 caught ~68% vs the nominal 80%). CQR
fixes this with a finite-sample guarantee: on a held-out calibration set, measure
how far reality fell outside the predicted band, then widen the band by that
amount. Distribution-free, model-agnostic, one number.

    E_i = max(lo_i - y_i, y_i - hi_i)          # signed miss on calibration set
    Q   = ceil((m+1)(1-alpha))/m  quantile of E
    band -> [lo - Q, hi + Q]                    # guarantees >= (1-alpha) coverage
"""
from __future__ import annotations

import numpy as np


def conformal_delta(y_cal, lo_cal, hi_cal, alpha: float = 0.2) -> float:
    """Width adjustment Q for a (1-alpha) interval, from calibration residuals."""
    y = np.asarray(y_cal, float)
    lo = np.asarray(lo_cal, float)
    hi = np.asarray(hi_cal, float)
    E = np.maximum(lo - y, y - hi)
    m = len(E)
    level = min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)   # finite-sample correction
    return float(np.quantile(E, level, method="higher"))


def apply_delta(lo, hi, q: float):
    return np.asarray(lo, float) - q, np.asarray(hi, float) + q

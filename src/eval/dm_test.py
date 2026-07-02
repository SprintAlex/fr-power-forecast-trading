"""Diebold-Mariano test — is one forecast significantly better, or just luck?

EPF standard (Lago et al. use it to claim model superiority). Tests whether the
loss differential between two forecasts has zero mean, with a HAC (Newey-West)
variance so serial correlation in daily errors doesn't fake significance.

  d_t = L(e1_t) - L(e2_t)      (loss diff; L = squared or absolute error)
  DM  = mean(d) / sqrt(HAC_var(d) / n)   ~ N(0,1)
  DM < 0 and p small  =>  forecast 1 is significantly better than forecast 2.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _loss(err, kind):
    return err ** 2 if kind == "se" else np.abs(err)


def dm_test(y, p1, p2, kind: str = "ae", h: int = 1):
    """Return (DM stat, two-sided p-value). Negative stat => p1 better than p2."""
    y, p1, p2 = np.asarray(y, float), np.asarray(p1, float), np.asarray(p2, float)
    d = _loss(y - p1, kind) - _loss(y - p2, kind)
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance with (h-1) lags.
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, h):
        cov = np.cov(d[k:], d[:-k])[0, 1]
        var += 2 * (1 - k / h) * cov
    dm = dbar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def report(pred):
    """Compare LightGBM P50 against naive D-7 and LEAR on the saved predictions."""
    y = pred["actual"]
    pairs = [("LightGBM P50", "p50", "naive D-7", "naive_d7"),
             ("LightGBM P50", "p50", "LEAR", "lear")]
    print("Diebold-Mariano (MAE loss). Negative DM => first model better; p<0.05 significant.\n")
    for n1, c1, n2, c2 in pairs:
        dm, p = dm_test(y, pred[c1], pred[c2], kind="ae")
        verdict = f"{n1} better (p={p:.1e})" if dm < 0 and p < 0.05 else \
                  (f"{n2} better (p={p:.1e})" if dm > 0 and p < 0.05 else "no sig. difference")
        print(f"  {n1:14} vs {n2:10}: DM={dm:+6.2f}  ->  {verdict}")


if __name__ == "__main__":
    import pandas as pd

    from src.forecast.train import PRED_PATH

    report(pd.read_parquet(PRED_PATH))

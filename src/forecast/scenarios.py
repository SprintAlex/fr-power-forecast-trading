"""Correlated intraday price scenarios from the quantile forecast.

Marginal quantiles (P10/P50/P90) ignore how hours move TOGETHER, but a battery
cares about the whole price *path* (charge hour 3, discharge hour 19 -> their
joint move is the risk). We turn the marginals into scenario paths: Gaussian with
per-hour std implied by the P10-P90 width and an AR(1) cross-hour correlation, so
sampled days look like plausible price curves, not 24 independent draws.

These feed the CVaR battery optimiser (battery_lp.optimize_day_cvar).
"""
from __future__ import annotations

import numpy as np

_Z90 = 1.2815515594  # norm ppf(0.9): P90-P50 ~ Z90 * sigma


def gen_scenarios(p10, p50, p90, n: int = 200, rho: float = 0.8, seed: int = 0) -> np.ndarray:
    """Return array (n, H) of price paths consistent with the quantile forecast."""
    p10, p50, p90 = np.asarray(p10, float), np.asarray(p50, float), np.asarray(p90, float)
    H = len(p50)
    sigma = np.clip((p90 - p10) / (2 * _Z90), 1e-6, None)
    idx = np.arange(H)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])      # AR(1) cross-hour
    L = np.linalg.cholesky(corr + 1e-9 * np.eye(H))
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n, H)) @ L.T
    paths = p50[None, :] + eps * sigma[None, :]
    return np.clip(paths, -500.0, 4000.0)

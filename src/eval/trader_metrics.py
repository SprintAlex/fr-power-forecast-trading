"""Trader risk metrics on a daily P&L series — what a desk actually judges.

A worse-MAE forecast that captures more perfect-foresight P&L with a higher
Sharpe is the better model. These metrics, not RMSE, are the project's verdict.
Power trades 365 days/yr -> annualisation factor sqrt(365).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(365)


def sharpe(pnl: pd.Series) -> float:
    """Annualised Sharpe of daily P&L (risk-free = 0)."""
    s = pnl.std(ddof=1)
    return float(pnl.mean() / s * ANN) if s > 0 else float("nan")


def var(pnl: pd.Series, level: float = 0.95) -> float:
    """Daily Value-at-Risk: the loss not exceeded with `level` probability
    (returned as a positive number = magnitude of loss)."""
    return float(-pnl.quantile(1 - level))


def cvar(pnl: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean loss in the worst (1-level) tail."""
    thr = pnl.quantile(1 - level)
    tail = pnl[pnl <= thr]
    return float(-tail.mean()) if len(tail) else float("nan")


def max_drawdown(pnl: pd.Series) -> float:
    """Max peak-to-trough drop of cumulative P&L (positive magnitude)."""
    cum = pnl.cumsum()
    return float((cum.cummax() - cum).max())


def report(pnl: pd.Series, capacity_mwh: float) -> dict:
    return {
        "total_eur": float(pnl.sum()),
        "eur_per_mwh_cap": float(pnl.sum() / capacity_mwh),
        "mean_daily": float(pnl.mean()),
        "sharpe_ann": sharpe(pnl),
        "VaR95_daily": var(pnl, 0.95),
        "CVaR95_daily": cvar(pnl, 0.95),
        "max_drawdown": max_drawdown(pnl),
        "pct_profitable_days": float((pnl > 0).mean() * 100),
        "worst_day": float(pnl.min()),
        "best_day": float(pnl.max()),
    }

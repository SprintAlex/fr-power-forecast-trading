"""Forecast metrics: point (MAE/RMSE/R2/DirAcc) + probabilistic (pinball/coverage).

Trader-facing P&L / risk metrics live in src/eval/trader_metrics.py (battery layer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def dir_acc(y, p, ref):
    """Directional accuracy vs a reference (e.g. price 24h earlier): does the
    forecast get the sign of the change right? `ref` = baseline level."""
    y, p, ref = np.asarray(y), np.asarray(p), np.asarray(ref)
    return float(np.mean(np.sign(y - ref) == np.sign(p - ref)))


def pinball_loss(y, q_pred, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level q in (0,1)."""
    y, q_pred = np.asarray(y), np.asarray(q_pred)
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def mean_pinball(y, quantile_preds: dict[float, np.ndarray]) -> float:
    """Average pinball across provided quantile levels (proxy for CRPS)."""
    return float(np.mean([pinball_loss(y, p, q) for q, p in quantile_preds.items()]))


def coverage(y, lo, hi) -> float:
    """Empirical coverage of a prediction interval [lo, hi]."""
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def spike_recall(y, p, thr: float) -> float:
    """Recall on price spikes (y > thr): of true spikes, how many did the
    forecast also flag as > thr."""
    y, p = np.asarray(y), np.asarray(p)
    true = y > thr
    if true.sum() == 0:
        return float("nan")
    return float(((p > thr) & true).sum() / true.sum())


def negative_recall(y, p) -> float:
    """Recall on negative prices (y < 0)."""
    y, p = np.asarray(y), np.asarray(p)
    true = y < 0
    if true.sum() == 0:
        return float("nan")
    return float(((p < 0) & true).sum() / true.sum())


def point_report(y, p, ref=None) -> dict:
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p)}
    if ref is not None:
        out["DirAcc"] = dir_acc(y, p, ref)
    return out

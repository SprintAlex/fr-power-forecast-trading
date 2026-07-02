"""Generate the README result figures from the real ENTSO-E pipeline outputs.

Writes PNGs to docs/img/ (committed, so they render on GitHub).
Run after a real forecast: `python -m src.forecast.train --real` then this.

    python scripts/make_figures.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.data.entsoe_pull import raw_path
from src.forecast.train import PRED_PATH
from src.trading.battery_lp import backtest

IMG = config.ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

pred = pd.read_parquet(PRED_PATH)
raw = pd.read_parquet(raw_path("FR"))
if raw.index.tz is None:
    raw.index = raw.index.tz_localize("UTC")


def fig_price_vs_gas():
    fig, ax = plt.subplots(figsize=(10, 4))
    p = raw["price_eur_mwh"].resample("1W").mean()
    ax.plot(p.index, p.values, label="FR power (weekly mean)", color="tab:blue")
    ax.set_ylabel("EUR/MWh power"); ax.set_xlabel("")
    ax2 = ax.twinx()
    g = raw["gas_ttf_eur_mwh"].resample("1W").mean()
    ax2.plot(g.index, g.values, color="tab:red", alpha=0.7, label="gas TTF")
    ax2.set_ylabel("EUR/MWh gas", color="tab:red")
    ax.set_title("Real FR day-ahead power vs gas TTF — the 2022 crisis drives the level")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(IMG / "price_vs_gas.png"); plt.close(fig)


def fig_forecast_fan():
    sl = pred.iloc[:14 * 24]
    lo = sl["p10_cal"] if "p10_cal" in sl else sl["p10"]
    hi = sl["p90_cal"] if "p90_cal" in sl else sl["p90"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(sl.index, lo, hi, alpha=0.25, label="P10-P90 (conformal)", color="tab:blue")
    ax.plot(sl.index, sl["p50"], lw=1.3, label="P50 forecast", color="tab:blue")
    ax.plot(sl.index, sl["actual"], color="k", lw=0.9, label="actual")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Probabilistic day-ahead forecast — first 2 test weeks (real FR 2024)")
    ax.legend(); fig.tight_layout(); fig.savefig(IMG / "forecast_fan.png"); plt.close(fig)


def fig_battery_pnl():
    fc = backtest(pred, "p50")["pnl_realised"].cumsum()
    pf = backtest(pred, "actual")["pnl_realised"].cumsum()
    cap = fc.iloc[-1] / pf.iloc[-1] * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fc.index, fc.values, label=f"forecast strategy ({cap:.0f}% of perfect)")
    ax.plot(pf.index, pf.values, ls="--", label="perfect foresight")
    ax.set_ylabel("cumulative EUR")
    ax.set_title("Battery arbitrage P&L — real FR 2024 (1 MW / 2 MWh)")
    ax.legend(); fig.tight_layout(); fig.savefig(IMG / "battery_pnl.png"); plt.close(fig)


def fig_regime():
    from src.eval import regime_test

    tab = regime_test.run(synthetic=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["R2"], 0.4, label="forecast R²", color="tab:blue")
    ax.bar(x + 0.2, tab["%_perfect"] / 100, 0.4, label="% perfect captured", color="tab:green")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{y}\ngas {g:.0f}€"
                    for y, g in zip(tab.index, tab["gas_mean"])])
    ax.set_title("Crisis stress test — model breaks in 2022 (R²<0), recovers after")
    ax.legend(); fig.tight_layout(); fig.savefig(IMG / "regime_crisis.png"); plt.close(fig)


def fig_spread():
    from src.trading.spread_strategy import forecast_zone

    fr = forecast_zone("FR", synthetic=False)
    de = forecast_zone("DE", synthetic=False)
    real = fr["actual"] - de["actual"]
    fc = fr["p50"] - de["p50"]
    naive = fr["lag168"] - de["lag168"]
    accs = {
        "naive (D-7 sign)": (np.sign(naive) == np.sign(real)).mean(),
        "model (P50 sign)": (np.sign(fc) == np.sign(real)).mean(),
        "coin flip": 0.5,
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:gray", "tab:green", "lightgray"]
    ax.bar(list(accs), [v * 100 for v in accs.values()], color=colors)
    for i, v in enumerate(accs.values()):
        ax.text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center")
    ax.set_ylabel("spread-sign accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("FR-DE spread — the sign is what pays (real 2024)")
    fig.tight_layout(); fig.savefig(IMG / "spread_signacc.png"); plt.close(fig)


if __name__ == "__main__":
    fig_price_vs_gas(); print("price_vs_gas.png")
    fig_forecast_fan(); print("forecast_fan.png")
    fig_battery_pnl(); print("battery_pnl.png")
    fig_spread(); print("spread_signacc.png")
    fig_regime(); print("regime_crisis.png")
    print(f"\nAll figures -> {IMG}")

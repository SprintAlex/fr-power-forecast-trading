"""Capstone: forecast + battery backtest -> trader metrics + figures.

Run:  python -m src.eval.dashboard          # uses saved predictions
Produces results/*.png and prints the final trader scorecard.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.eval import trader_metrics as TM
from src.forecast.train import PRED_PATH
from src.trading.battery_lp import backtest


def _fan_chart(pred: pd.DataFrame, ax, days: int = 10):
    sl = pred.iloc[: days * 24]
    lo = sl["p10_cal"] if "p10_cal" in sl else sl["p10"]
    hi = sl["p90_cal"] if "p90_cal" in sl else sl["p90"]
    ax.fill_between(sl.index, lo, hi, alpha=0.25, label="P10-P90 (conformal)")
    ax.plot(sl.index, sl["p50"], lw=1.2, label="P50 forecast")
    ax.plot(sl.index, sl["actual"], lw=1.0, color="k", label="actual")
    ax.set_title(f"Probabilistic forecast — first {days} test days")
    ax.set_ylabel("EUR/MWh")
    ax.legend(fontsize=8)


def _pnl_curve(fc, pf, ax):
    ax.plot(fc.index, fc["pnl_realised"].cumsum(), label="forecast strategy")
    ax.plot(pf.index, pf["pnl_realised"].cumsum(), ls="--", label="perfect foresight")
    ax.set_title("Cumulative battery P&L")
    ax.set_ylabel("EUR")
    ax.legend(fontsize=8)


def _pnl_hist(fc, ax):
    ax.hist(fc["pnl_realised"], bins=40)
    v = TM.var(fc["pnl_realised"])
    ax.axvline(-v, color="r", ls="--", label=f"VaR95 = {v:,.0f}")
    ax.set_title("Daily P&L distribution")
    ax.set_xlabel("EUR/day")
    ax.legend(fontsize=8)


def run():
    pred = pd.read_parquet(PRED_PATH)
    batt = config.BATTERY
    fc = backtest(pred, decision_col="p50")
    pf = backtest(pred, decision_col="actual")
    capture = fc["pnl_realised"].sum() / pf["pnl_realised"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _fan_chart(pred, axes[0, 0])
    _pnl_curve(fc, pf, axes[0, 1])
    _pnl_hist(fc, axes[1, 0])
    # capture text panel
    m = TM.report(fc["pnl_realised"], batt["capacity_mwh"])
    axes[1, 1].axis("off")
    txt = (
        "TRADER SCORECARD (battery arbitrage, forecast strategy)\n"
        f"{'-' * 46}\n"
        f"% of perfect foresight : {capture * 100:6.1f}%\n"
        f"Total P&L              : {m['total_eur']:10,.0f} EUR\n"
        f"  per MWh capacity      : {m['eur_per_mwh_cap']:10,.0f} EUR/MWh\n"
        f"Sharpe (annualised)    : {m['sharpe_ann']:10.2f}\n"
        f"VaR 95% (daily)        : {m['VaR95_daily']:10,.0f} EUR\n"
        f"CVaR 95% (daily)       : {m['CVaR95_daily']:10,.0f} EUR\n"
        f"Max drawdown           : {m['max_drawdown']:10,.0f} EUR\n"
        f"Profitable days        : {m['pct_profitable_days']:9.1f}%\n"
        f"Worst / best day       : {m['worst_day']:,.0f} / {m['best_day']:,.0f} EUR\n"
    )
    axes[1, 1].text(0.0, 0.95, txt, family="monospace", va="top", fontsize=10)

    fig.tight_layout()
    out = config.RESULTS / "dashboard.png"
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")

    print("\n" + txt)
    return m


if __name__ == "__main__":
    run()

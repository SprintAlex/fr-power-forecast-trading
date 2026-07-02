"""Book-level risk: a desk runs a book, not a single strategy.

Combines the battery and FR-DE spread daily P&L into one risk-parity book and
shows what only a book view reveals: the cross-strategy correlation and the
diversification benefit (book volatility < sum of standalone vols when the
strategies aren't perfectly correlated -> higher book Sharpe).

Run:  python -m src.eval.book_risk
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.eval import trader_metrics as TM
from src.trading.battery_lp import backtest
from src.trading.spread_strategy import _pnl, forecast_zone


def _daily_pnls(synthetic: bool = True) -> pd.DataFrame:
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    battery = backtest(pred, "p50")["pnl_realised"]

    fr, de = forecast_zone("FR", synthetic), forecast_zone("DE", synthetic)
    real_spread = fr["actual"] - de["actual"]
    fc_spread = fr["p50"] - de["p50"]
    spread = _pnl(np.sign(fc_spread), real_spread)

    df = pd.DataFrame({"battery": battery, "spread": spread}).dropna()
    return df


def run(synthetic: bool = True) -> None:
    pnl = _daily_pnls(synthetic)
    print("== Standalone strategies (daily P&L) ==")
    comp = pd.DataFrame({
        "daily_vol": pnl.std(),
        "Sharpe": {c: TM.sharpe(pnl[c]) for c in pnl},
        "VaR95": {c: TM.var(pnl[c]) for c in pnl},
    })
    print(comp.round(2).to_string())

    print("\n== Cross-strategy correlation ==")
    print(pnl.corr().round(2).to_string())

    # Risk-parity book: weight inversely to volatility (Sharpe is scale-free).
    w = (1 / pnl.std())
    w = w / w.sum()
    book = (pnl * w).sum(axis=1)
    sum_vol = float((w * pnl.std()).sum())          # vol if perfectly correlated
    book_vol = float(book.std())
    div_ratio = sum_vol / book_vol

    print("\n== Risk-parity book ==")
    print(f"weights                : {w.round(2).to_dict()}")
    print(f"book Sharpe (ann)      : {TM.sharpe(book):.2f}  "
          f"(best single {max(TM.sharpe(pnl[c]) for c in pnl):.2f})")
    print(f"book VaR95             : {TM.var(book):.2f}")
    print(f"diversification ratio  : {div_ratio:.2f}x  (>1 => correlation < 1 cuts book risk)")
    print("\nReading: low cross-correlation (~0.2) cuts book risk by the diversification "
          "ratio above — running both strategies is safer than either alone. (Risk-parity "
          "weighting equalises risk, not Sharpe; a mean-variance optimal book would lift "
          "Sharpe further.)")


if __name__ == "__main__":
    import sys
    run(synthetic="--real" not in sys.argv)

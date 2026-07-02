"""Robustness: multi-seed dispersion + parameter sensitivity.

Multi-seed kills 'lucky single run' — re-draw the synthetic world N times and
report mean ± std of the headline numbers. Sensitivity shows how the verdict
moves with battery sizing, efficiency, reserve price and forecast quality — the
'what drives the P&L' table a desk asks for.

Run:  python -m src.eval.robustness            # 5 seeds + sensitivity sweep
      python -m src.eval.robustness 8
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from src.data import synthetic
from src.eval import metrics as M
from src.eval import trader_metrics as TM
from src.forecast.features import FEATURE_COLS, build_features, xy
from src.forecast.train import TEST_START, _fit_block
from src.trading.battery_lp import backtest


def _one_world(seed: int) -> dict:
    raw = synthetic.generate("FR", seed=seed)
    feat = build_features(raw)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    ts = pd.Timestamp(TEST_START, tz="UTC")
    train, test = feat[feat.index < ts], feat[feat.index >= ts]
    preds, _, _ = _fit_block(train, test, cols)
    preds["actual"] = test["target"]
    preds["naive_d7"] = test["price_lag_168"]
    fc = backtest(preds, "p50")
    pf = backtest(preds, "actual")
    return {
        "R2": M.r2(preds["actual"], preds["p50"]),
        "capture_%": fc["pnl_realised"].sum() / pf["pnl_realised"].sum() * 100,
        "PnL": fc["pnl_realised"].sum(),
        "Sharpe": TM.sharpe(fc["pnl_realised"]),
    }


def multi_seed(n: int = 5) -> pd.DataFrame:
    print(f"== Multi-seed robustness ({n} synthetic worlds) ==")
    rows = [_one_world(s) for s in range(n)]
    df = pd.DataFrame(rows)
    summary = df.agg(["mean", "std"]).T
    print(summary.round(2).to_string())
    print("Reading: tight std => results are structural, not a lucky draw.\n")
    return summary


def sensitivity() -> None:
    """Re-backtest the saved predictions under perturbed assumptions (cheap)."""
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    pf_base = backtest(pred, "actual")["pnl_realised"].sum()

    print("== Sensitivity (battery + forecast-quality sweeps) ==")
    rows = []
    for cap in [1.0, 2.0, 4.0]:
        b = {**config.BATTERY, "capacity_mwh": cap}
        pnl = backtest(pred, "p50", b)["pnl_realised"].sum()
        rows.append({"knob": f"capacity {cap} MWh", "PnL": pnl})
    for eff in [0.90, 0.95, 0.99]:
        b = {**config.BATTERY, "eff_charge": eff, "eff_discharge": eff}
        pnl = backtest(pred, "p50", b)["pnl_realised"].sum()
        rows.append({"knob": f"eff {eff}", "PnL": pnl})
    # Forecast-quality sweep: degrade P50 with noise -> capture should fall.
    rng = np.random.default_rng(0)
    for noise in [0, 15, 30]:
        p = pred.copy()
        p["p50"] = pred["p50"] + rng.normal(0, noise, len(pred))
        pnl = backtest(p, "p50")["pnl_realised"].sum()
        rows.append({"knob": f"forecast +N(0,{noise})", "PnL": pnl,
                     "capture_%": pnl / pf_base * 100})
    tab = pd.DataFrame(rows).set_index("knob")
    print(tab.round(0).to_string())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
    multi_seed(n)
    sensitivity()

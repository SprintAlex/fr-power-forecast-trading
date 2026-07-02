"""Regime stress test — does the model hold through the 2022 gas crisis?

Yearly expanding walk-forward: train on all prior years, predict the next whole
year out-of-sample. The hard case is 2022 — the model is trained only on calm
2021 and must face a crisis it has never seen (gas TTF spiking to ~339). That is
exactly where a desk model earns or loses its trust.

Per year reports forecast skill (MAE/R²/coverage) AND battery P&L + % perfect-
foresight captured, so you see whether trading performance survives the regime
break, not just the error metric.

Run:  python -m src.eval.regime_test            # synthetic (has a 2022 crisis)
      python -m src.eval.regime_test --real      # real ENTSO-E once pulled
"""
from __future__ import annotations

import sys

import pandas as pd

import config
from src.eval import metrics as M
from src.forecast.features import FEATURE_COLS, build_features, load_raw
from src.forecast.train import _fit_block
from src.trading.battery_lp import backtest

YEARS = [2022, 2023, 2024]


def run(synthetic: bool = True) -> pd.DataFrame:
    raw = load_raw(synthetic=synthetic)
    feat = build_features(raw)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    gas_year = raw["gas_ttf_eur_mwh"].groupby(raw.index.year).mean() if "gas_ttf_eur_mwh" in raw else None

    rows = []
    for y in YEARS:
        y0 = pd.Timestamp(f"{y}-01-01", tz="UTC")
        y1 = pd.Timestamp(f"{y + 1}-01-01", tz="UTC")
        train = feat[feat.index < y0]
        test = feat[(feat.index >= y0) & (feat.index < y1)]
        if train.empty or test.empty:
            continue
        preds, _, _ = _fit_block(train, test, cols)
        df = preds.copy()
        df["actual"] = test["target"]

        y_true, p50 = df["actual"], df["p50"]
        fc = backtest(df, decision_col="p50")
        pf = backtest(df, decision_col="actual")
        capture = fc["pnl_realised"].sum() / pf["pnl_realised"].sum()
        rows.append(
            {
                "year": y,
                "gas_mean": float(gas_year.get(y)) if gas_year is not None else float("nan"),
                "price_mean": float(y_true.mean()),
                "MAE": M.mae(y_true, p50),
                "R2": M.r2(y_true, p50),
                "cover80": M.coverage(y_true, df["p10_cal"], df["p90_cal"]) * 100,
                "batt_PnL": fc["pnl_realised"].sum(),
                "%_perfect": capture * 100,
            }
        )
    tab = pd.DataFrame(rows).set_index("year")
    print("Yearly expanding walk-forward (train < year, predict year) — crisis stress test\n")
    print(tab.round({"gas_mean": 0, "price_mean": 1, "MAE": 2, "R2": 3,
                     "cover80": 1, "batt_PnL": 0, "%_perfect": 1}).to_string())

    print("\n--- READING ---")
    if 2022 in tab.index:
        r = tab.loc[2022]
        print(f"2022 crisis (gas {r['gas_mean']:.0f}, price {r['price_mean']:.0f}): "
              f"MAE {r['MAE']:.1f}, captures {r['%_perfect']:.0f}% of perfect, "
              f"battery P&L {r['batt_PnL']:+,.0f} EUR.")
        print("Crisis = highest price level -> largest spreads -> biggest absolute P&L, but also "
              "where forecast error is most expensive. Watch % captured, not raw euros.")
    return tab


if __name__ == "__main__":
    run(synthetic="--real" not in sys.argv)

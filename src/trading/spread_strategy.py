"""FR-DE location spread: directional trading + sign accuracy.

The cross-zone hero metric desks watch: not the price level but the **sign of the
spread**. A model with worse MAE but better spread-sign hit-rate makes more money
on a spread book. Here we forecast each zone's day-ahead price, take a 1 MW
directional spread position = sign(forecast spread) each hour, and settle at the
realised spread net of a transaction cost.

Benchmarks: naive (sign of last week's spread) and perfect foresight (sign of the
realised spread = ceiling).

Run:  python -m src.trading.spread_strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data.synthetic import syn_path
from src.eval import trader_metrics as TM
from src.forecast.features import FEATURE_COLS, build_features, xy
from src.forecast.lgbm_quantile import QuantileLGBM
from src.forecast.train import TEST_START

SPREAD_TX = 0.5            # EUR/MWh per leg traded (round trip on the spread)


def forecast_zone(zone: str, synthetic: bool = True) -> pd.DataFrame:
    """Single-fit P50 forecast for one zone; returns test actual + p50 + lag168."""
    from src.data.entsoe_pull import raw_path
    raw = pd.read_parquet(syn_path(zone) if synthetic else raw_path(zone))
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    feat = build_features(raw, zone=zone)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    ts = pd.Timestamp(TEST_START, tz="UTC")
    train, test = feat[feat.index < ts], feat[feat.index >= ts]
    q = QuantileLGBM(quantiles=[0.5]).fit(*xy(train, cols))
    p50 = q.predict(xy(test, cols)[0])["p50"]
    return pd.DataFrame({"actual": test["target"], "p50": p50,
                         "lag168": test["price_lag_168"]}, index=test.index)


def _pnl(position: pd.Series, real_spread: pd.Series) -> pd.Series:
    """Daily P&L of a 1 MW directional spread position, net of tx cost."""
    gross = position * real_spread
    # cost on every hour we hold a non-zero position (open the leg pair).
    cost = SPREAD_TX * position.abs()
    net = gross - cost
    day = net.index.tz_convert(config.ZONE_TZ).normalize()
    return net.groupby(day).sum()


def run(synthetic: bool = True) -> pd.DataFrame:
    fr = forecast_zone("FR", synthetic)
    de = forecast_zone("DE", synthetic)
    df = pd.DataFrame(index=fr.index)
    df["real_spread"] = fr["actual"] - de["actual"]
    df["fc_spread"] = fr["p50"] - de["p50"]
    df["naive_spread"] = fr["lag168"] - de["lag168"]

    signals = {
        "naive (D-7 sign)": np.sign(df["naive_spread"]),
        "model (P50 sign)": np.sign(df["fc_spread"]),
        "perfect foresight": np.sign(df["real_spread"]),
    }
    C = 1.0  # 1 MW spread position
    rows = []
    for label, pos in signals.items():
        acc = float((pos == np.sign(df["real_spread"])).mean())
        pnl = _pnl(pos, df["real_spread"])
        m = TM.report(pnl, C)
        rows.append({"strategy": label, "sign_acc_%": acc * 100,
                     "P&L_eur": m["total_eur"], "Sharpe": m["sharpe_ann"],
                     "VaR95": m["VaR95_daily"], "win_days_%": m["pct_profitable_days"]})
    tab = pd.DataFrame(rows).set_index("strategy")

    print(f"FR-DE spread | test {df.index.min().date()}..{df.index.max().date()} | "
          f"tx {SPREAD_TX} EUR/MWh | real spread mean {df['real_spread'].mean():.1f} std {df['real_spread'].std():.1f}\n")
    print(tab.round({"sign_acc_%": 1, "P&L_eur": 0, "Sharpe": 2, "VaR95": 0, "win_days_%": 1}).to_string())

    model = tab.loc["model (P50 sign)"]
    naive = tab.loc["naive (D-7 sign)"]
    print("\n--- SPREAD VERDICT ---")
    print(f"Sign accuracy          : {model['sign_acc_%']:.1f}%  (naive {naive['sign_acc_%']:.1f}%, coin-flip 50%)")
    print(f"Model spread P&L       : {model['P&L_eur']:,.0f} EUR  (naive {naive['P&L_eur']:,.0f})")
    print(f"Profitable (net tx)    : {'YES' if model['P&L_eur'] > 0 else 'NO'}")
    print(f"Beats naive            : {'YES' if model['P&L_eur'] > naive['P&L_eur'] else 'NO'}")
    return tab


if __name__ == "__main__":
    import sys
    run(synthetic="--real" not in sys.argv)

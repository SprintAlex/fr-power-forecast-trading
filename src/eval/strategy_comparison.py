"""Desk verdict: does a better forecast make more money?

Runs the battery LP on each forecast's price signal, settles every schedule at
the SAME realised prices, and ranks by realised P&L / risk. This is how a desk
decides if a model earns its keep — not MAE, but euros and Sharpe net of cost.

A model is "worth it" only if it beats the cheap naive_d7 forecast in realised
P&L. Perfect foresight is the ceiling; % captured is the forecast's trading grade.

Run:  python -m src.eval.strategy_comparison
"""
from __future__ import annotations

import pandas as pd

import config
from src.eval import trader_metrics as TM
from src.forecast.train import PRED_PATH
from src.trading.battery_lp import backtest, backtest_robust

# decision signal -> label. 'actual' = perfect-foresight ceiling.
SIGNALS = {
    "naive_d7": "naive D-7",
    "lear": "LEAR",
    "p50": "LightGBM P50",
    "actual": "perfect foresight",
}


def run() -> pd.DataFrame:
    pred = pd.read_parquet(PRED_PATH)
    C = config.BATTERY["capacity_mwh"]

    bt = {sig: backtest(pred, decision_col=sig) for sig in SIGNALS}

    # Ordered strategy -> daily P&L, with the risk-aware robust run slotted in.
    series = {SIGNALS[s]: bt[s]["pnl_realised"] for s in ["naive_d7", "lear", "p50"]}
    if {"p10_cal", "p90_cal"} <= set(pred.columns):
        series["LightGBM robust (P10/P90)"] = backtest_robust(pred)["pnl_realised"]
    series["perfect foresight"] = bt["actual"]["pnl_realised"]
    ceiling = series["perfect foresight"].sum()

    rows = []
    for label, pnl in series.items():
        m = TM.report(pnl, C)
        rows.append(
            {
                "strategy": label,
                "P&L_eur": m["total_eur"],
                "eur/MWh-cap": m["eur_per_mwh_cap"],
                "%_perfect": pnl.sum() / ceiling * 100,
                "Sharpe": m["sharpe_ann"],
                "VaR95": m["VaR95_daily"],
                "maxDD": m["max_drawdown"],
                "win_days_%": m["pct_profitable_days"],
                "worst_day": m["worst_day"],
            }
        )
    tab = pd.DataFrame(rows).set_index("strategy")

    print(f"\nBattery 1MW/2MWh | throughput cost {config.BATTERY['cost_per_mwh_cycle']} EUR/MWh "
          f"| test {pred.index.min().date()}..{pred.index.max().date()}\n")
    print(tab.round({"P&L_eur": 0, "eur/MWh-cap": 0, "%_perfect": 1, "Sharpe": 2,
                     "VaR95": 0, "maxDD": 0, "win_days_%": 1, "worst_day": 0}).to_string())

    # Verdict logic
    lgbm = tab.loc["LightGBM P50"]
    naive = tab.loc["naive D-7"]
    edge = lgbm["P&L_eur"] - naive["P&L_eur"]
    print("\n--- DESK VERDICT ---")
    print(f"LightGBM realised P&L  : {lgbm['P&L_eur']:,.0f} EUR ({lgbm['%_perfect']:.0f}% of perfect)")
    naive_note = "naive loses money" if naive["P&L_eur"] <= 0 else f"naive {naive['P&L_eur']:,.0f}"
    print(f"Edge vs naive D-7      : {edge:+,.0f} EUR  ({naive_note})")
    profitable = lgbm["P&L_eur"] > 0
    beats_naive = lgbm["P&L_eur"] > naive["P&L_eur"]
    print(f"Profitable (net cost)  : {'YES' if profitable else 'NO'}")
    print(f"Beats cheap benchmark  : {'YES' if beats_naive else 'NO'}")
    return tab


if __name__ == "__main__":
    run()

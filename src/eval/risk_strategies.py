"""Risk treatment of the battery: point (P50) vs robust (P10/P90) vs CVaR.

Same forecast, three ways of using its uncertainty in the decision:
  - P50         : trade the median (uncertainty ignored in the decision)
  - robust      : pessimistic P10/P90 valuation (heuristic)
  - CVaR        : stochastic optimisation over correlated scenario paths
All settled at realised prices. Shows the risk/return trade the quantiles enable.

Run:  python -m src.eval.risk_strategies
"""
from __future__ import annotations

import pandas as pd

import config
from src.eval import trader_metrics as TM
from src.forecast.scenarios import gen_scenarios
from src.trading.battery_lp import (_settle, optimize_day, optimize_day_cvar,
                                     optimize_day_robust)

N_SCEN = 120
LAM = 1.0


def run() -> pd.DataFrame:
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    batt = config.BATTERY
    cost = batt["cost_per_mwh_cycle"]
    day = pred.index.tz_convert(config.ZONE_TZ).normalize()

    out = {"P50": [], "robust P10/P90": [], "CVaR stochastic": []}
    days = []
    for d, g in pred.groupby(day):
        if len(g) < 23:
            continue
        act = g["actual"].to_numpy()
        days.append(d)
        out["P50"].append(_settle(optimize_day(g["p50"].to_numpy(), batt), act, cost, batt))
        out["robust P10/P90"].append(
            _settle(optimize_day_robust(g["p10_cal"].to_numpy(), g["p90_cal"].to_numpy(), batt), act, cost, batt))
        sc = gen_scenarios(g["p10_cal"], g["p50"], g["p90_cal"], n=N_SCEN, seed=int(d.value % 2**31))
        out["CVaR stochastic"].append(_settle(optimize_day_cvar(sc, batt, lam=LAM), act, cost, batt))

    C = batt["capacity_mwh"]
    rows = []
    for k, v in out.items():
        pnl = pd.Series(v, index=days)
        m = TM.report(pnl, C)
        rows.append({"strategy": k, "P&L_eur": m["total_eur"], "Sharpe": m["sharpe_ann"],
                     "VaR95": m["VaR95_daily"], "maxDD": m["max_drawdown"],
                     "win_days_%": m["pct_profitable_days"], "worst_day": m["worst_day"]})
    tab = pd.DataFrame(rows).set_index("strategy")
    print(f"Battery risk treatments | {N_SCEN} scenarios | CVaR lambda {LAM} "
          f"| test {pred.index.min().date()}..{pred.index.max().date()}\n")
    print(tab.round({"P&L_eur": 0, "Sharpe": 2, "VaR95": 0, "maxDD": 0,
                     "win_days_%": 1, "worst_day": 0}).to_string())
    print("\nReading: P50 maximises P&L; robust/CVaR trade P&L for lower tail risk "
          "(VaR, worst day) — the risk/return choice the quantile forecast enables.")
    return tab


if __name__ == "__main__":
    run()

"""Battery revenue: arbitrage only vs arbitrage + frequency-reserve stacking.

Shows the real battery business case — energy arbitrage alone underuses the asset;
committing spare capacity to reserve (FCR) adds a steady availability payment.
Both schedules are decided on the P50 forecast; energy settles at realised price,
reserve earns its capacity payment.

Run:  python -m src.trading.value_stacking
"""
from __future__ import annotations

import pandas as pd

import config
from src.trading.battery_lp import _settle, optimize_day, optimize_day_reserve


def run() -> pd.DataFrame:
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    batt = config.BATTERY
    cost = batt["cost_per_mwh_cycle"]
    rprice = batt["reserve_price_eur_mw_h"]
    day = pred.index.tz_convert(config.ZONE_TZ).normalize()

    arb_only, stack_arb, stack_res = 0.0, 0.0, 0.0
    for d, grp in pred.groupby(day):
        if len(grp) < 23:
            continue
        dec, act = grp["p50"].to_numpy(), grp["actual"].to_numpy()
        arb_only += _settle(optimize_day(dec, batt), act, cost, batt)
        s = optimize_day_reserve(dec, batt)
        stack_arb += _settle(s, act, cost, batt)
        stack_res += rprice * s["reserve"].sum()

    C = batt["capacity_mwh"]
    tab = pd.DataFrame({
        "arbitrage": [arb_only, 0.0, arb_only],
        "reserve": [0.0, 0.0, 0.0],
        "total": [arb_only, 0.0, arb_only],
    }, index=["arbitrage-only", "_", "stacked"])
    tab.loc["stacked"] = [stack_arb, stack_res, stack_arb + stack_res]
    tab = tab.drop("_")
    tab["eur/MWh-cap"] = tab["total"] / C

    print(f"Battery {batt['power_mw']}MW/{C}MWh | reserve {rprice} EUR/MW/h | "
          f"test {pred.index.min().date()}..{pred.index.max().date()}\n")
    print(tab.round(0).to_string())
    uplift = (tab.loc["stacked", "total"] / tab.loc["arbitrage-only", "total"] - 1) * 100
    print(f"\nReserve stacking uplift : +{uplift:.0f}% total revenue "
          f"(+{tab.loc['stacked', 'reserve']:,.0f} EUR reserve, "
          f"arbitrage {tab.loc['stacked','arbitrage'] - tab.loc['arbitrage-only','arbitrage']:+,.0f} EUR vs arb-only)")
    return tab


if __name__ == "__main__":
    run()

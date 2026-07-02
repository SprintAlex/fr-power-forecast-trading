"""Day-ahead + intraday two-stage battery — the desk doesn't stop at the DA gate.

After the DA auction clears, a sharper forecast arrives closer to delivery. The
battery re-optimises and trades the *adjustment* on the intraday (XBID) market at
the intraday price. This recovers P&L that a DA-only schedule leaves on the table
when the DA forecast was wrong.

Two-stage settlement:
  Stage 1 (DA gate): schedule x_da on the DA forecast (P50), sold at the DA price.
  Stage 2 (ID gate): re-optimise to x_id on a refreshed (more accurate) forecast;
                     trade the delta (x_id - x_da) on intraday at the ID price.
  P&L = DA_price · x_da + ID_price · (x_id - x_da) - throughput cost.

Synthetic ID inputs (real version needs actual XBID data): the ID forecast is a
blend pulling toward outturn (info gained as gate nears); the ID price is the DA
outturn plus a small intraday premium.

Run:  python -m src.trading.intraday
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.trading.battery_lp import execution_cost, optimize_day


def run(seed: int = 0) -> dict:
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    batt = config.BATTERY
    cost = batt["cost_per_mwh_cycle"]
    rng = np.random.default_rng(seed)
    day = pred.index.tz_convert(config.ZONE_TZ).normalize()

    da_only, with_id = 0.0, 0.0
    for d, g in pred.groupby(day):
        if len(g) < 23:
            continue
        actual = g["actual"].to_numpy()
        da_fc = g["p50"].to_numpy()
        # Refreshed ID forecast: pulled toward outturn (info gained near delivery).
        id_fc = 0.4 * da_fc + 0.6 * actual + rng.normal(0, 6, len(actual))
        # Intraday price = DA outturn + small intraday premium/noise.
        id_price = actual + rng.normal(0, 8, len(actual))

        s_da = optimize_day(da_fc, batt)
        s_id = optimize_day(id_fc, batt)
        net_da, net_id = s_da["net_mwh"], s_id["net_mwh"]

        da_only += float(actual @ net_da - cost * s_da["discharge"].sum()) - execution_cost(s_da, batt)
        with_id += (float(actual @ net_da + id_price @ (net_id - net_da)
                          - cost * s_id["discharge"].sum()) - execution_cost(s_id, batt))

    C = batt["capacity_mwh"]
    uplift = (with_id / da_only - 1) * 100
    print(f"Battery DA vs DA+intraday | test {pred.index.min().date()}..{pred.index.max().date()}\n")
    print(f"DA-only P&L            : {da_only:11,.0f} EUR ({da_only / C:,.0f}/MWh-cap)")
    print(f"DA + intraday P&L      : {with_id:11,.0f} EUR ({with_id / C:,.0f}/MWh-cap)")
    print(f"Intraday uplift        : +{with_id - da_only:,.0f} EUR  (+{uplift:.0f}%)")
    print("\nReading: re-optimising on a sharper intraday forecast corrects DA timing "
          "errors; the delta trades at the ID price. Real desks run exactly this loop.")
    return {"da_only": da_only, "with_id": with_id}


if __name__ == "__main__":
    run()

"""Battery arbitrage via linear programming (cvxpy).

The trading layer that turns a price forecast into P&L. Each market day we solve
a small LP for the charge/discharge schedule that maximises arbitrage profit
given a *decision* price vector, subject to battery physics.

The honest backtest (the whole point):
  - optimise the schedule on the **forecast** price (what a desk actually has at
    gate), then **settle that schedule at the realised** price. Forecast error
    costs real money -> this is the true strategy P&L.
  - the **perfect-foresight** run optimises directly on realised prices = the
    theoretical ceiling. `% captured = forecast P&L / perfect P&L` is the single
    number that says how good the forecast is *for trading* (not for MAE).

Battery model (1h steps, so MW == MWh per step):
  soc_{t+1} = soc_t + eff_c * charge_t - discharge_t / eff_d
  0 <= charge,discharge <= power ;  0 <= soc <= capacity
  daily throughput cap: sum(discharge) <= max_cycles_per_day * capacity
  soc resets to soc_init each market day (independent daily arbitrage).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

import config


def optimize_day(prices: np.ndarray, batt: dict) -> dict:
    """LP for one day. Returns schedule + optimised value (on `prices`)."""
    n = len(prices)
    P, C = batt["power_mw"], batt["capacity_mwh"]
    ec, ed = batt["eff_charge"], batt["eff_discharge"]
    soc0 = batt["soc_init_frac"] * C
    cyc = batt["max_cycles_per_day"]

    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)

    cons = [soc[0] == soc0, soc >= 0, soc <= C, charge <= P, discharge <= P]
    for t in range(n):
        cons.append(soc[t + 1] == soc[t] + ec * charge[t] - discharge[t] / ed)
    cons.append(cp.sum(discharge) <= cyc * C)            # anti-degradation cap

    # Profit = energy arbitrage - throughput (degradation+fee) cost. The cost
    # term stops the battery cycling for spreads thinner than it.
    cost = batt.get("cost_per_mwh_cycle", 0.0)
    profit = prices @ (discharge - charge) - cost * cp.sum(discharge)
    prob = cp.Problem(cp.Maximize(profit), cons)
    prob.solve(solver=cp.CLARABEL)

    c = np.asarray(charge.value).ravel()
    d = np.asarray(discharge.value).ravel()
    return {"charge": c, "discharge": d, "net_mwh": d - c}


def optimize_day_robust(p_lo: np.ndarray, p_hi: np.ndarray, batt: dict) -> dict:
    """Risk-aware LP using the forecast *distribution*, not just the median.

    Pessimistic valuation: discharge revenue priced at the LOW quantile (P10),
    charge cost priced at the HIGH quantile (P90). The battery only commits to a
    cycle when even the unfavourable end of the interval clears the throughput
    cost -> it trades less, but on spreads it is confident about. This is the
    'why forecast quantiles?' answer: the tails drive the *decision*, not just
    the report.
    """
    n = len(p_lo)
    P, C = batt["power_mw"], batt["capacity_mwh"]
    ec, ed = batt["eff_charge"], batt["eff_discharge"]
    soc0 = batt["soc_init_frac"] * C
    cyc = batt["max_cycles_per_day"]
    cost = batt.get("cost_per_mwh_cycle", 0.0)

    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    cons = [soc[0] == soc0, soc >= 0, soc <= C, charge <= P, discharge <= P]
    for t in range(n):
        cons.append(soc[t + 1] == soc[t] + ec * charge[t] - discharge[t] / ed)
    cons.append(cp.sum(discharge) <= cyc * C)

    profit = p_lo @ discharge - p_hi @ charge - cost * cp.sum(discharge)
    cp.Problem(cp.Maximize(profit), cons).solve(solver=cp.CLARABEL)
    c = np.asarray(charge.value).ravel()
    d = np.asarray(discharge.value).ravel()
    return {"charge": c, "discharge": d, "net_mwh": d - c}


def optimize_day_reserve(prices: np.ndarray, batt: dict) -> dict:
    """Value stacking: co-optimise energy arbitrage + frequency-reserve capacity.

    The real battery business case. Each hour the unit can commit symmetric
    reserve capacity r (MW) for an availability payment, but must keep power AND
    energy headroom to deliver it: discharge+r<=P, charge+r<=P, and 1h of SOC
    room both ways. Reserve competes with arbitrage for the same MW/MWh — the LP
    splits them optimally.
    """
    n = len(prices)
    P, C = batt["power_mw"], batt["capacity_mwh"]
    ec, ed = batt["eff_charge"], batt["eff_discharge"]
    soc0 = batt["soc_init_frac"] * C
    cyc = batt["max_cycles_per_day"]
    cost = batt.get("cost_per_mwh_cycle", 0.0)
    rprice = batt.get("reserve_price_eur_mw_h", 0.0)

    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    reserve = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    cons = [soc[0] == soc0, soc >= 0, soc <= C]
    for t in range(n):
        cons += [soc[t + 1] == soc[t] + ec * charge[t] - discharge[t] / ed,
                 discharge[t] + reserve[t] <= P,        # power headroom up
                 charge[t] + reserve[t] <= P,           # power headroom down
                 soc[t] - reserve[t] / ed >= 0,         # 1h energy headroom to discharge
                 soc[t] + ec * reserve[t] <= C]         # 1h energy headroom to charge
    cons.append(cp.sum(discharge) <= cyc * C)

    arb = prices @ (discharge - charge) - cost * cp.sum(discharge)
    cp.Problem(cp.Maximize(arb + rprice * cp.sum(reserve)), cons).solve(solver=cp.CLARABEL)
    c = np.asarray(charge.value).ravel()
    d = np.asarray(discharge.value).ravel()
    r = np.asarray(reserve.value).ravel()
    return {"charge": c, "discharge": d, "reserve": r, "net_mwh": d - c}


def optimize_day_cvar(scenarios: np.ndarray, batt: dict, lam: float = 1.0,
                      beta: float = 0.95) -> dict:
    """Stochastic battery: one here-and-now schedule over price scenarios,
    maximising  E[profit] - lam * CVaR_beta(loss).

    A single schedule has linear profit in price, so E[profit] alone reduces to
    the mean path — but the CVaR term penalises schedules whose P&L is fragile
    across scenarios, steering the battery away from bets it is uncertain about.
    The principled version of the P10/P90 robust heuristic. (Rockafellar-Uryasev
    CVaR LP.)
    """
    S, n = scenarios.shape
    P, C = batt["power_mw"], batt["capacity_mwh"]
    ec, ed = batt["eff_charge"], batt["eff_discharge"]
    soc0 = batt["soc_init_frac"] * C
    cyc = batt["max_cycles_per_day"]
    cost = batt.get("cost_per_mwh_cycle", 0.0)

    charge = cp.Variable(n, nonneg=True)
    discharge = cp.Variable(n, nonneg=True)
    soc = cp.Variable(n + 1)
    alpha = cp.Variable()                 # VaR level
    u = cp.Variable(S, nonneg=True)       # CVaR tail excess per scenario

    cons = [soc[0] == soc0, soc >= 0, soc <= C, charge <= P, discharge <= P]
    for t in range(n):
        cons.append(soc[t + 1] == soc[t] + ec * charge[t] - discharge[t] / ed)
    cons.append(cp.sum(discharge) <= cyc * C)

    net = discharge - charge
    profit_s = scenarios @ net - cost * cp.sum(discharge)      # (S,)
    cons += [u >= -profit_s - alpha]
    cvar = alpha + cp.sum(u) / ((1 - beta) * S)
    expected = cp.sum(profit_s) / S
    cp.Problem(cp.Maximize(expected - lam * cvar), cons).solve(solver=cp.CLARABEL)
    c = np.asarray(charge.value).ravel()
    d = np.asarray(discharge.value).ravel()
    return {"charge": c, "discharge": d, "net_mwh": d - c}


def execution_cost(sched: dict, batt: dict) -> float:
    """Realistic fill cost: bid-ask half-spread on every MWh + linear market
    impact (cost grows with the square of the volume pushed into the hour). Kills
    the 'infinite liquidity at clearing price' assumption."""
    fee = batt.get("exec_fee_per_mwh", 0.0)
    imp = batt.get("impact_coeff_eur_per_mw", 0.0)
    c, d = sched["charge"], sched["discharge"]
    return float(fee * (c + d).sum() + imp * ((c ** 2 + d ** 2).sum()))


def _settle(sched: dict, prices: np.ndarray, cost: float, batt: dict | None = None) -> float:
    """Cash from a fixed schedule at `prices`, net of throughput (+ execution) cost."""
    cash = float(prices @ sched["net_mwh"] - cost * sched["discharge"].sum())
    if batt is not None:
        cash -= execution_cost(sched, batt)
    return cash


def backtest(pred: pd.DataFrame, decision_col: str, batt: dict | None = None) -> pd.DataFrame:
    """Run the daily LP over the test set.

    decision_col : price column the optimiser sees ('p50' = forecast strategy,
                   'actual' = perfect foresight). Settlement is always at 'actual'.
    Returns a per-day DataFrame: pnl_decision (value on decision prices),
    pnl_realised (schedule settled at actual).
    """
    batt = batt or config.BATTERY
    cost = batt.get("cost_per_mwh_cycle", 0.0)
    local_day = pred.index.tz_convert(config.ZONE_TZ).normalize()
    rows = []
    for day, grp in pred.groupby(local_day):
        if len(grp) < 23:                                # skip DST-broken stub days
            continue
        dec = grp[decision_col].to_numpy()
        act = grp["actual"].to_numpy()
        sched = optimize_day(dec, batt)
        pnl_real = _settle(sched, act, cost, batt)       # decided on forecast, paid at realised
        pnl_dec = _settle(sched, dec, cost, batt)
        rows.append({"day": day, "pnl_realised": pnl_real, "pnl_decision": pnl_dec})
    res = pd.DataFrame(rows).set_index("day")
    return res


def backtest_robust(pred: pd.DataFrame, lo_col: str = "p10_cal", hi_col: str = "p90_cal",
                    batt: dict | None = None) -> pd.DataFrame:
    """Risk-aware backtest: schedule from the robust (P10/P90) LP, settled at actual."""
    batt = batt or config.BATTERY
    cost = batt.get("cost_per_mwh_cycle", 0.0)
    local_day = pred.index.tz_convert(config.ZONE_TZ).normalize()
    rows = []
    for day, grp in pred.groupby(local_day):
        if len(grp) < 23:
            continue
        sched = optimize_day_robust(grp[lo_col].to_numpy(), grp[hi_col].to_numpy(), batt)
        rows.append({"day": day, "pnl_realised": _settle(sched, grp["actual"].to_numpy(), cost, batt)})
    return pd.DataFrame(rows).set_index("day")


if __name__ == "__main__":
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    fc = backtest(pred, decision_col="p50")
    pf = backtest(pred, decision_col="actual")
    capture = fc["pnl_realised"].sum() / pf["pnl_realised"].sum()
    C = config.BATTERY["capacity_mwh"]
    print(f"Days backtested        : {len(fc)}")
    print(f"Forecast P&L (realised): {fc['pnl_realised'].sum():,.0f} EUR "
          f"({fc['pnl_realised'].sum() / C:,.0f} EUR/MWh-cap/yr)")
    print(f"Perfect-foresight P&L  : {pf['pnl_realised'].sum():,.0f} EUR")
    print(f"% of perfect captured  : {capture * 100:.1f}%")
    print(f"Mean daily P&L         : {fc['pnl_realised'].mean():,.1f} EUR | "
          f"worst day {fc['pnl_realised'].min():,.0f} | best {fc['pnl_realised'].max():,.0f}")

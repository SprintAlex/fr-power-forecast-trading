"""Central config: market zone, date range, battery specs, paths.

Decisions locked (see HANDOFF.md s4):
  - Zone       : FR  (TotalEnergies / EDF Trading relevance, deep liquid data)
  - Strategy   : battery arbitrage via LP, settled at realized day-ahead prices
  - Forecast   : LEAR baseline + LightGBM quantile (P10/P50/P90)
"""
from __future__ import annotations

from pathlib import Path

# --- Market zone -----------------------------------------------------------
ZONE = "FR"                      # entsoe-py country_code
ZONE_TZ = "Europe/Paris"         # local market time (CET/CEST); auctions are local-hour
ZONE_EIC = "10YFR-RTE------C"    # France bidding zone EIC, for reference

# --- Backtest window -------------------------------------------------------
# 2021-2024: spans the 2022 energy crisis -> fat right tail + negative prices,
# exactly the regime where trader metrics (spike recall, VaR) bite.
START = "2021-01-01"
END = "2025-01-01"               # exclusive upper bound

# --- Battery (asset under optimization) ------------------------------------
# 1 MW / 2 MWh grid battery, 2-hour duration, the standard arbitrage unit.
BATTERY = dict(
    power_mw=1.0,                # max charge/discharge rate
    capacity_mwh=2.0,            # usable energy
    eff_charge=0.95,             # one-way; round-trip = 0.95*0.95 = 0.9025
    eff_discharge=0.95,
    soc_init_frac=0.0,           # start empty
    max_cycles_per_day=1.0,      # throughput cap (anti-degradation)
    cost_per_mwh_cycle=3.0,      # degradation + grid fees per MWh discharged,
    reserve_price_eur_mw_h=6.0,  # FCR availability payment per committed MW per hour
    exec_fee_per_mwh=0.30,       # bid-ask half-spread paid on every MWh traded
    impact_coeff_eur_per_mw=0.40,  # linear *marginal* impact -> quadratic cost in trade size (see execution_cost)
)                                # the friction that makes "profitable?" non-trivial

# --- Forecast quantiles ----------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]      # P10 / P50 / P90

# --- Gate discipline -------------------------------------------------------
# Day-ahead auction gate closes 12:00 local D-1. Only features known by then
# are legal: lagged prices (D-1, D-2, D-7), TSO load/wind/solar *forecasts*,
# calendar. (Realized post-clearing variables = leakage; see HANDOFF s1.)
GATE_HOUR_LOCAL = 12

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for _p in (DATA_RAW, DATA_PROCESSED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

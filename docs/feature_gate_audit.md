# Feature gate audit — day-ahead FR

**The decision timestamp (gate): 12:00 local, D-1.** We forecast all 24 hours of
delivery day **D** in one shot. A feature is *legal* only if its value is already
knowable by 12:00 D-1. This table is the leakage-discipline proof — every column
in `src/forecast/features.py::FEATURE_COLS`, plus the tempting illegals we exclude.

## Legal features (used)

| Feature | Value determined | Known by 12:00 D-1? | Source |
|---|---|---|---|
| `price_lag_24` | same hour, D-1 | ✅ cleared at the D-2 auction | realized DA price |
| `price_lag_48` | same hour, D-2 | ✅ | realized DA price |
| `price_lag_72` | same hour, D-3 | ✅ | realized DA price |
| `price_lag_168` | same hour, D-7 | ✅ | realized DA price |
| `price_d1_mean/min/max` | D-1 daily aggregates | ✅ D-1 fully cleared at the D-2 auction | realized DA price |
| `load_fc_mw` | TSO day-ahead **load forecast** for D | ✅ published morning of D-1 (ENTSO-E A01) | TSO forecast |
| `res_wind_total_mw` | TSO day-ahead **wind forecast** for D | ✅ published morning of D-1 | TSO forecast |
| `res_solar_mw` | TSO day-ahead **solar forecast** for D | ✅ published morning of D-1 | TSO forecast |
| `residual_load_mw` | load_fc − wind_fc − solar_fc | ✅ derived from legal forecasts | derived |
| `gas_ttf_eur_mwh` | TTF settle, **lagged 1 calendar day** | ✅ previous-day settle, no same-day peek | yfinance `TTF=F` |
| `co2_eur_t` | EUA settle, **lagged 1 calendar day** | ✅ previous-day settle | yfinance `CO2.L` |
| `gas_chg_7d` | 7-day % change of lagged gas | ✅ derived from legal gas | derived |
| `load_ramp_24h` | Δ of load forecast vs 24h earlier | ✅ derived from legal forecast | derived |
| `hour`,`dow`,`month`,`is_weekend`,`hour_sin`,`hour_cos` | calendar of delivery hour | ✅ deterministic | calendar |
| `is_holiday` | public-holiday flag, **zone-specific** | ✅ published years ahead | `holidays` lib |

## Illegal — tempting but excluded (would be leakage)

| Would-be feature | Value determined | Why it's a leak |
|---|---|---|
| `BalancingPowerPrice*` / imbalance prices | **after** market clearing, real time | the SIC7002 war story: ≈ spot in 73 % of rows, corr ~1.0 → it *is* the answer |
| realized load / wind / solar **outturn** for D | after delivery | not known at gate; only the *forecast* is legal |
| price lags < 24h (`lag_1`, `lag_12`) | during/after D | not yet cleared at the 12:00 D-1 gate |
| same-day gas/CO2 close | D close (after 12:00) | 1-day lag enforced to avoid same-day peek |

## How it's enforced in code

- Only lags ≥ 24h: `PRICE_LAGS = [24, 48, 72, 168]`, asserted in
  `tests/test_pipeline.py::test_all_price_lags_at_least_24h`.
- Fundamentals lagged 1 day, **no back-fill** (`fundamentals.py::fetch_daily`) — the
  pre-Oct-2021 CO2 gap stays NaN and is dropped, never filled with a future settle.
- No feature is a near-perfect target proxy: corr < 0.99, checked on **synthetic and
  real** data (`test_no_feature_is_target_proxy`, `test_no_leak_on_real_data`).
- Zone-correct holiday calendar (`build_features(df, zone=...)`,
  `test_holiday_calendar_matches_zone`).

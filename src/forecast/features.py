"""Gate-legal feature engineering for day-ahead price forecasting.

THE leakage discipline (HANDOFF s1): the day-ahead auction gate closes 12:00
local on D-1. When we forecast all 24 hours of delivery day D, a feature is
*legal* only if its value is known by 12:00 D-1. Concretely:

  LEGAL
    - Lagged realised prices at horizons >= 24h: same hour on D-1, D-2, D-3, D-7.
      (D-1 prices were cleared at the D-2 auction -> public well before our gate.)
    - Previous-day price aggregates (mean/min/max of D-1), known by gate.
    - TSO day-ahead *forecasts* for day D: load, wind, solar, residual load.
      These are published the morning of D-1 -> legal exogenous inputs.
    - Calendar: hour, day-of-week, month, weekend.

  ILLEGAL (would be leakage)
    - Realised load/wind/solar *outturn* for day D (known only after delivery).
    - Any balancing/imbalance price (set after market clearing).
    - Price lags < 24h (not yet cleared at our gate).

Output is a tidy per-delivery-hour table: one row per (UTC hour), target = the
realised price for that hour, all columns gate-legal. Works as-is for LightGBM
(single model, `hour` as feature) and, filtered by hour, for LEAR's 24 per-hour
LASSO regressions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Same-hour price lags (hours). All >= 24 -> known at the 12:00 D-1 gate.
PRICE_LAGS = [24, 48, 72, 168]            # D-1, D-2, D-3, D-7
# Gate-legal exogenous inputs for the delivery hour: TSO RES/load forecasts +
# gas TTF / CO2 EUA (forward prices known at gate). Fundamentals optional — the
# real ENTSO-E pull won't carry gas/CO2 until a fundamentals source is wired in;
# build_features adds whichever columns are present.
EXOG_COLS = [
    "load_fc_mw", "res_wind_total_mw", "res_solar_mw", "residual_load_mw",
    "gas_ttf_eur_mwh", "co2_eur_t",
]


def load_raw(synthetic: bool = True) -> pd.DataFrame:
    """Load the raw hourly UTC dataset (synthetic placeholder or real ENTSO-E)."""
    from src.data.synthetic import SYN_PATH
    from src.data.entsoe_pull import RAW_PATH

    path = SYN_PATH if synthetic else RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run "
            f"{'python -m src.data.synthetic' if synthetic else 'python -m src.data.entsoe_pull'} first."
        )
    df = pd.read_parquet(path).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-hour feature table with a gate-legal column set + target."""
    df = df.sort_index().copy()
    price = df["price_eur_mwh"]
    out = pd.DataFrame(index=df.index)
    out["target"] = price

    # --- Lagged realised prices (same hour, >=24h back) -------------------
    for lag in PRICE_LAGS:
        out[f"price_lag_{lag}"] = price.shift(lag)

    # --- Previous-day price aggregates (known at gate) -------------------
    daily = price.groupby(price.index.normalize()).agg(["mean", "min", "max"])
    daily_prev = daily.shift(1)                       # D-1 aggregates for day D
    day_key = price.index.normalize()
    out["price_d1_mean"] = daily_prev["mean"].reindex(day_key).to_numpy()
    out["price_d1_min"] = daily_prev["min"].reindex(day_key).to_numpy()
    out["price_d1_max"] = daily_prev["max"].reindex(day_key).to_numpy()

    # --- Gate-legal exogenous forecasts for the delivery hour ------------
    for c in EXOG_COLS:
        if c in df.columns:
            out[c] = df[c]

    # --- Fundamentals momentum (gate-legal: derived from gas/CO2 already legal) -
    if "gas_ttf_eur_mwh" in df.columns:
        out["gas_chg_7d"] = df["gas_ttf_eur_mwh"].pct_change(24 * 7).to_numpy()
    if "load_fc_mw" in df.columns:
        out["load_ramp_24h"] = (df["load_fc_mw"] - df["load_fc_mw"].shift(24)).to_numpy()

    # --- Calendar --------------------------------------------------------
    local = df.index.tz_convert(config.ZONE_TZ)
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    # Public holidays (known in advance -> legal). Country = config zone.
    import holidays as _hol

    hcal = _hol.country_holidays(config.ZONE, years=range(local.year.min(), local.year.max() + 1))
    out["is_holiday"] = np.array([d.date() in hcal for d in local]).astype(int)
    # Cyclical encodings (help tree splits less, but standard for LEAR/linear).
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out = out.dropna()                                # drop warm-up (first 7d)
    return out


FEATURE_COLS = (
    [f"price_lag_{l}" for l in PRICE_LAGS]
    + ["price_d1_mean", "price_d1_min", "price_d1_max"]
    + EXOG_COLS
    + ["gas_chg_7d", "load_ramp_24h"]
    + ["hour", "dow", "month", "is_weekend", "is_holiday", "hour_sin", "hour_cos"]
)


def xy(feat: pd.DataFrame, cols: list[str] | None = None):
    cols = cols or [c for c in FEATURE_COLS if c in feat.columns]
    return feat[cols], feat["target"]


if __name__ == "__main__":
    raw = load_raw(synthetic=True)
    f = build_features(raw)
    X, y = xy(f)
    print(f"Feature table: {f.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Range {f.index.min()} .. {f.index.max()}")
    print("Features:", list(X.columns))
    print("\nLeakage self-check (corr of each feature with target):")
    corr = pd.concat([X, y], axis=1).corr()["target"].drop("target").abs().sort_values(ascending=False)
    print(corr.round(3).to_string())
    print("\n[gate check] no feature should be a near-perfect (>0.99) proxy for target.")

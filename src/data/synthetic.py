"""Synthetic day-ahead datasets — same schema as entsoe_pull, FR and DE-LU.

Lets the forecast / trading / eval layers (incl. the FR-DE spread) be built and
tested before the real ENTSO-E token lands. Columns + UTC-hourly index match the
real pull, so swapping in real data is a path change.

Two zones share the SAME gas TTF / CO2 fuels (one European fuel complex) but have
their own load, RES and latent shocks:
  - FR: nuclear-heavy, moderate wind -> fewer negatives.
  - DE: much more wind+solar -> deeper/more frequent negatives, cheaper when windy.
The FR-DE spread is therefore driven by relative RES + independent outages — the
realistic location-spread story.

Run:  python -m src.data.synthetic        # builds FR + DE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Zone knobs (MW). Gas/CO2 are shared (see _fuels), seeded independently of zone.
ZONES = {
    "FR": dict(load_base=54000, load_amp=12000, wind_base=8000, wind_amp=4000,
               wind_ar=3000, wind_cap=22000, solar_base=8000, solar_amp=7000,
               latent=8.0, surplus_frac=0.52, seed=42),
    "DE": dict(load_base=58000, load_amp=10000, wind_base=13000, wind_amp=6000,
               wind_ar=5000, wind_cap=36000, solar_base=10000, solar_amp=8000,
               latent=8.0, surplus_frac=0.56, seed=7),
}

SYN_PATH = config.DATA_RAW / f"{config.ZONE.lower()}_synthetic_raw.parquet"


def syn_path(zone: str):
    return config.DATA_RAW / f"{zone.lower()}_synthetic_raw.parquet"


def _seasonal(doy, peak_day, amp, base):
    return base + amp * np.cos(2 * np.pi * (doy - peak_day) / 365.25)


def _fuels(idx, local):
    """Shared gas TTF (EUR/MWh) + CO2 EUA (EUR/t), daily, fwd-filled to hourly.
    Fixed seed -> identical fuels across zones (one European fuel complex)."""
    rng = np.random.default_rng(2024)
    day_idx = pd.DatetimeIndex(sorted(local.normalize().unique()))
    nd = len(day_idx)
    dyear = day_idx.year.to_numpy()
    gas_base = pd.Series(dyear).map({2021: 19, 2022: 78, 2023: 35, 2024: 27}).fillna(30).to_numpy()
    co2_base = pd.Series(dyear).map({2021: 52, 2022: 80, 2023: 85, 2024: 67}).fillna(70).to_numpy()
    gw = np.zeros(nd); cw = np.zeros(nd)
    for t in range(1, nd):
        gw[t] = 0.97 * gw[t - 1] + rng.normal(0, 1.0)
        cw[t] = 0.99 * cw[t - 1] + rng.normal(0, 1.0)
    gas_daily = np.clip(gas_base * (1 + 0.18 * gw), 5, None)
    co2_daily = np.clip(co2_base + 6.0 * cw, 5, None)
    d2i = {d: i for i, d in enumerate(day_idx)}
    pos = np.array([d2i[d] for d in local.normalize()])
    return gas_daily[pos], co2_daily[pos], gas_base.mean()


def generate(zone: str = "FR", start=None, end=None, seed: int | None = None) -> pd.DataFrame:
    z = ZONES[zone]
    start = start or config.START
    end = end or config.END
    rng = np.random.default_rng(z["seed"] if seed is None else seed)

    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    local = idx.tz_convert(config.ZONE_TZ)
    hod, dow, doy = local.hour.to_numpy(), local.dayofweek.to_numpy(), local.dayofyear.to_numpy()
    n = len(idx)

    # --- Load (MW) -------------------------------------------------------
    load_season = _seasonal(doy, 15, z["load_amp"], z["load_base"])
    daily = (7000 * np.exp(-0.5 * ((hod - 8) / 2.0) ** 2)
             + 9000 * np.exp(-0.5 * ((hod - 19) / 2.5) ** 2)
             - 6000 * np.exp(-0.5 * ((hod - 4) / 2.0) ** 2))
    weekend = np.where(dow >= 5, -6000.0, 0.0)
    # Public holidays act like weekends (industry off) -> load dip.
    import holidays as _hol

    country = "DE" if zone == "DE" else "FR"
    hcal = _hol.country_holidays(country, years=range(local.year.min(), local.year.max() + 1))
    is_hol = np.array([d.date() in hcal for d in local])
    holiday = np.where(is_hol, -7000.0, 0.0)
    load = np.clip(load_season + daily + weekend + holiday + rng.normal(0, 1500, n), 30000, None)

    # --- Solar (MW) ------------------------------------------------------
    solar_cap = _seasonal(doy, 172, z["solar_amp"], z["solar_base"])
    solar_shape = np.clip(np.sin(np.pi * (hod - 6) / 12.0), 0, None)
    solar = np.clip(solar_cap * solar_shape * rng.uniform(0.6, 1.0, n), 0, None)

    # --- Wind (MW), AR(1) ------------------------------------------------
    wind_season = _seasonal(doy, 15, z["wind_amp"], z["wind_base"])
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.95 * ar[t - 1] + rng.normal(0, 1.0)
    wind = np.clip(wind_season + z["wind_ar"] * ar, 0, z["wind_cap"])

    residual = load - wind - solar

    # --- Shared fuels + merit-order price --------------------------------
    gas_ttf, co2, gas_mean = _fuels(idx, local)
    mc_gas = gas_ttf / 0.50 + 0.35 * co2
    r = (residual - 30000) / 1000.0
    premium = 0.45 * r + 0.012 * np.clip(r, 0, None) ** 2
    price = mc_gas + premium + rng.normal(0, 7, n)
    gas_level = gas_ttf / gas_mean

    # Unobserved AR(1) latent (intraday gas / outages / neighbour shocks), per-zone.
    lat = np.zeros(n)
    for t in range(1, n):
        lat[t] = 0.85 * lat[t - 1] + rng.normal(0, 1.0)
    price = price + z["latent"] * lat * gas_level

    # Negative prices: RES surplus floods a slack hour (curtailment).
    s = np.clip((wind + solar) - z["surplus_frac"] * load, 0, None) / 1000.0
    price = price - 14.0 * s - 1.5 * s ** 2

    # Scarcity spikes (learnable tightness + random outages).
    tight = residual > np.quantile(residual, 0.90)
    spike_hit = ((rng.random(n) < 0.035) & tight) | (rng.random(n) < 0.004)
    price = price + spike_hit * rng.uniform(120, 550, n)

    price = np.clip(price, -500.0, 4000.0)               # EPEX bounds

    df = pd.DataFrame(
        {
            "price_eur_mwh": price,
            "load_fc_mw": load,
            "res_solar_mw": solar,
            "res_wind_onshore_mw": wind,
            "res_wind_total_mw": wind,
            "res_total_mw": wind + solar,
            "residual_load_mw": residual,
            "gas_ttf_eur_mwh": gas_ttf,
            "co2_eur_t": co2,
        },
        index=idx,
    )
    df.index.name = "ts_utc"
    return df


def build(zone: str = "FR") -> pd.DataFrame:
    df = generate(zone)
    path = syn_path(zone)
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    p = df["price_eur_mwh"]
    print(f"[{zone}] {len(df):,} rows -> {path.name} | "
          f"mean {p.mean():.1f} median {p.median():.1f} min {p.min():.0f} max {p.max():.0f} "
          f"neg {(p < 0).mean() * 100:.1f}% >200 {(p > 200).mean() * 100:.1f}%")
    return df


if __name__ == "__main__":
    fr = build("FR")
    de = build("DE")
    spread = fr["price_eur_mwh"] - de["price_eur_mwh"]
    print(f"\nFR-DE spread: mean {spread.mean():.1f} | std {spread.std():.1f} | "
          f"FR>DE {(spread > 0).mean() * 100:.0f}% of hours")

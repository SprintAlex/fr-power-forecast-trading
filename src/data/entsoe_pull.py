"""Pull day-ahead price + day-ahead load/wind/solar forecasts from ENTSO-E.

All series are stored on a single **UTC** hourly index (DST-safe — the leakage
lesson's twin: never let a DST fold silently duplicate or drop an hour). Local
calendar/hour features are derived downstream from the UTC index.

Legality (12:00 D-1 gate): every column here is published the morning of D-1 or
is a realized *price* used only as the target / as lags >= D-1. The load and
RES series are TSO **forecasts** (process_type A01 = day-ahead), not outturn.

Run:  python -m src.data.entsoe_pull            # full config window
      python -m src.data.entsoe_pull 2023 2024  # explicit year range
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

import config

def raw_path(zone: str = config.ZONE):
    return config.DATA_RAW / f"{zone.lower()}_entsoe_raw.parquet"


RAW_PATH = raw_path(config.ZONE)


def _client():
    load_dotenv()
    token = os.getenv("ENTSOE_API_TOKEN")
    if not token or token == "your_token_here":
        raise RuntimeError(
            "ENTSOE_API_TOKEN missing. Copy .env.example -> .env and paste your "
            "ENTSO-E API token (see .env.example for how to obtain it)."
        )
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=token)


def _to_utc_hourly(obj: pd.Series | pd.DataFrame, how: str = "mean"):
    """Force tz-aware UTC index at 1h resolution. Sub-hourly -> aggregated."""
    if obj.index.tz is None:
        obj = obj.tz_localize(config.ZONE_TZ)
    obj = obj.tz_convert("UTC")
    obj = obj[~obj.index.duplicated(keep="first")].sort_index()
    agg = obj.resample("1h")
    return getattr(agg, how)()


def _bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{year}-01-01", tz=config.ZONE_TZ)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=config.ZONE_TZ)
    return start, end


def _retry(fn, *a, tries: int = 3, pause: float = 5.0, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001 - ENTSO-E throws assorted errors
            if i == tries - 1:
                raise
            print(f"    retry {i + 1}/{tries} after error: {e}")
            time.sleep(pause)


# Filename zone -> entsoe-py country_code (DE bidding zone is DE_LU since 2018).
ENTSOE_CODE = {"FR": "FR", "DE": "DE_LU"}


def pull_year(client, year: int, zone: str = config.ZONE) -> pd.DataFrame:
    start, end = _bounds(year)
    code = ENTSOE_CODE.get(zone, zone)
    print(f"  [{zone} {year}] day-ahead prices ...")
    price = _retry(client.query_day_ahead_prices, code, start=start, end=end)
    price = _to_utc_hourly(price, "mean").rename("price_eur_mwh")

    print(f"  [{year}] load forecast (day-ahead) ...")
    load = _retry(client.query_load_forecast, code, start=start, end=end)
    if isinstance(load, pd.DataFrame):
        load = load.iloc[:, 0]
    load = _to_utc_hourly(load, "mean").rename("load_fc_mw")

    print(f"  [{year}] wind/solar forecast (day-ahead) ...")
    res = _retry(client.query_wind_and_solar_forecast, code, start=start, end=end)
    res = _to_utc_hourly(res, "mean")
    res.columns = [
        "res_" + c.lower().replace(" ", "_") for c in res.columns
    ]  # res_solar, res_wind_onshore, [res_wind_offshore]

    df = pd.concat([price, load, res], axis=1)
    return df


def build_dataset(years: list[int] | None = None, zone: str = config.ZONE) -> pd.DataFrame:
    if years is None:
        y0 = pd.Timestamp(config.START).year
        y1 = pd.Timestamp(config.END).year - 1  # END is exclusive
        years = list(range(y0, y1 + 1))
    client = _client()
    frames = []
    for y in years:
        frames.append(pull_year(client, y, zone))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Derived RES totals (handy for residual-load feature later).
    wind_cols = [c for c in df.columns if c.startswith("res_wind")]
    if wind_cols:
        df["res_wind_total_mw"] = df[wind_cols].sum(axis=1)
    if "res_solar" in df.columns:
        df = df.rename(columns={"res_solar": "res_solar_mw"})
    if "res_wind_total_mw" in df.columns and "res_solar_mw" in df.columns:
        df["res_total_mw"] = df["res_wind_total_mw"] + df["res_solar_mw"]
        # Residual load = the fundamental price driver (demand net of free RES).
        df["residual_load_mw"] = df["load_fc_mw"] - df["res_total_mw"]

    # Attach real gas TTF + CO2 EUA fundamentals (gate-legal, 1-day lagged).
    try:
        from src.data.fundamentals import attach

        df = attach(df)
        print("Attached gas TTF + CO2 EUA fundamentals.")
    except Exception as e:  # noqa: BLE001 - fundamentals are optional enrichment
        print(f"WARN: fundamentals not attached ({e}). Forecast runs without gas/CO2.")

    df.index.name = "ts_utc"
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out_path = raw_path(zone)
    df.to_parquet(out_path)
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} cols -> {out_path}")
    print(f"Range {df.index.min()} .. {df.index.max()} (UTC)")
    print("NaN per col:\n" + df.isna().sum().to_string())
    return df


if __name__ == "__main__":
    args = sys.argv[1:]
    zone = next((a.upper() for a in args if a.upper() in ("FR", "DE")), config.ZONE)
    yrs_arg = [a for a in args if a.isdigit()]
    yrs = list(range(int(yrs_arg[0]), int(yrs_arg[1]) + 1)) if len(yrs_arg) == 2 else None
    build_dataset(yrs, zone)

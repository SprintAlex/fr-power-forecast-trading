"""Real gas TTF + CO2 EUA fundamentals via yfinance — the fuel complex.

Power price tracks the marginal gas plant, so gas TTF + CO2 EUA are the #1
fundamental inputs. These are settlement/forward prices, so they're known at the
12:00 D-1 gate — but to stay strictly legal we lag them by one calendar day
(use the previous day's settle), never peeking at same-day close.

Tickers (yfinance):
  TTF=F  Dutch TTF natural gas front-month  (EUR/MWh) -> captures the 2022 spike (peak ~339)
  CO2.L  ICE EUA carbon                      (EUR/t)   -> coverage starts Oct-2021 (bfill earlier)

Run:  python -m src.data.fundamentals        # fetch + save data/raw/fundamentals.parquet
"""
from __future__ import annotations

import pandas as pd

import config

GAS_TICKER = "TTF=F"
CO2_TICKER = "CO2.L"
FUND_PATH = config.DATA_RAW / "fundamentals.parquet"


def _close(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if d.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    c = d["Close"]
    if isinstance(c, pd.DataFrame):           # multi-index columns -> single col
        c = c.iloc[:, 0]
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    return c.rename(ticker)


def fetch_daily(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Daily gas + CO2 on a continuous calendar (weekends/holidays ffilled).

    Only forward-fill: carrying the last known settle over a weekend/holiday is
    gate-legal. We deliberately do NOT back-fill — the CO2.L series starts
    Oct-2021, so earlier rows stay NaN and get dropped downstream. Back-filling
    would inject a future settle into the past (a look-ahead), which is exactly
    the discipline this repo is about (see HANDOFF s1).
    """
    start = start or config.START
    end = end or config.END
    gas = _close(GAS_TICKER, start, end)
    co2 = _close(CO2_TICKER, start, end)
    cal = pd.date_range(start, end, freq="D", inclusive="left")
    df = pd.DataFrame(index=cal)
    df["gas_ttf_eur_mwh"] = gas.reindex(cal).ffill()
    df["co2_eur_t"] = co2.reindex(cal).ffill()   # pre-Oct-2021 stays NaN (dropped later)
    return df


def attach(raw: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Add gate-legal gas/CO2 columns to an hourly-UTC ENTSO-E frame.

    Daily settles are lagged one day (previous-day close) -> no same-day lookahead.
    """
    daily = fetch_daily(start, end).shift(1).ffill()   # 1-day legality lag, no back-fill
    local_date = raw.index.tz_convert(config.ZONE_TZ).normalize().tz_localize(None)
    out = raw.copy()
    out["gas_ttf_eur_mwh"] = daily["gas_ttf_eur_mwh"].reindex(local_date).to_numpy()
    out["co2_eur_t"] = daily["co2_eur_t"].reindex(local_date).to_numpy()
    return out


if __name__ == "__main__":
    df = fetch_daily()
    df.to_parquet(FUND_PATH)
    print(f"Saved {len(df):,} days -> {FUND_PATH}")
    print(df.groupby(df.index.year).agg(["mean", "max"]).round(1).to_string())

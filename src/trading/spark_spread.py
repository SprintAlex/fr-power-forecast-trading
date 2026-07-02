"""Clean spark spread — the gas plant's margin, the fundamental floor of price.

CSS = power_price - gas_TTF / efficiency - CO2 * emission_factor

It's what a CCGT earns per MWh; >0 means gas is in-merit (worth running). Desks
trade and hedge it directly (a generator sells power + CO2, buys gas). Here we:
  - compute realised CSS and report % of hours gas is in-merit, per year,
  - score the model's CSS forecast (does it predict the generation margin and the
    sign of its day-on-day move),
A dark spread (coal) needs API2 coal — not fetched; gas CSS only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data.synthetic import syn_path
from src.eval import metrics as M

EFF_GAS = 0.50          # CCGT efficiency
EF_GAS = 0.35           # tCO2 / MWh_e  (matches the synthetic merit-order)


def clean_spark(price, gas_ttf, co2):
    return price - gas_ttf / EFF_GAS - EF_GAS * co2


def run(synthetic: bool = True) -> pd.DataFrame:
    from src.forecast.train import PRED_PATH

    pred = pd.read_parquet(PRED_PATH)
    raw = pd.read_parquet(syn_path("FR") if synthetic else
                          config.DATA_RAW / f"{config.ZONE.lower()}_entsoe_raw.parquet")
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    gas = raw["gas_ttf_eur_mwh"].reindex(pred.index)
    co2 = raw["co2_eur_t"].reindex(pred.index)

    css_real = clean_spark(pred["actual"], gas, co2)
    css_fc = clean_spark(pred["p50"], gas, co2)

    print("Clean spark spread (gas plant margin), test period\n")
    yr = css_real.groupby(css_real.index.year)
    tab = pd.DataFrame({
        "CSS_mean": yr.mean(),
        "CSS_std": yr.std(),
        "in_merit_%": yr.apply(lambda s: (s > 0).mean() * 100),
    })
    print(tab.round(1).to_string())

    # Forecast skill on the margin itself.
    mae = M.mae(css_real, css_fc)
    # Daily-mean CSS move: does the model call the direction of tomorrow's margin?
    d_real = css_real.resample("1D").mean().diff().dropna()
    d_fc = css_fc.resample("1D").mean().diff().reindex(d_real.index)
    sign_acc = float((np.sign(d_real) == np.sign(d_fc)).mean()) * 100
    print(f"\nCSS forecast MAE        : {mae:.2f} EUR/MWh")
    print(f"CSS day-move sign acc.  : {sign_acc:.1f}%  (coin-flip 50%)")
    print(f"Hours gas in-merit      : {(css_real > 0).mean() * 100:.1f}%")
    return tab


if __name__ == "__main__":
    run()

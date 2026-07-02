"""Pipeline guardrails. The leakage + LP-feasibility tests are the ones a desk
cares about: they prove no look-ahead and a physically valid battery schedule.

Run:  python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data import synthetic
from src.forecast.conformal import apply_delta, conformal_delta
from src.forecast.features import FEATURE_COLS, build_features, xy
from src.trading.battery_lp import _settle, optimize_day

TOL = 1e-5


# --- Data ------------------------------------------------------------------
def test_synthetic_reproducible():
    a = synthetic.generate("FR")
    b = synthetic.generate("FR")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_respects_price_bounds():
    df = synthetic.generate("FR")
    assert df["price_eur_mwh"].min() >= -500 - TOL
    assert df["price_eur_mwh"].max() <= 4000 + TOL


# --- Gate / leakage --------------------------------------------------------
def test_no_feature_is_target_proxy():
    feat = build_features(synthetic.generate("FR"))
    X, y = xy(feat)
    corr = pd.concat([X, y], axis=1).corr()["target"].drop("target").abs()
    assert corr.max() < 0.99, f"possible leak: {corr.idxmax()} corr={corr.max():.3f}"


def test_price_lag_is_truly_past():
    raw = synthetic.generate("FR")
    feat = build_features(raw)
    # price_lag_24 at t must equal the realised price 24h earlier — never the future.
    sample = feat.index[5000]
    assert abs(feat.loc[sample, "price_lag_24"] - raw["price_eur_mwh"].loc[sample - pd.Timedelta(hours=24)]) < TOL
    assert abs(feat.loc[sample, "price_lag_168"] - raw["price_eur_mwh"].loc[sample - pd.Timedelta(hours=168)]) < TOL


def test_all_price_lags_at_least_24h():
    # Anything < 24h would not be cleared at the 12:00 D-1 gate -> illegal.
    from src.forecast.features import PRICE_LAGS

    assert min(PRICE_LAGS) >= 24


# --- Battery LP ------------------------------------------------------------
def test_battery_schedule_is_feasible():
    rng = np.random.default_rng(0)
    batt = config.BATTERY
    prices = rng.normal(60, 40, 24)
    s = optimize_day(prices, batt)
    P, C = batt["power_mw"], batt["capacity_mwh"]
    assert (s["charge"] <= P + TOL).all() and (s["charge"] >= -TOL).all()
    assert (s["discharge"] <= P + TOL).all() and (s["discharge"] >= -TOL).all()
    # SOC stays within [0, C] under the dynamics.
    soc = batt["soc_init_frac"] * C
    for t in range(24):
        soc += batt["eff_charge"] * s["charge"][t] - s["discharge"][t] / batt["eff_discharge"]
        assert -TOL <= soc <= C + TOL
    # Throughput (cycle) cap respected.
    assert s["discharge"].sum() <= batt["max_cycles_per_day"] * C + 1e-3


def test_battery_buys_low_sells_high():
    batt = config.BATTERY
    prices = np.array([10.0] * 12 + [200.0] * 12)   # cheap then expensive
    s = optimize_day(prices, batt)
    assert s["charge"][:12].sum() > s["charge"][12:].sum()      # charge in cheap window
    assert s["discharge"][12:].sum() > s["discharge"][:12].sum()  # discharge in pricey window


def test_perfect_foresight_beats_forecast():
    # On the same realised prices, perfect-foresight scheduling cannot earn less
    # than scheduling on a noisy forecast settled at realised.
    rng = np.random.default_rng(1)
    batt = config.BATTERY
    cost = batt["cost_per_mwh_cycle"]
    actual = rng.normal(60, 40, 24)
    fcast = actual + rng.normal(0, 25, 24)
    pf = _settle(optimize_day(actual, batt), actual, cost)
    fc = _settle(optimize_day(fcast, batt), actual, cost)
    assert pf >= fc - TOL


# --- Conformal -------------------------------------------------------------
def test_conformal_restores_coverage():
    rng = np.random.default_rng(2)
    y_cal = rng.normal(0, 1, 4000)
    lo_cal, hi_cal = np.full(4000, -0.4), np.full(4000, 0.4)   # deliberately too narrow
    q = conformal_delta(y_cal, lo_cal, hi_cal, alpha=0.2)
    y_te = rng.normal(0, 1, 4000)
    lo, hi = apply_delta(np.full(4000, -0.4), np.full(4000, 0.4), q)
    cover = np.mean((y_te >= lo) & (y_te <= hi))
    assert cover >= 0.78, f"coverage {cover:.3f} below 80% target band"

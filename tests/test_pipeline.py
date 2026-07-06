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
from src.data.entsoe_pull import RAW_PATH
from src.eval.metrics import mae
from src.forecast import baselines
from src.forecast.conformal import apply_delta, conformal_delta
from src.forecast.features import FEATURE_COLS, build_features, load_raw, xy
from src.forecast.lgbm_quantile import QuantileLGBM
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


@pytest.mark.skipif(not RAW_PATH.exists(), reason="real ENTSO-E parquet not present")
def test_no_leak_on_real_data():
    # The interview claim ("no post-clearing leak at the 12:00 gate") must hold on
    # the REAL pull, not just synthetic. Runs whenever data/raw/*_entsoe_raw.parquet
    # exists; skipped in a clean checkout.
    feat = build_features(load_raw(synthetic=False))
    X, y = xy(feat)
    corr = pd.concat([X, y], axis=1).corr()["target"].drop("target").abs()
    assert corr.max() < 0.99, f"possible leak on real data: {corr.idxmax()} corr={corr.max():.3f}"


def test_holiday_calendar_matches_zone():
    # Regression guard: features must use the zone's OWN holiday calendar. A DE
    # model trained on FR holidays (the bug) corrupts the load-dip feature.
    de = build_features(synthetic.generate("DE"), zone="DE")
    fr = build_features(synthetic.generate("FR"), zone="FR")

    def flag_on(feat, month, day):
        local = feat.index.tz_convert(config.ZONE_TZ)
        return feat.loc[(local.month == month) & (local.day == day), "is_holiday"]

    # 3 Oct = German Unity Day (DE holiday, FR working day).
    assert (flag_on(de, 10, 3) == 1).all() and (flag_on(fr, 10, 3) == 0).all()
    # 14 Jul = Bastille Day (FR holiday, DE working day).
    assert (flag_on(fr, 7, 14) == 1).all() and (flag_on(de, 7, 14) == 0).all()


def test_feature_grain_is_contiguous_hourly():
    # After the warm-up dropna, the table must be a clean 1h grid — no gaps, no
    # duplicated hours (a DST fold silently dropping/duplicating an hour = bug).
    feat = build_features(synthetic.generate("FR"))
    diffs = feat.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all(), "non-contiguous hourly grain"


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


# --- Model (train-backed guardrails; a few seconds each) -------------------
def _fr_split():
    feat = build_features(synthetic.generate("FR"))
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    return feat[feat.index < ts], feat[feat.index >= ts], cols


def test_forecast_is_deterministic():
    # Same seed + config -> identical predictions. Guards the random_state pin.
    tr, te, cols = _fr_split()
    Xtr, ytr = xy(tr, cols)
    Xte = xy(te, cols)[0]
    a = QuantileLGBM(quantiles=[0.5]).fit(Xtr, ytr).predict(Xte)["p50"].to_numpy()
    b = QuantileLGBM(quantiles=[0.5]).fit(Xtr, ytr).predict(Xte)["p50"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_forecast_beats_naive_d7():
    # Baseline-regression guard: the P50 model must beat the cheap D-7 baseline on
    # MAE, else the features earn nothing and something silently regressed.
    tr, te, cols = _fr_split()
    p50 = QuantileLGBM(quantiles=[0.5]).fit(*xy(tr, cols)).predict(xy(te, cols)[0])["p50"]
    assert mae(te["target"], p50) < mae(te["target"], baselines.naive_d7(te))

"""Fit forecasters, compare on a held-out test year, save predictions.

Two protocols:
  - single fit (default): one fit on the train block, predict all of test.
  - walk-forward (--walkforward): expanding-window monthly recalibration — at each
    month of the test year, refit on everything known up to that month and predict
    it. This is the realistic desk cadence (a model is retrained as data arrives)
    and the honest way to report out-of-sample skill.

Both wrap the LightGBM quantiles in Conformalized Quantile Regression: the tail of
each training block is held out as a calibration set to widen the P10-P90 band to
its nominal 80% coverage.

Run:  python -m src.forecast.train                  # single fit, synthetic
      python -m src.forecast.train --walkforward     # monthly recalibration
      python -m src.forecast.train --real            # real ENTSO-E parquet
"""
from __future__ import annotations

import sys

import pandas as pd

import config
from src.eval import metrics as M
from src.forecast import baselines
from src.forecast.conformal import apply_delta, conformal_delta
from src.forecast.features import FEATURE_COLS, build_features, load_raw, xy
from src.forecast.lear import LEAR
from src.forecast.lgbm_quantile import QuantileLGBM

TEST_START = "2024-01-01"
CAL_DAYS = 90              # calibration tail held out of each training block
ALPHA = 0.2               # 1-alpha = 80% target coverage for P10-P90
PRED_PATH = config.DATA_PROCESSED / "predictions.parquet"


def _fit_block(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str]):
    """Fit LEAR + conformalised LightGBM on `train`, predict `pred`."""
    cal_start = train.index.max() - pd.Timedelta(days=CAL_DAYS)
    proper = train[train.index < cal_start]
    cal = train[train.index >= cal_start]

    lear = LEAR().fit(proper, cols)
    Xp, yp = xy(proper, cols)
    from src.forecast.tune import load_tuned

    q = QuantileLGBM(params=load_tuned()).fit(Xp, yp)

    qc = q.predict(xy(cal, cols)[0])
    delta = conformal_delta(cal["target"], qc["p10"], qc["p90"], ALPHA)

    qt = q.predict(xy(pred, cols)[0])
    p10c, p90c = apply_delta(qt["p10"], qt["p90"], delta)

    out = pd.DataFrame(index=pred.index)
    out["lear"] = lear.predict(pred)
    out[["p10", "p50", "p90"]] = qt[["p10", "p50", "p90"]]
    out["p10_cal"] = p10c
    out["p90_cal"] = p90c
    return out, q, delta


def run(synthetic: bool = True, walkforward: bool = False) -> pd.DataFrame:
    raw = load_raw(synthetic=synthetic)
    feat = build_features(raw)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    print(f"Features ({len(cols)}): {cols}")

    test_start = pd.Timestamp(TEST_START, tz="UTC")
    train, test = feat[feat.index < test_start], feat[feat.index >= test_start]
    print(f"Train {train.index.min().date()}..{train.index.max().date()} ({len(train):,} h) | "
          f"Test {test.index.min().date()}..{test.index.max().date()} ({len(test):,} h)")

    out = pd.DataFrame(index=test.index)
    out["actual"] = test["target"]
    out["ref_d1"] = test["price_lag_24"]
    out["naive_d7"] = baselines.naive_d7(test)
    out["naive_d1"] = baselines.naive_d1(test)

    last_q = None
    if walkforward:
        folds = pd.date_range(test_start, test.index.max(), freq="MS")
        print(f"Walk-forward: {len(folds)} monthly refits (expanding window) ...")
        parts = []
        for f0 in folds:
            f1 = f0 + pd.offsets.MonthBegin(1)
            fold = test[(test.index >= f0) & (test.index < f1)]
            if fold.empty:
                continue
            tr = feat[feat.index < f0]
            block, last_q, _ = _fit_block(tr, fold, cols)
            parts.append(block)
            print(f"  {f0.date()}  train<{f0.date()} ({len(tr):,} h) -> pred {len(fold)} h")
        preds = pd.concat(parts)
    else:
        print("Single fit + conformal ...")
        preds, last_q, delta = _fit_block(train, test, cols)
        print(f"  conformal delta = {delta:.1f} EUR/MWh")

    out = out.join(preds)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_PATH)
    print(f"Saved predictions -> {PRED_PATH}")
    _report(out, last_q)
    return out


def _report(out: pd.DataFrame, qlgbm):
    y = out["actual"]
    ref = out["ref_d1"]
    spike_thr = float(y.quantile(0.95))

    print("\n=== Point forecast ===")
    rows = []
    for name in ["naive_d7", "lear", "p50"]:
        p = out[name]
        rows.append({"model": name, **M.point_report(y, p, ref),
                     "spikeRecall": M.spike_recall(y, p, spike_thr),
                     "negRecall": M.negative_recall(y, p)})
    print(pd.DataFrame(rows).set_index("model").round(3).to_string())

    print("\n=== Probabilistic ===")
    qp = {0.1: out["p10"], 0.5: out["p50"], 0.9: out["p90"]}
    print(f"Mean pinball (raw)     : {M.mean_pinball(y, {k: v.to_numpy() for k, v in qp.items()}):.3f}")
    print(f"P10-P90 coverage RAW   : {M.coverage(y, out['p10'], out['p90']) * 100:.1f}%  (target 80%)")
    print(f"P10-P90 coverage CONF. : {M.coverage(y, out['p10_cal'], out['p90_cal']) * 100:.1f}%  <- conformal")

    if qlgbm is not None:
        print("\n=== Top features (last P50 model) ===")
        print(qlgbm.feature_importance().head(8).to_string())


if __name__ == "__main__":
    run(synthetic="--real" not in sys.argv, walkforward="--walkforward" in sys.argv)

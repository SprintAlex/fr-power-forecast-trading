"""Optuna hyperparameter tuning for the LightGBM quantile forecaster.

Carries over the school project's tuning discipline (TPE search, time-ordered
validation — never shuffle a time series). Tunes on the P50 pinball loss over a
held-out tail of the training block; the best params are reused for all three
quantile models. Saved to results/lgbm_params.json and auto-loaded by train.py.

Run:  python -m src.forecast.tune              # 40 trials, synthetic
      python -m src.forecast.tune 80 --real
"""
from __future__ import annotations

import json
import sys

import optuna
import pandas as pd
from lightgbm import LGBMRegressor

import config
from src.eval.metrics import pinball_loss
from src.forecast.features import FEATURE_COLS, build_features, load_raw, xy
from src.forecast.train import TEST_START

PARAM_PATH = config.RESULTS / "lgbm_params.json"
VAL_DAYS = 120


def load_tuned() -> dict | None:
    if PARAM_PATH.exists():
        return json.loads(PARAM_PATH.read_text())
    return None


def tune(synthetic: bool = True, n_trials: int = 40) -> dict:
    raw = load_raw(synthetic=synthetic)
    feat = build_features(raw)
    cols = [c for c in FEATURE_COLS if c in feat.columns]
    train = feat[feat.index < pd.Timestamp(TEST_START, tz="UTC")]
    cut = train.index.max() - pd.Timedelta(days=VAL_DAYS)
    fit, val = train[train.index < cut], train[train.index >= cut]
    Xf, yf = xy(fit, cols)
    Xv, yv = xy(val, cols)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            subsample_freq=1, n_jobs=-1, verbose=-1,
        )
        m = LGBMRegressor(objective="quantile", alpha=0.5, **params)
        m.fit(Xf, yf)
        return pinball_loss(yv, m.predict(Xv), 0.5)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**study.best_params, "subsample_freq": 1, "n_jobs": -1, "verbose": -1}
    PARAM_PATH.write_text(json.dumps(best, indent=2))
    print(f"Best P50 pinball (val): {study.best_value:.3f}  ({n_trials} trials)")
    print(f"Saved tuned params -> {PARAM_PATH}")
    print(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nt = int(args[0]) if args else 40
    tune(synthetic="--real" not in sys.argv, n_trials=nt)

# Battery arbitrage on the FR day-ahead power market

Probabilistic day-ahead price forecast → battery arbitrage LP → P&L backtest with
trader risk metrics. Portfolio piece for energy trading / quant interviews
(TotalEnergies, EDF Trading, Engie GEM, …). Full brief in [HANDOFF.md](HANDOFF.md).

## Locked decisions
| Choice | Pick | Why |
|---|---|---|
| Zone | **FR** | Desk-relevant (Total/EDF), deep liquid data, strong FR↔DE spread |
| Hero strategy | **Battery arbitrage (LP)** | Real asset-optimization desks run daily |
| Forecast | **LEAR + LightGBM quantile** | EPF standard baseline + P10/P50/P90 tails |

## Pipeline
1. **Data** — ENTSO-E: DA price + DA load/wind/solar forecasts, UTC hourly, DST-safe. + gas TTF / CO2 EUA.
2. **Forecast** — naive D-7 + LEAR baselines; LightGBM quantile (α 0.1/0.5/0.9) + **conformal** calibration;
   strict 12:00 D-1 gate; **walk-forward** monthly recalibration.
3. **Trading** — (a) battery LP (`cvxpy`), decisions on forecast settled at realized; (b) FR-DE location spread.
4. **Eval** — P&L, % perfect-foresight captured, Sharpe, VaR/drawdown, spread-sign accuracy, pinball, coverage,
   spike/neg-price recall.

## Results (real ENTSO-E, FR/DE 2021-2024, test = 2024)
| Strategy | metric | model | naive | perfect |
|---|---|---|---|---|
| Battery arbitrage | % perfect captured | **83%** | 78% | 100% |
| FR-DE spread | sign accuracy | **72%** | 55% | 100% |
| | % perfect captured | **91%** | — | 100% |

Forecast (FR 2024 out-of-sample): P50 MAE **15.7 €/MWh**, R² **0.72–0.75**, DirAcc **0.73**, spike-recall
**0.77**; P10-P90 coverage **≈78%** (conformal, from ~55% raw). Top features: gas TTF momentum, wind, CO2.

**Crisis stress test (real 2022):** trained only on calm 2021, the model faces the real gas crisis (price
mean **276 €/MWh**) and breaks — R² **−0.53**, MAE 127 — then recovers once the crisis is in training
(2023 R² 0.79). The quantified case for a trader overriding the model out-of-regime.

A 1 MW price-taker inflates absolute Sharpe (no market impact) — read **% captured / sign accuracy**.
Talking points: [docs/interview_notes.md](docs/interview_notes.md). Full walk-through:
[notebooks/power_trading_quant.ipynb](notebooks/power_trading_quant.ipynb).

## Run
```bash
python run_all.py            # build data -> forecast -> battery + spread verdicts -> dashboard
python run_all.py --single   # single fit instead of walk-forward
python run_all.py --real     # use real ENTSO-E parquet (after token + pull)
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your ENTSO-E API token
python -m src.data.entsoe_pull   # pulls config window -> data/raw/fr_entsoe_raw.parquet
```

## Layout
```
config.py            zone, window, battery specs, quantiles, costs, paths
run_all.py           one-command end-to-end pipeline
notebooks/           power_trading_quant.ipynb  <- the narrative walk-through (start here)
src/data/            entsoe_pull (real) + synthetic (FR+DE) + fundamentals (gas/CO2 via yfinance)
src/forecast/        features (gate-legal), baselines, lear, lgbm_quantile, conformal, tune, scenarios, train
src/trading/         battery_lp (point/robust/CVaR/reserve), spread_strategy, spark_spread, value_stacking, intraday
src/eval/            metrics, trader_metrics, dm_test, strategy_comparison, risk_strategies, regime_test, book_risk, robustness, dashboard
tests/               pytest guardrails (leakage gate, LP feasibility, conformal)
docs/                interview_notes.md
data/raw|processed/  parquet (gitignored)
results/             figures, metrics
```

The notebook [notebooks/power_trading_quant.ipynb](notebooks/power_trading_quant.ipynb) is the recommended
entry point — it tells the whole story (leakage → forecast → 2022 crisis → strategies → risk) with figures.

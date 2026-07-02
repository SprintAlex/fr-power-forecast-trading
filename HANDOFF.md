# Handoff — Personal project: trader-style power price forecasting & trading

> Paste this into a fresh conversation to start building. Self-contained brief.

## 0. One-line goal
Build an **end-to-end power-trading project** (probabilistic day-ahead price forecast → trading
strategy → P&L backtest with risk metrics) as a **portfolio piece for energy trading / quant
interviews** (TotalEnergies trading desk, Engie GEM, EDF Trading, Vitol, Statkraft, etc.).

## 1. Who I am
- Maurel — AI engineering student @ Telecom SudParis.
- Already built a **school project** (SIC7002): day-ahead Danish electricity spot price forecast.
  - Honest day-ahead XGBoost: **MAE 23.26 / RMSE 37.78 / R² 0.626 / DirAcc 76.0%**. LSTM 25.32. Naive 36.65.
  - Methodology: EDS + DMI data, DST-safe UTC pipeline, TimeSeriesSplit CV, Optuna TPE (80 trials),
    pseudo-Huber loss, SHAP interpretability.
  - **Key win — leakage discipline**: a teammate's model scored R²=0.97 / MAE 1.41 by leaking
    post-spot variables (balancing/imbalance prices realized AFTER market clearing, e.g.
    `BalancingPowerPriceUpEUR` = spot price in 73% of rows) plus T-1 lags illegal at day-ahead gate.
    Removing the leak on his own pipeline collapsed MAE 1.69 → 25.55 (×15). I understand the
    difference between a legal feature at gate closure and a leak. **This is a strong interview hook.**
- Now I want a **personal project** that mirrors what real desks actually do, not an academic MAE chase.

## 2. What real energy traders / quants actually do (the realism target)

**Market timeline — the price forms in stages, each is a tradable market:**
| Market | Horizon | Mechanism |
|---|---|---|
| Forward / Futures | months–years | OTC + EEX, hedging |
| **Day-Ahead (spot)** | D-1, gate 12:00 | hourly auction, 1 clearing price/hour (Nord Pool, EPEX) |
| Intraday (XBID) | D → delivery-5min | continuous order book |
| Balancing / mFRR | real time | TSO, imbalance penalties |

**Two schools:**
- **Fundamental**: don't predict price directly — predict drivers, rebuild price via the **merit-order
  curve**. Reine variable = **residual load** (demand − wind − solar). Fuels set the top of the stack:
  **gas TTF, coal API2, CO2 (EUA)**. Plus hydro levels, plant outages (REMIT), interconnector flows.
  "Windy Denmark" thesis (Mauritzen): wind pushes price down the stack.
- **Statistical / ML — EPF (Electricity Price Forecasting)**: reference benchmark **Lago et al. 2021**.
  Standard models = **LEAR** (LASSO auto-regressive) and **DNN**. All 24 hours predicted at once
  (multivariate), not recursive. Legal features at 12:00 gate: lagged prices (D-1, D-2, **D-7**),
  TSO load/wind/solar forecasts, fuels, CO2, calendar.

**What they really forecast:** residual load before price; **spreads not levels** (DA↔ID, peak↔off-peak,
location DK1↔DK2↔DE, calendar); **probabilistic** (quantiles P10/P50/P90, not a point), evaluated with
**pinball loss / CRPS**; weather from **ensembles (ECMWF 51 members)**, not a single forecast.

**Forecast → money:** cross-market arbitrage (forecast DA > auction → buy DA, sell ID/balancing);
**asset optimization** (battery charge low / discharge spikes, hydro, demand-response) ← the real cash;
spread trading (bet on sign(DK1−DE)).

**Evaluation that matters (≠ MAE):** P&L backtest, Sharpe, **sign accuracy of the spread** (hit rate),
behavior on **spikes & negative prices** (the tails hold all the risk and P&L). A worse-MAE model with
better spread-sign accuracy makes more money.

## 3. The project to build (recommended spec)

**Working title:** "Battery arbitrage on the day-ahead power market: probabilistic forecast + P&L backtest"

**Why this one:** battery/asset arbitrage is exactly an asset-optimization problem desks run daily; it
forces a probabilistic forecast, a trading layer, and P&L/risk metrics — the full trader stack.

**Pipeline (MVP):**
1. **Data** — ENTSO-E Transparency API (free, use `entsoe-py`): day-ahead prices + load forecast +
   wind/solar forecast for one bidding zone. Add fuels: **gas TTF + CO2 EUA** (the biggest fundamental
   gap vs a real desk). Weather optional (ensemble = stretch).
2. **Forecast** — probabilistic day-ahead, 24h multi-output:
   - Baseline: **LEAR** (EPF standard, cheap credibility) + **naive D-7**.
   - Main: **quantile gradient boosting** (LightGBM/XGBoost `quantile`, alpha 0.1/0.5/0.9) → P10/P50/P90.
   - Strict gate discipline: only features known at 12:00 D-1 (carry over the leakage lesson).
3. **Trading layer** — **battery arbitrage optimizer** (LP via `cvxpy` or `PuLP`): given forecast prices
   + battery specs (power MW, capacity MWh, round-trip efficiency, cycle limit), optimize charge/discharge.
   Settle at **realized** DA prices. Benchmark vs **perfect-foresight** upper bound (= % captured).
4. **Evaluation (trader metrics):** P&L (€/MWh capacity/yr), **% of perfect foresight captured**, Sharpe,
   daily-P&L VaR / worst day / drawdown, spread-sign accuracy, pinball loss, spike & negative-price recall.

**Stretch:** location spread model (DK1−DK2 or country-country); fundamental residual-load features;
LEAR vs LightGBM vs naive comparison; intraday spread layer; ensemble weather.

## 4. Decisions to make at the start of the new conversation
- **Zone**: DE-LU or FR (most relevant to TotalEnergies/Engie/EDF + deepest, most liquid data) vs DK
  (continuity with existing project). → recommend **DE-LU or FR** for the interview audience.
- **Hero strategy**: battery arbitrage (recommended) vs spread trading vs pure cross-market arbitrage.
- **Forecast model**: start with LEAR + quantile LightGBM (recommended) vs go straight to a DNN.

## 5. Tech stack
Python · pandas/numpy · `entsoe-py` (ENTSO-E API, needs free API token) · LightGBM or XGBoost (quantile)
· `cvxpy` or `PuLP` (battery LP) · `epftoolbox` (has LEAR + Lago benchmarks) · matplotlib/plotly.

## 6. Interview framing (why this lands)
- Demonstrates **market microstructure** understanding (timeline + which features are legal at gate closure).
- **Leakage awareness** — most candidates fail this; I have a concrete war story.
- Thinks in **P&L and risk**, not just ML metrics.
- **Battery arbitrage** = real asset-optimization desks care about.
- **Probabilistic forecasting** = what actual EPF desks do.

## 7. Reusable assets from the old project
- Day-ahead pipeline patterns at `~/Claude/Projects/Cours/SIC/projet-elec-dk` (DST-safe UTC, TimeSeriesSplit,
  Optuna, SHAP, EDS ingest).
- Leakage demo (Matisse repo `TeebooGH/dk-power-spot-forecaster`, was cloned to /tmp, may be wiped).
- The leakage-discipline narrative as a talking point.

## 8. Suggested first steps in the new conversation
1. Lock zone + hero strategy + model (section 4).
2. Get ENTSO-E API token, pull prices + load/wind/solar forecasts.
3. Baselines: naive D-7 + LEAR.
4. Quantile LightGBM forecast (P10/P50/P90).
5. Battery LP backtester + perfect-foresight benchmark.
6. Trader-metrics dashboard (P&L, Sharpe, VaR, pinball, spike recall).

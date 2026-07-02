# Interview notes — talking points & results

Portfolio piece for energy trading / quant desks (TotalEnergies, EDF Trading, Engie GEM…).
Numbers below are on **real ENTSO-E FR/DE day-ahead data (2021-2024)** + real gas TTF / CO2 EUA;
a 1 MW price-taker inflates absolute Sharpe (no market impact), so the **% captured and sign accuracy
carry the signal**.

## The 30-second pitch
Built the full trader stack on the FR day-ahead market: **probabilistic price forecast →
trading layer → P&L backtest with risk metrics**. Two desk strategies — **battery arbitrage**
(asset optimisation) and **FR-DE location spread** (relative value). The point isn't MAE; it's
whether the forecast *makes money* versus a cheap benchmark, net of cost.

## Five things that land

### 1. Leakage discipline (the war story)
Day-ahead gate closes 12:00 D-1. A feature is legal only if known by then: price lags ≥24h,
TSO load/wind/solar *forecasts*, gas/CO2 forwards, calendar. Realised outturn and balancing
prices are leaks. In a school project a teammate scored R²=0.97 by leaking post-clearing
variables; removing the leak collapsed his MAE ×15 (1.7 → 25.5). My feature builder enforces
the gate and ships a leakage self-check (no feature is a >0.99 proxy of the target).

### 2. MAE ≠ money
The verdict is P&L, not RMSE. On the battery book the model's **edge vs naive D-7 is +18k€/yr**
while the naive forecast actually *loses* money (it pays the cycling cost without good timing).
Before adding gas/CO2, LightGBM and LEAR tied on P&L despite LightGBM's better MAE — a concrete
example that lower error ≠ more cash.

### 3. The forecast's trading grade = % of perfect foresight
Every strategy is benchmarked against a perfect-foresight upper bound (real 2024).
- Battery arbitrage: **~83%** captured — but the edge over naive is small in a *calm* year (2024);
  the model's real value shows in volatile regimes.
- FR-DE spread: **~91%** captured at **72% sign accuracy** (vs 55% naive) — spreads are more
  forecastable than levels (the shared gas/CO2 cancels, leaving the RES differential).

### 4. Gas + CO2 are the lever that pays
Gas TTF momentum, wind, and CO2 are the **top features** on real FR data — the model reconstructs the
merit order rather than curve-fitting price. This is the fundamental layer real desks live on.
On real 2024, LightGBM (R² 0.72, MAE 15.7) clears LEAR (0.64) and naive D-7 (0.17).

### 5. Probabilistic, calibrated, honestly recalibrated
- **Quantiles P10/P50/P90** (LightGBM quantile), not a point — a trader sizes on the
  distribution, not the mean.
- Raw quantiles under-cover (~55% vs 80% on real data). **Conformalized Quantile Regression** repairs
  it to **~78%** with a finite-sample guarantee — intervals a trader can actually size on.
- **Walk-forward monthly recalibration** (expanding window) = the realistic out-of-sample
  protocol, not one optimistic fit.

## "Is it profitable, like on a desk?"
Yes, net of degradation + execution cost (real 2024): battery captures **83%** of perfect foresight;
the **FR-DE spread book** is the standout — **72% sign accuracy** (vs 55% naive) and **91%** of perfect
captured, +193k€ on a 1 MW position. Honest nuance: in a calm year the battery's edge over naive is
small — the model earns its keep most in **volatile / crisis** regimes (see below).

## What a real desk adds beyond the model (know your role)
Model forecasts; trader makes money. Sizing on the distribution, real-time info the model lacks
(outages, intraday gas, news), knowing when the model is out-of-regime (2022 crisis) and to
override it, execution/microstructure, and book-level risk. The model is a commodity; the edge
is in the use.

## Crisis stress test (real 2022) — the headline honesty
Trained only on calm 2021, the model meets the real gas crisis (2024 test price mean 58 €/MWh vs 2022's
**276**): R² goes **−0.53**, MAE 127 — it *breaks* out-of-distribution. It still captures 63% (crisis =
huge spreads) and recovers once 2022 is in training (2023 R² 0.79). This is the quantified reason a
trader must recognise a broken regime and override the model.

## Honest limitations (say them first)
- A 1 MW price-taker inflates absolute Sharpe — no market impact modelled at that size.
- Battery edge over naive is small in calm years; the value is in volatile regimes.
- DE spread forecast is a single-fit P50; a full walk-forward + hedged legs would tighten it.
- Next: order-book execution, weather-ensemble uncertainty, a live daily pipeline at the 12:00 gate.

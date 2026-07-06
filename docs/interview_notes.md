# Notes — what this is and why I built it

I'm an AI-engineering student at Télécom SudParis, prepping for energy-trading and quant
interviews. I'd already done a school project forecasting Danish day-ahead prices, and it
bothered me that it was, in the end, an academic MAE contest. A desk doesn't get paid for a low
RMSE — it gets paid for P&L. So I rebuilt the thing the way I think a desk actually works:
forecast the price *with* its uncertainty, turn that into a trade, and judge it on money and risk.
This runs on real ENTSO-E data for France and Germany, 2021–2024, plus real gas TTF and CO2.

One caveat I'd rather say myself than have someone catch: this is a 1 MW price-taker with no
market impact, so the absolute Sharpe is not meaningful. I look at **% of perfect-foresight
captured** and **spread sign accuracy** instead — those survive the toy-size assumption.

## The decisions I made, and why

- **France, not DE-LU or Denmark.** DK was my school zone and would have been the easy continuation,
  but FR is where the desks I'm targeting actually sit (Total, EDF), and the FR↔DE spread gives me a
  second, more forecastable thing to trade. I kept DK's DST-safe pipeline habits.
- **Battery arbitrage as the main strategy.** It forces the whole stack — you can't fake it with a
  point forecast. You need the distribution, an optimiser, and a settlement that punishes forecast
  error. It's also a real asset-optimisation problem desks run daily.
- **LEAR + LightGBM quantile, not a DNN.** LEAR is the EPF reference (Lago et al.) and gives me a
  cheap, credible baseline; LightGBM quantile gets me P10/P50/P90 without a training circus. I wanted
  something I could defend line by line, not a black box.

## What I actually learned

**Leakage is the thing most candidates get wrong, and I have a concrete story.** On the school
project a teammate hit R²=0.97 — he was leaking post-clearing variables (balancing prices that only
exist *after* the market clears). Removing the leak took his MAE from ~1.7 to ~25. It stuck with me,
so here the feature builder enforces the 12:00 D-1 gate and I ship a self-check that no feature is a
near-perfect proxy of the target. On real data the top features end up being gas-TTF momentum, wind,
and CO2 — the model is rebuilding the merit order, which is roughly what a fundamental trader does.

**A better forecast isn't automatically more money.** Early on, before I added gas and CO2, LightGBM
had a lower MAE than LEAR but made about the same P&L. That surprised me and it's the point: I now
benchmark every strategy against a perfect-foresight ceiling and report % captured, because that's
the number that decides if the model is worth running.

**The model breaks in a crisis — and I think that's the most honest result here.** Asked to forecast
2022 (gas spiking, prices averaging ~276 €/MWh vs ~58 in 2024) with only ~2 months of legal pre-crisis
history — real EUA carbon data only starts Oct-2021 — forecast R² collapses to ~0.14 and MAE blows up
to ~86. Yet the battery still captures ~78% of the perfect-foresight P&L because the spreads are
enormous, and skill recovers as 2022 enters the training window (2023 R²≈0.46). I like that this is
visible rather than hidden — it's the quantified reason a trader has to recognise an out-of-regime
moment and read P&L capture, not R².

## Numbers (real 2024 test, walk-forward monthly)

- Forecast: LightGBM P50 MAE ~25.8 €/MWh, R²≈0.37, beats LEAR (0.32) and naive D-7 (0.17). The edge over
  naive on raw error is modest — the point of the project is that trading P&L, not MAE, is the verdict.
- Quantiles under-cover raw (~45% for a nominal 80% band); conformal calibration brings it to ~73%
  with a finite-sample guarantee. Still short of nominal on 2024 — a real 2023→2024 distribution shift
  that violates exchangeability, worth flagging honestly rather than claiming 80%.
- Battery: captures ~83% of perfect foresight. Honest nuance — in a calm year its edge over the naive
  forecast is small; the value shows up in volatile regimes.
- FR-DE spread: 72% sign accuracy (vs 55% for a lagged-spread naive), ~91% of perfect captured. On a
  spread book the *sign* is what pays, not the MAE — this is the result I'm happiest with.

## What I know is weak

- 1 MW price-taker: no market impact, so absolute Sharpe/€ figures are optimistic.
- The DE-side spread forecast is a single fit, not the full walk-forward I use on FR.
- Capture is capped by forecast error; closing the gap needs intraday signal and richer fundamentals.
- I haven't wired weather ensembles — real desks drive price uncertainty from ECMWF members, not from
  a post-hoc quantile model.

## What I'd do next

A live daily pipeline that pulls at the gate and retrains, an order-book execution model so the P&L
is defensible at real size, and weather-ensemble-driven uncertainty. And I'd want to see the 2022
behaviour on the intraday market, not just day-ahead.

## Know your role: what a trader adds that my model doesn't

The model forecasts; the trader makes the money. Sizing on the distribution rather than the point,
the real-time information the model never sees (outages, an intraday gas move, news), the judgement
to override the model when the regime breaks, execution, and the book-level risk view. The model is
close to a commodity — most of the edge is in how it's used.

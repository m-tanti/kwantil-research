# Tail-risk backtest, and the null sensitivity

Walk-forward evaluation of 5 VaR and expected-shortfall forecasters on the same
3-asset book over 2,172 graded trading days: parametric EWMA, historical
simulation, GARCH with a t-copula, an adaptive conformal-PID calibrator, and a
joint VaR/ES variant.

Article: [Tail Risk Backtesting: When the Test Improves and the Model Does Not](https://kwantil.com/papers/did-the-model-get-better/)
Live scoreboard and full methodology: [var-backtest.mtanti.com](https://var-backtest.mtanti.com)

## What it found

The headline result is about the test, not the models.

An expected-shortfall diagnostic improved after three simultaneous changes, and
the development log credited the model change. A factorial separating them shows
the reference distribution moved the acceptance threshold by −0.104 while the
model change moved the statistic by +0.0096: **the test did roughly 11 times the
work of the model.** Against the original null, the improved model still fails.

Re-scoring all 5 methods at 2 confidence levels against 6 nulls, with the models,
the data, the VaR and the ES all held fixed, moves the count from **0 green under
a Gaussian null to 10 green under a df-5 null**.

Two consequences follow. Because the null is calibrated from each model's own
claimed kurtosis, claiming fatter tails buys a more permissive threshold. And
under a distribution-free null nothing is rejected at all, including a baseline
that breaches at twice its stated rate, which puts a hard limit on what the test
can detect at 24 to 79 exceedances.

## Data

SPY, TLT, GLD daily closes via `yfinance`. No key required.

The price cache is **not** redistributed; Yahoo Finance terms do not permit it.
`fetch_prices` rebuilds it on first run. `output/details.csv` is row-level and
derived from those prices, so it is rebuilt rather than shipped.

Shipped: `output/summary.csv` and the null-sensitivity results under
`experiments/factorial/`.

## Running

`pytest -q` runs 31 tests in about 26 seconds. The full pipeline and the
factorial are in
[REPRODUCING.md](../../REPRODUCING.md#2-tail-risk-backtest-and-the-null-sensitivity).

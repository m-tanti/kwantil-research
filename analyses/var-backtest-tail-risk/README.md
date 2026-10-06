# Tail-risk backtest, and the null sensitivity

Walk-forward evaluation of 5 VaR and expected-shortfall forecasters on the same
3-asset book over 2,172 graded trading days: parametric EWMA, historical
simulation, GARCH with a t-copula, an adaptive conformal-PID calibrator, and a
joint VaR/ES variant.

Article: [Tail Risk Backtesting: When the Test Improves and the Model Does Not](https://kwantil.com/papers/did-the-model-get-better/)
Scoreboard and full methodology: [kwantil.com/var-backtest](https://kwantil.com/var-backtest/) (a snapshot of the 2026-08-24 run)

## What it found

The headline result is about the test, not the models.

An expected-shortfall diagnostic improved after three simultaneous changes, and
the development log credited the model change. A factorial separating them shows
the reference distribution moved the acceptance threshold by −0.104 while the
model change moved the statistic by +0.0096: **the test did roughly 11 times the
work of the model.** Against the original null, the improved model still fails.

The statistic is the conditional Acerbi-Székely Z1 (the mean shortfall ratio
over breach days). Re-scoring all 5 methods at 2 confidence levels against 6
nulls, with the models, the data, the VaR and the ES all held fixed, moves Z1
from **0 of 10 cells green under a Gaussian null to 10 of 10 under a df-5 null**.

Re-run with the unconditional Z2 on the same simulated paths, the threshold
moves as much but the verdicts do not: the same 6 of 10 cells pass under every
parametric null, and the 4 that fail are the ones the exceedance-count test
rejects.

Three consequences follow. Because the null is calibrated from each model's own
claimed kurtosis, claiming fatter tails buys a more permissive Z1 threshold. Z1
cannot see an excess number of breaches by construction, so it must be read
with a count test; the count test did catch the baseline that breaches at twice
its stated rate. And with 24 to 79 exceedances, neither statistic, scored
against a simulated null, caught a model with the right breach rate and a thin
tail.

Up to package 0.1.x the code called the conditional statistic Z2 and the
unconditional one Z1. The names now follow Acerbi and Székely (2014); see
[CHANGELOG.md](CHANGELOG.md).

## Data

SPY, TLT, GLD daily closes via `yfinance`. No key required.

The price cache is **not** redistributed; Yahoo Finance terms do not permit it.
`fetch_prices` rebuilds it on first run. `output/details.csv` is row-level and
derived from those prices, so it is rebuilt rather than shipped.

Shipped: `output/summary.csv` and the null-sensitivity results under
`experiments/factorial/`.

## Running

`pytest -q` runs 33 tests in about 26 seconds. The full pipeline and the
factorial are in
[REPRODUCING.md](../../REPRODUCING.md#2-tail-risk-backtest-and-the-null-sensitivity).

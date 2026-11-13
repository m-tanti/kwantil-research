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
its stated rate. And with 17 to 79 exceedances, neither statistic, scored
against a simulated null, caught a model with the right breach rate and a thin
tail.

Up to package 0.1.x the code called the conditional statistic Z2 and the
unconditional one Z1. The names now follow Acerbi and Székely (2014); see
[CHANGELOG.md](CHANGELOG.md).

## The specimen Model Health Check

`scripts/audit_health_check.py` turns the same run into the deliverable a client
receives: a findings register whose severities are derived by rule, the evidence
tables, an assumptions register and a lineage block. It is published at
[kwantil.com/specimen/model-health-check](https://kwantil.com/specimen/model-health-check/).

It scores 7 findings. The ones the article does not already carry:

- **The integrity check found a defect in one of the 5 models.** As first
  released, the joint VaR and ES model stated an expected shortfall shallower
  than its own VaR on 463 of 2,172 days at the 99% level and 1,880 of 2,172 at
  97.5%. That is fixed (see [CHANGELOG.md](CHANGELOG.md)); the model's ES
  numbers changed, its VaR did not, and its zone is the same in every cell of
  the reference sweep, so no count in the article moved. The old output is the
  report's control: the case where the check fires.
- **The breach record in supervisory terms.** At the 99% level the Parametric
  model sat at 5 or more overshootings in 54% of 1,923 rolling 250-day windows
  (95% month-block interval 28% to 86%), against 10.8% for a correctly stated
  model.
- **An illustration in money.** On an assumed EUR 100 million book, the addend
  of Article 366 of Regulation (EU) No 575/2013 raised the VaR component of the
  own funds requirement by EUR 1.621 million on average (interval EUR 0.629
  million to EUR 2.643 million). The book value is an assumption, only the VaR
  component is computed, and the rule is the one applicable until 1 January
  2027. With the addend included, that model still carried EUR 3.5 million less
  than the model that held its level.

The model rule was revised once after the first run; `meta.json` records what
changed and why. Rule text, versions and retrieval date are in `meta.json` under
`sources`.

```bash
python scripts/audit_health_check.py      # ~20 s, needs experiments/factorial/t/details.csv
```

Shipped: `artifacts/health_check/`. What it does not support: any claim about a
real firm's capital, about another book, or that a model scored Observation is
adequate.

## Data

SPY, TLT, GLD daily closes via `yfinance`. No key required.

The price cache is **not** redistributed; Yahoo Finance terms do not permit it.
`fetch_prices` rebuilds it on first run. `output/details.csv` is row-level and
derived from those prices, so it is rebuilt rather than shipped.

Shipped: `output/summary.csv` and the null-sensitivity results under
`experiments/factorial/`.

## Running

`pytest -q` runs 54 tests in about 27 seconds. The full pipeline and the
factorial are in
[REPRODUCING.md](../../REPRODUCING.md#2-tail-risk-backtest-and-the-null-sensitivity).

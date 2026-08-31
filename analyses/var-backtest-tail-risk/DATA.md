# Data

The only data files committed here are small aggregates: `output/summary.csv`
and the two `experiments/factorial/*/as_null_sensitivity.csv` files. No inputs
and no row-level outputs are committed. This document is how you rebuild
everything.

## Source

Daily adjusted closes for **SPY**, **TLT**, **GLD** via
[`yfinance`](https://github.com/ranaroussi/yfinance). No account, no key.

**Why the cache is not shipped.** Yahoo Finance terms do not permit
redistributing their market data. `fetch_prices` rebuilds it on first run, so
the omission costs one download.

## Published window

Price history from `2010-01-04`. The **graded** window is `2018-01-02` to
`2026-08-24`, 2,172 trading days.

Those differ because the run takes a `--start` (default `2018-01-01`) and the
forecasters consume history before it: a 500-day fit window and 500
out-of-sample warmup steps. The earlier history is used but never graded.

Running today fetches a later end date, so counts will move. Pass `--end` to
pin it.

## Rebuild

```bash
pip install -e .
pytest -q                                        # 31 tests, ~26 s

python scripts/run_backtest.py --method all      # ~6 min
```

Writes `output/summary.csv` and `output/details.csv`. `summary.csv` is
aggregate and **is** committed, so compare your run against it. `details.csv`
is row-level and price-derived, so it is rebuilt rather than shipped.

### The factorial and the null sweep

The central claim of the article. Two runs, one per copula; the null is applied
after the fact and costs nothing to vary.

```bash
python scripts/run_backtest.py --copula t        --no-dashboard \
    --output-dir experiments/factorial/t
python scripts/run_backtest.py --copula gaussian --no-dashboard \
    --output-dir experiments/factorial/gaussian

python scripts/as_null_sensitivity.py --details experiments/factorial/t/details.csv
python scripts/as_null_sensitivity.py --details experiments/factorial/gaussian/details.csv
```

Expected: **0 green under a Gaussian null, 10 green under a df-5 null**, with
the shipped moment-matched t at 6.

## What gets built

| Artifact | Size | Built by |
|---|---|---|
| `data/cache/*.parquet` | ~600 KB | `fetch_prices`, on first run |
| `output/summary.csv` | ~4 KB | `run_backtest.py` |
| `output/details.csv` | ~2 MB | `run_backtest.py` |
| `experiments/factorial/*/as_null_sensitivity.csv` | ~8 KB each | `as_null_sensitivity.py` |

## Cost and time

Free. About 6 minutes per backtest arm, so roughly 12 for the full factorial.
The null sweep is about a minute.

## Determinism

Every Monte-Carlo step is seeded in the calling script. A clean re-run
reproduces published breach counts and Z2 statistics bit-for-bit, which has been
verified.

The exception is upstream: Yahoo restates history for splits and dividends, so a
pull months from now may not equal a pull today.

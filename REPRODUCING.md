# Reproducing

Every result in these analyses is seeded. Given the same inputs, a clean run
reproduces the published figures exactly, not approximately. Where a run is
stochastic the seed is fixed in the calling script rather than passed on the
command line, so there is nothing to remember.

If a number here does not match a number in the corresponding article, that is
a bug worth an issue.

---

## Environment

**Windows note.** Every runnable script reconfigures stdout to UTF-8 on start.
Without that, a Windows console defaults to cp1252 and any character outside it
kills the script at the moment it prints a result. If you fork a script, keep
the guard.


Python 3.11. Each analysis pins its own dependencies; there is no shared
lockfile because the three have little overlap and pinning them together would
constrain all of them to satisfy the union.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

---

## 1. German grid forecast calibration

**Data.** ENTSO-E Transparency Platform. Registration is free; the API token
arrives by email after requesting API access from your account settings.

```bash
cd analyses/german-grid-forecast-calibration
pip install pandas numpy scipy pyarrow entsoe-py

export ENTSOE_API_TOKEN=...        # Windows: $env:ENTSOE_API_TOKEN="..."

python src/pull_data.py            # DE-LU load, wind, solar; day-ahead vs actual
python src/pull_data_nl.py         # NL comparison zone
python src/assemble.py             # builds artifacts/panel.parquet
python src/calibration.py          # coverage and conditional bias
python src/monthly_coverage.py     # coverage by month, the regime-break result
pytest -q                          # 11 tests, ~2 s
python src/nl_monetize.py          # imbalance exposure in euros
python src/audit_specimen.py --zone NL   # the specimen Vendor Band Audit
```

The pull is the slow step and rate-limited at the source. `artifacts/panel.parquet`
is not shipped; `assemble.py` rebuilds it.

No results are shipped for this analysis. Every artifact above is rebuilt by the
commands above; see `DATA.md`.

**Note on window.** The published audit covers 2025-07-01 to 2026-07-31. Running
today pulls a later window, so figures will move. Pass explicit dates to
`pull_data.py` to reproduce the published numbers.

---

## 2. Tail-risk backtest and the null sensitivity

**Data.** SPY, TLT, GLD daily closes via `yfinance`. No key required. Fetched to
a local Parquet cache on first run.

```bash
cd analyses/var-backtest-tail-risk
pip install -e .
pytest -q                          # 31 tests, ~26 s
```

### The full backtest

```bash
python scripts/run_backtest.py --method all
```

Roughly 6 minutes: 5 methods, 2 confidence levels, 2,172 graded trading days
walk-forward, with a 500-day fit window and 500 out-of-sample warmup steps
before grading begins.

Writes `output/summary.csv` and `output/details.csv`. `summary.csv` is
aggregate and **is** shipped, so compare against it. `details.csv` is row-level
and price-derived, so it is rebuilt rather than shipped.

### The de-confounding factorial

The article's central claim. Two backtest runs, one per copula; the null is
applied after the fact and costs nothing to vary.

```bash
python scripts/run_backtest.py --copula t        --no-dashboard \
    --output-dir experiments/factorial/t
python scripts/run_backtest.py --copula gaussian --no-dashboard \
    --output-dir experiments/factorial/gaussian
```

### The six-null sweep

```bash
python scripts/as_null_sensitivity.py --details experiments/factorial/t/details.csv
python scripts/as_null_sensitivity.py --details experiments/factorial/gaussian/details.csv
```

Re-scores every model against six reference distributions with the model, the
data, the VaR and the ES all held fixed. Only the acceptance threshold moves.

Expected: **0 green under a Gaussian null, 10 green under a df-5 null**, with
the moment-matched t at 6. The result files
`experiments/factorial/*/as_null_sensitivity.csv` are shipped, so you can check
your run against them without rebuilding anything.

Reading the two arms together isolates the two effects. Down a column, changing
only the copula moves the statistic by +0.0096. Across a row, changing only the
null moves the green threshold by −0.104.

---

## 3. Tool wear calibration

**Data.** NASA Milling Data Set (Agogino and Goebel, 2007; BEST Lab, UC
Berkeley), distributed by the NASA Prognostics Center of Excellence. Not
redistributed here. Download
https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip (14.7 MB) and extract
`mill.mat` (69 MB) anywhere under `artifacts/raw/`; the scripts search for it.

```bash
cd analyses/tool-wear-calibration
pip install numpy scipy pandas scikit-learn

pytest -q                          # 15 tests, ~2 s
python src/features.py             # windowed features; flags clipped channels
python src/characterise.py         # per-insert wear trajectories
python src/calibrate.py            # 6 interval constructions, leave-one-insert-out
python src/decision.py             # change-policy simulation
python src/ablation.py             # sensitivity to the wear limit
python src/export_figures.py       # writes artifacts/figures/*.json
```

`calibrate.py` is the long one: 6 constructions × 4 feature sets × 25 seeds ×
16 leave-one-insert-out folds.

The figure JSON under `artifacts/figures/` is shipped, so the published charts
can be checked without running the pipeline.

**The split that matters.** Every construction holds out whole *inserts*, never
rows. Splitting by row leaves cuts from the same insert on both sides and
produces intervals that report 89.8% against a nominal 90%, pass any marginal
check, and are 19% too narrow. `src/review_checks.py` demonstrates the failure
directly.

---

## Determinism, and where it stops

Seeded and exactly reproducible: all Monte-Carlo p-values, the AS bootstrap, the
conformal calibration draws, the cluster bootstraps.

Not fully reproducible, and honestly so:

- **Upstream data revisions.** ENTSO-E restates actuals. Yahoo adjusts for
  splits and dividends retroactively. Both change historical values under you.
- **Solver and BLAS versions.** GARCH fits are numerical optimisations; a
  different SciPy or a different BLAS can move a fitted parameter in the last
  couple of digits. That has never moved a published conclusion here, but it can
  move a digit.
- **Wall-clock.** Nothing depends on it. No result changes with the date it is
  run on, only with the data window it covers.

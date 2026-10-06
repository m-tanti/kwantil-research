# Changelog

All notable changes to `var-backtest-uq` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the methodology-level history (when the t-copula was introduced, when
the joint VaR-ES variant was added, what the empirical findings were), see
the [methodology changelog in the dashboard manual](https://kwantil.com/var-backtest/manual#changelog).
This file tracks the package's API and packaging changes only.

## [Unreleased]

### Changed (breaking)

- **Acerbi-Szekely statistics renamed to match the paper.** Up to 0.1.x the
  names were swapped. `acerbi_szekely_test_1` is now the conditional test (Z1:
  mean of X_t/|ES_t| over breach days, plus 1; no `alpha` argument) and
  `acerbi_szekely_test_2` the unconditional one (Z2: the same sum over T*alpha;
  takes `alpha`). A call written against 0.1.x raises a `TypeError` on the
  `alpha` argument rather than silently returning the other statistic.
- Summary columns follow: `as_z1_*` is now the conditional statistic and
  `as_z2_*` the unconditional one. Each keeps the bootstrap seed it had before,
  so 0.1.x outputs reproduce exactly under the corrected labels. The shipped
  `output/summary.csv` has been relabelled, with no value changed.
  `scripts/compare_runs.py` detects a 0.1.x summary from the exact identity
  Z2 = 1 + N_breach/(T*alpha) * (Z1 - 1) and relabels it on load.
- The dashboard alias `es_zone` / `es_mean_shortfall_ratio` still points at the
  conditional statistic (now `as_z1_*`), as before.
- `scripts/as_null_sensitivity.py` scores both statistics on the same simulated
  paths (`z1_*` and `z2_*` columns, plus `kupiec_pvalue`), and adds 2
  known-broken models built from `conformal_pid`: `broken_scaled_0.8` (too many
  breaches) and `broken_gauss_matched` (right breach rate, thin tail). Its
  `z1_*` columns reproduce the previous `z2` / `zone` / `green_threshold`
  columns bit for bit.
- Validation PDF: VaR backtesting is labelled as CRR Article 366, not FRTB. The
  previous "FRTB IMA" headings mixed regimes: FRTB (MAR32) uses a different
  multiplier table and desk-level tests, and does not backtest ES for capital.
  There is no FRTB mode. The ES section reports Z1 and Z2 side by side, states
  that it is a validation diagnostic rather than a regulatory test, and
  distinguishes its bootstrap zones from the Basel traffic light. A supervisory
  references section lists the ECB guide to internal models, PRA SS1/23,
  OCC 2026-13 and CESR/10-788.

### Added

- `scripts/export_article_figures.py`, which reshapes the shipped sweep files
  into the article's figure data.
- Tests for the Z1/Z2 identity and for Z1's blindness to breach frequency.

## [0.1.0] 2026-05-08

Initial public release.

### Added

- Walk-forward VaR/ES backtester (`BacktestRunner` + `BacktestResult`) with
  five candidate methods: parametric Gaussian, historical 250-day, GARCH +
  Gaussian/t-copula, Conformal-PID (VaR-only), and Conformal-PID + ES
  (joint).
- Conformal calibration via `ConformalCalibrator` (VaR only) and
  `JointVarEsCalibrator` (joint VaR-ES with per-α `EsCorrector` feedback
  loop). Implements Angelopoulos, Candès & Tibshirani (2024) "Conformal
  PID Control for Time Series Prediction" with vol-EMA normalisation and
  MDN-style width scaling.
- Regulatory backtests: Kupiec POF, Christoffersen independence and
  conditional coverage (with Monte-Carlo or asymptotic p-values), Basel III
  traffic light per CRR Article 366, and Acerbi-Szekely Z₁ / Z₂ ES tests
  (names swapped relative to the paper in this release; fixed in Unreleased).
- Bootstrap H₀ samplers for the AS tests: `gaussian_h0_sampler` for
  Gaussian predictive distributions; `student_t_h0_sampler` (per-step,
  moment-matched) for fat-tailed distributions.
- t-copula option on `DistributionalForecaster` with the copula df fit from
  the average of per-asset GARCH-fitted ν, floored at 4 for MC stability.
- Reporting outputs: `write_dashboard_payload(result, out_dir)` emits JSON
  and Parquet artefacts for the dashboard SvelteKit frontend (not yet
  published; it will live alongside this repository on GitHub); `compile_report(result, out_path)` produces a
  regulatory PDF mapped section by section to OCC 2026-13 lifecycle and
  FRTB IMA backtesting headings via Typst (the FRTB labels were incorrect;
  relabelled to CRR Article 366 in Unreleased).
- Typed result records (`KupiecResult`, `ChristoffersenIndependenceResult`,
  `ChristoffersenCCResult`, `AcerbiSzekelyResult`, `BaselTrafficLightResult`)
  for IDE / type-checker friendliness.
- CLI entry point `scripts/run_backtest.py` (thin argparse layer) over the
  library function `var_backtest_uq.cli.run_backtest_pipeline` so the same
  pipeline can run programmatically without subprocess.
- 31-test pytest suite covering forecasters, calibrators, backtest
  statistics, and runner integration.

### Notes

- This is the initial public release; earlier development happened in a
  private repository. The methodology has been iterated since April 2026
  (see the [methodology changelog](https://kwantil.com/var-backtest/manual#changelog)
  for the t-copula switch, the Student-t H₀ bootstrap fix, and the joint
  VaR-ES variant additions, all dated 2026-05-07).
- No private dependencies; install pulls only numpy, scipy, pandas, arch,
  yfinance, pyarrow and pydantic.

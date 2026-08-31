# Changelog

All notable changes to `var-backtest-uq` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For the methodology-level history (when the t-copula was introduced, when
the joint VaR-ES variant was added, what the empirical findings were), see
the [methodology changelog in the dashboard manual](https://var-backtest.mtanti.com/manual#changelog).
This file tracks the package's API and packaging changes only.

## [Unreleased]

No changes yet beyond v0.1.0.

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
  traffic light per CRR Article 366, and Acerbi-Szekely Z₁ / Z₂ ES tests.
- Bootstrap H₀ samplers for the AS tests: `gaussian_h0_sampler` for
  Gaussian predictive distributions; `student_t_h0_sampler` (per-step,
  moment-matched) for fat-tailed distributions.
- t-copula option on `DistributionalForecaster` with the copula df fit from
  the average of per-asset GARCH-fitted ν, floored at 4 for MC stability.
- Reporting outputs: `write_dashboard_payload(result, out_dir)` emits JSON
  and Parquet artefacts for the dashboard SvelteKit frontend (not yet
  published; it will live alongside this repository on GitHub); `compile_report(result, out_path)` produces a
  regulatory PDF mapped section by section to OCC 2026-13 lifecycle and
  FRTB IMA backtesting headings via Typst.
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
  (see the [methodology changelog](https://var-backtest.mtanti.com/manual#changelog)
  for the t-copula switch, the Student-t H₀ bootstrap fix, and the joint
  VaR-ES variant additions, all dated 2026-05-07).
- No private dependencies; install pulls only numpy, scipy, pandas, arch,
  yfinance, pyarrow and pydantic.

"""Test compute_stats produces all expected columns including IMA capital."""

import numpy as np
import pandas as pd
import pytest

from var_backtest_uq.backtest.runner import BacktestResult
from var_backtest_uq.backtest.stats import (
    compute_stats,
    ima_capital_series,
    pairwise_capital_impact,
)


def _toy_result(n: int = 600) -> BacktestResult:
    """Synthetic two-method, two-alpha BacktestResult."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    realized = pd.Series(rng.normal(0, 0.01, n), index=dates, name="realized_pnl")

    rows: list[dict] = []
    breach_rows: list[dict] = []
    for method, sigma in [("good", 0.012), ("bad", 0.005)]:
        for alpha in [0.025, 0.01]:
            from scipy.stats import norm
            z = float(norm.ppf(alpha))
            var = sigma * z
            es = -sigma * float(norm.pdf(z)) / alpha
            for d, r in zip(dates, realized):
                rows.append({
                    "date": d, "method": method, "alpha": alpha,
                    "var": var, "es": es,
                    "pnl_mu": 0.0, "pnl_sigma": sigma,
                })
                breach_rows.append({
                    "date": d, "method": method, "alpha": alpha,
                    "breached": bool(r < var),
                })
    return BacktestResult(
        forecasts=pd.DataFrame(rows),
        realized=realized,
        breaches=pd.DataFrame(breach_rows),
    )


def test_compute_stats_has_ima_columns():
    result = _toy_result(600)
    df = compute_stats(result, basel_window=250, coverage_n_simulations=500, es_n_simulations=500)
    expected = {
        "method", "alpha", "kupiec_pvalue", "kupiec_pvalue_method",
        "as_z1_statistic", "as_z1_zone", "as_z1_pvalue",
        "as_z2_statistic", "as_z2_zone", "as_z2_pvalue",
        "mean_var_abs", "mean_ima_capital", "median_ima_capital", "peak_ima_capital",
    }
    assert expected.issubset(set(df.columns)), df.columns


def test_ima_capital_series_per_step():
    result = _toy_result(400)
    series = ima_capital_series(result, basel_window=250)
    assert {"date", "method", "alpha", "ima_capital"}.issubset(set(series.columns))
    # 400 dates × 2 methods × 2 alphas = 1600 rows
    assert len(series) == 400 * 2 * 2


def test_pairwise_capital_impact_alphas():
    result = _toy_result(400)
    df = compute_stats(result, basel_window=250, coverage_n_simulations=500, es_n_simulations=500)
    pw = pairwise_capital_impact(df)
    expected_cols = {
        "alpha", "adopt_method", "replace_method",
        "mean_ima_capital_change", "multiplier_change", "kupiec_p_change",
    }
    assert expected_cols.issubset(set(pw.columns))
    # 2 methods, 2 alphas, ordered pairs (a≠b): 2*1*2 = 4 rows
    assert len(pw) == 2 * 1 * 2

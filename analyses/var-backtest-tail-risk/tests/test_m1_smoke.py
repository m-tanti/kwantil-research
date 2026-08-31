"""M1 smoke test: the walk-forward loop runs end-to-end on synthetic data and
yields breach rates near nominal when the data-generating process matches the
parametric Gaussian assumption.
"""

import numpy as np
import pandas as pd

from var_backtest_uq.backtest.runner import BacktestRunner
from var_backtest_uq.forecasters.parametric import ParametricGaussianForecaster


def _synthetic_prices(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = np.array([
        [0.0001, 0.00002, 0.00001],
        [0.00002, 0.00005, 0.00001],
        [0.00001, 0.00001, 0.00007],
    ])
    returns = rng.multivariate_normal(mean=[0.0003, 0.0001, 0.0002], cov=cov, size=n)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(np.cumprod(np.exp(returns), axis=0) * 100,
                        index=dates, columns=["SPY", "TLT", "GLD"])


def test_walk_forward_runs_and_produces_expected_shapes():
    runner = BacktestRunner(
        methods={"parametric": ParametricGaussianForecaster(lam=0.94)},
        portfolio_weights={"SPY": 1/3, "TLT": 1/3, "GLD": 1/3},
        fit_window=500,
        alpha_targets=(0.025, 0.01),
    )
    result = runner.run(_synthetic_prices())

    n_steps = len(result.realized)
    assert n_steps > 400
    assert result.forecasts.shape[0] == n_steps * 2  # 2 alphas
    assert result.breaches.shape[0] == n_steps * 2
    required = {"date", "method", "alpha", "var", "es"}
    assert required.issubset(set(result.forecasts.columns))


def test_breach_rates_near_nominal_under_gaussian_dgp():
    runner = BacktestRunner(
        methods={"parametric": ParametricGaussianForecaster(lam=0.94)},
        portfolio_weights={"SPY": 1/3, "TLT": 1/3, "GLD": 1/3},
        fit_window=500,
        alpha_targets=(0.025, 0.01),
    )
    result = runner.run(_synthetic_prices(n=2000, seed=1))

    rate_25 = result.breaches.query("alpha == 0.025")["breached"].mean()
    rate_10 = result.breaches.query("alpha == 0.01")["breached"].mean()
    assert 0.005 < rate_25 < 0.05, f"alpha=0.025 breach rate {rate_25} out of sane band"
    assert 0.0 < rate_10 < 0.03, f"alpha=0.01 breach rate {rate_10} out of sane band"


def test_var_below_es_for_lower_tail():
    """ES at the lower tail must be at least as negative as VaR (E[X|X<=q] <= q)."""
    runner = BacktestRunner(
        methods={"parametric": ParametricGaussianForecaster(lam=0.94)},
        portfolio_weights={"SPY": 1/3, "TLT": 1/3, "GLD": 1/3},
        fit_window=500,
        alpha_targets=(0.025,),
    )
    result = runner.run(_synthetic_prices())
    f = result.forecasts.query("alpha == 0.025")
    assert (f["es"] <= f["var"] + 1e-9).all(), "ES_a should be <= VaR_a for the lower tail"

"""Tests for the Student-t marginal distribution + t-GARCH base."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from var_backtest_uq.forecasters.distributional import (
    DistributionalForecaster,
    StudentTDistribution,
)


def test_student_t_quantile_matches_scipy():
    d = StudentTDistribution(mu=0.0, sigma=1.0, nu=5.0)
    for p in [0.01, 0.025, 0.05, 0.5, 0.95]:
        assert d.quantile(p) == pytest.approx(float(student_t.ppf(p, df=5)))


def test_student_t_es_matches_mc_estimate():
    """Analytic ES for Student-t should agree with MC."""
    d = StudentTDistribution(mu=0.0, sigma=1.0, nu=6.0)
    rng = np.random.default_rng(0)
    n = 500_000
    samples = d.sample(n, rng=rng)
    var_emp = np.percentile(samples, 2.5)
    es_emp = float(samples[samples <= var_emp].mean())
    es_analytic = d.es(0.025)
    assert es_analytic == pytest.approx(es_emp, rel=0.05)


def test_student_t_es_more_conservative_than_gaussian():
    """For the same sigma, Student-t ES is more negative (deeper) than Gaussian."""
    from var_backtest_uq.forecasters.parametric import GaussianDistribution

    sigma = 0.01
    t_dist = StudentTDistribution(mu=0.0, sigma=sigma, nu=5.0)
    g_dist = GaussianDistribution(mu=0.0, sigma=sigma)
    assert t_dist.es(0.025) < g_dist.es(0.025), (
        f"Student-t ES should be more negative than Gaussian; "
        f"got t={t_dist.es(0.025)}, g={g_dist.es(0.025)}"
    )


def test_t_garch_fit_predict_smoke():
    """t-GARCH end-to-end: fit + predict return a valid portfolio_pnl distribution."""
    rng = np.random.default_rng(0)
    n = 600
    cov = np.array([
        [0.0001, 2e-5, 1e-5],
        [2e-5, 5e-5, 1e-5],
        [1e-5, 1e-5, 7e-5],
    ])
    returns = pd.DataFrame(
        rng.multivariate_normal(np.zeros(3), cov, size=n),
        columns=["SPY", "TLT", "GLD"],
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    forecaster = DistributionalForecaster(refit_every=20, dist="t")
    forecaster.fit(returns)
    forecast = forecaster.predict(weights=np.array([1/3, 1/3, 1/3]))
    pnl = forecast.portfolio_pnl()

    # nu was fitted; should be well above 2 for a fat-tailed estimate.
    nus = [p.get("nu") for p in forecaster._params]
    assert all(nu is not None and nu > 2.0 for nu in nus), f"nu out of range: {nus}"

    var_001 = pnl.quantile(0.01)
    es_001 = pnl.es(0.01)
    assert var_001 < 0.0
    assert es_001 < var_001


def test_t_garch_es_more_conservative_than_normal_garch():
    """On the same returns, t-GARCH should have deeper (more negative) ES at 1% than Gaussian-GARCH."""
    rng = np.random.default_rng(7)
    n = 800
    cov = np.array([
        [0.0001, 2e-5, 1e-5],
        [2e-5, 5e-5, 1e-5],
        [1e-5, 1e-5, 7e-5],
    ])
    returns = pd.DataFrame(
        rng.multivariate_normal(np.zeros(3), cov, size=n),
        columns=["SPY", "TLT", "GLD"],
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )
    weights = np.array([1/3, 1/3, 1/3])

    f_t = DistributionalForecaster(refit_every=20, dist="t")
    f_n = DistributionalForecaster(refit_every=20, dist="normal")
    f_t.fit(returns)
    f_n.fit(returns)
    es_t = f_t.predict(weights).portfolio_pnl().es(0.01)
    es_n = f_n.predict(weights).portfolio_pnl().es(0.01)
    assert es_t < es_n, f"expected t-GARCH ES more negative; got t={es_t}, n={es_n}"

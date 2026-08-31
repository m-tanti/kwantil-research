"""Shared test fixtures and factories.

Tests that need parametrised setup import the factories directly:

    from .conftest import synthetic_returns, make_conformal_calibrator
    returns = synthetic_returns(n=400, seed=1)
    cal = make_conformal_calibrator((0.025,))

Tests that just want the standard 800-day, 2-α calibrator can rely on the
fixtures:

    def test_x(returns_800, conformal_calibrator):
        cal.fit(returns_800); ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from var_backtest_uq.conformal.calibrator import ConformalCalibrator
from var_backtest_uq.conformal.controllers import ConformalPIDController
from var_backtest_uq.conformal.joint_calibrator import JointVarEsCalibrator
from var_backtest_uq.forecasters.distributional import DistributionalForecaster


# Synthetic data helpers. Deterministic, structure-only (the covariance is
# positive-definite but otherwise unrealistic). Use for unit tests where the
# calibrator's behaviour matters but realism does not.


def synthetic_returns(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """3-asset Gaussian returns with a fixed positive-definite covariance,
    indexed at business-day frequency from 2018-01-01."""
    rng = np.random.default_rng(seed)
    cov = np.array([
        [0.0001, 2e-5, 1e-5],
        [2e-5, 5e-5, 1e-5],
        [1e-5, 1e-5, 7e-5],
    ])
    arr = rng.multivariate_normal(np.zeros(3), cov, size=n)
    return pd.DataFrame(
        arr, columns=["SPY", "TLT", "GLD"],
        index=pd.date_range("2018-01-01", periods=n, freq="B"),
    )


def synthetic_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative-product price path from log-returns, base 100."""
    return pd.DataFrame(
        np.cumprod(np.exp(returns.to_numpy()), axis=0) * 100.0,
        index=returns.index, columns=returns.columns,
    )


# Calibrator factories. Default to a Gaussian-copula base so the tests
# don't pay the t-copula MC sampling cost.


def make_conformal_calibrator(
    alphas: tuple[float, ...] = (0.025, 0.01),
    *,
    copula: str = "gaussian",
) -> ConformalCalibrator:
    """Test-friendly `ConformalCalibrator` with 300-step window."""
    return ConformalCalibrator(
        base=DistributionalForecaster(refit_every=20, dist="t", copula=copula),
        portfolio_controllers={
            a: ConformalPIDController(alpha=a, window_size=300, smooth_alpha=1.0)
            for a in alphas
        },
        per_asset_controller_factory=lambda a, _asset: ConformalPIDController(
            alpha=a, window_size=300, smooth_alpha=1.0,
        ),
    )


def make_joint_calibrator(
    alphas: tuple[float, ...] = (0.025, 0.01),
    *,
    copula: str = "gaussian",
) -> JointVarEsCalibrator:
    """Test-friendly `JointVarEsCalibrator` (same VaR setup as
    `make_conformal_calibrator` plus the default ES corrector)."""
    return JointVarEsCalibrator(
        base=DistributionalForecaster(refit_every=20, dist="t", copula=copula),
        portfolio_controllers={
            a: ConformalPIDController(alpha=a, window_size=300) for a in alphas
        },
    )


# Fixtures for the standard-shape cases.


@pytest.fixture
def returns_800() -> pd.DataFrame:
    return synthetic_returns(n=800)


@pytest.fixture
def returns_600() -> pd.DataFrame:
    return synthetic_returns(n=600)


@pytest.fixture
def conformal_calibrator() -> ConformalCalibrator:
    """Default `ConformalCalibrator` at α ∈ (0.025, 0.01)."""
    return make_conformal_calibrator((0.025, 0.01))


@pytest.fixture
def joint_calibrator() -> JointVarEsCalibrator:
    """Default `JointVarEsCalibrator` at α ∈ (0.025, 0.01)."""
    return make_joint_calibrator((0.025, 0.01))

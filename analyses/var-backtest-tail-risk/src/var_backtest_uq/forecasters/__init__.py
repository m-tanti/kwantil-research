"""Forecaster implementations.

All forecasters implement the `Forecaster` protocol from `forecasters.base`
and return a `PortfolioForecast` whose `.portfolio_pnl()` and
`.asset_pnl(asset)` expose `Distribution`-typed marginals.

- `parametric`, RiskMetrics-style EWMA volatility + Gaussian copula.
- `historical`, empirical 250-day rolling-window simulation.
- `distributional`, t-GARCH(1,1) marginals + Gaussian or t-copula.

Forecasters are stateless across `predict` calls, fit once on the rolling
window, predict the next step, repeat. Calibrators (in
`var_backtest_uq.conformal`) wrap a base forecaster to add online conformal
calibration.
"""

from .base import Distribution, Forecaster, PortfolioForecast
from .distributional import DistributionalForecaster, MarginalsPortfolioForecast, StudentTDistribution
from .historical import EmpiricalDistribution, HistoricalPortfolioForecast, HistoricalSimulationForecaster
from .parametric import (
    GaussianDistribution,
    GaussianPortfolioForecast,
    ParametricGaussianForecaster,
    ewma_vol,
)

__all__ = [
    # Protocols
    "Forecaster",
    "PortfolioForecast",
    "Distribution",
    # Parametric
    "ParametricGaussianForecaster",
    "GaussianPortfolioForecast",
    "GaussianDistribution",
    "ewma_vol",
    # Historical
    "HistoricalSimulationForecaster",
    "HistoricalPortfolioForecast",
    "EmpiricalDistribution",
    # Distributional (GARCH + copula)
    "DistributionalForecaster",
    "MarginalsPortfolioForecast",
    "StudentTDistribution",
]

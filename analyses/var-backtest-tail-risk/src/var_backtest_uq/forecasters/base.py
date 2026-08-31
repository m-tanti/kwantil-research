"""Forecaster + Distribution + PortfolioForecast protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Distribution(Protocol):
    """One-step-ahead scalar predictive distribution."""

    def quantile(self, p: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...
    def es(self, p: float) -> float: ...


@runtime_checkable
class PortfolioForecast(Protocol):
    """Joint forecast over assets, with portfolio aggregation."""

    @property
    def weights(self) -> np.ndarray: ...
    @property
    def asset_names(self) -> tuple[str, ...]: ...

    def portfolio_pnl(self) -> Distribution: ...
    def asset_pnl(self, asset: str) -> Distribution: ...
    def joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray: ...


@runtime_checkable
class Forecaster(Protocol):
    """Fits on historical returns, predicts a one-step-ahead PortfolioForecast."""

    def fit(self, returns: pd.DataFrame) -> None: ...
    def predict(self, weights: np.ndarray) -> PortfolioForecast: ...

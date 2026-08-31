"""Typed contracts for the conformal calibrators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PredictiveDistribution(Protocol):
    """Predictive distribution answering quantile and CDF queries."""

    def quantile(self, q: float) -> np.ndarray: ...
    def cdf(self, x: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ConformalMethod(Protocol):
    """Online conformal calibrator: fit, predict, update."""

    @property
    def is_fitted(self) -> bool: ...
    def fit(self, forecasts: np.ndarray, actuals: np.ndarray) -> None: ...
    def predict(self, forecast: np.ndarray, **kwargs) -> PredictiveDistribution: ...
    def update(self, residual: np.ndarray, actual: np.ndarray) -> None: ...

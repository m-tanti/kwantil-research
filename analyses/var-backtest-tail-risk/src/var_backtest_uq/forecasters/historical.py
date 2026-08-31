"""Historical-simulation forecaster: empirical joint over a rolling lookback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Distribution, Forecaster, PortfolioForecast


class EmpiricalDistribution:
    """Empirical CDF / quantile / ES backed by a sample array."""

    def __init__(self, samples: np.ndarray):
        s = np.asarray(samples, dtype=float).ravel()
        if s.size == 0:
            raise ValueError("EmpiricalDistribution requires at least one sample")
        self.samples = s

    def quantile(self, p: float) -> float:
        return float(np.percentile(self.samples, p * 100.0))

    def cdf(self, x: float) -> float:
        return float(np.mean(self.samples <= x))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.choice(self.samples, size=n, replace=True)

    def es(self, p: float) -> float:
        threshold = self.quantile(p)
        tail = self.samples[self.samples <= threshold]
        if tail.size == 0:
            return threshold
        return float(np.mean(tail))


class HistoricalPortfolioForecast:
    """Empirical joint over assets in the lookback window, projected onto weights."""

    def __init__(
        self,
        return_matrix: np.ndarray,
        weights: np.ndarray,
        asset_names: tuple[str, ...],
    ):
        self._returns = np.asarray(return_matrix, dtype=float)  # (n_lookback, n_assets)
        self._weights = np.asarray(weights, dtype=float)
        self._asset_names = tuple(asset_names)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def asset_names(self) -> tuple[str, ...]:
        return self._asset_names

    def portfolio_pnl(self) -> Distribution:
        return EmpiricalDistribution(self._returns @ self._weights)

    def asset_pnl(self, asset: str) -> Distribution:
        idx = self._asset_names.index(asset)
        return EmpiricalDistribution(self._returns[:, idx])

    def joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        idx = rng.integers(0, self._returns.shape[0], size=n)
        return self._returns[idx]


class HistoricalSimulationForecaster:
    """Empirical 250-day rolling-window forecast.

    `lookback` <= runner's fit_window: only the most recent `lookback` rows
    are used, no reweighting needed when fit_window is larger.
    """

    def __init__(self, lookback: int = 250):
        if lookback < 30:
            raise ValueError(f"lookback={lookback} too short")
        self.lookback = lookback
        self._fitted = False
        self._asset_names: tuple[str, ...] = ()
        self._return_matrix: np.ndarray = np.zeros((0, 0))

    def fit(self, returns: pd.DataFrame) -> None:
        self._asset_names = tuple(returns.columns)
        if len(returns) < self.lookback:
            raise ValueError(
                f"need at least lookback={self.lookback} returns; got {len(returns)}"
            )
        self._return_matrix = returns.iloc[-self.lookback :].to_numpy(dtype=float)
        self._fitted = True

    def predict(self, weights: np.ndarray) -> PortfolioForecast:
        if not self._fitted:
            raise RuntimeError("HistoricalSimulationForecaster has not been fitted")
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != len(self._asset_names):
            raise ValueError(
                f"weights length {weights.shape[0]} != {len(self._asset_names)} assets"
            )
        return HistoricalPortfolioForecast(
            return_matrix=self._return_matrix,
            weights=weights,
            asset_names=self._asset_names,
        )

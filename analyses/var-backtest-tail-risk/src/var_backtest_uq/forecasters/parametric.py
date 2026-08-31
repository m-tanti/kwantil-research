"""RiskMetrics-style EWMA Gaussian baseline (zero-mean drift).

Deliberately the textbook foil: breaks under tail-volatility regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import Distribution, Forecaster, PortfolioForecast


class GaussianDistribution:
    """Analytic Gaussian with vectorised quantile/cdf for copula-MC pipelines."""

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def quantile(self, p):
        return self.mu + self.sigma * norm.ppf(p)

    def cdf(self, x):
        if self.sigma <= 0:
            arr = np.asarray(x, dtype=float)
            return np.where(arr >= self.mu, 1.0, 0.0)
        return norm.cdf((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.mu + self.sigma * rng.standard_normal(n)

    def es(self, p: float) -> float:
        z = float(norm.ppf(p))
        return self.mu - self.sigma * float(norm.pdf(z)) / p


class GaussianPortfolioForecast:
    """Joint Gaussian over assets with portfolio aggregation."""

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        weights: np.ndarray,
        asset_names: tuple[str, ...],
    ):
        self._mu = np.asarray(mu, dtype=float)
        self._cov = np.asarray(cov, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self._asset_names = tuple(asset_names)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def asset_names(self) -> tuple[str, ...]:
        return self._asset_names

    @property
    def correlation(self) -> np.ndarray:
        sigma = np.sqrt(np.maximum(np.diag(self._cov), 0.0))
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        return self._cov / np.outer(sigma_safe, sigma_safe)

    def asset_marginal(self, asset: str) -> Distribution:
        return self.asset_pnl(asset)

    def portfolio_pnl(self) -> Distribution:
        mu_p = float(self._weights @ self._mu)
        var_p = float(self._weights @ self._cov @ self._weights)
        return GaussianDistribution(mu=mu_p, sigma=np.sqrt(max(var_p, 0.0)))

    def asset_pnl(self, asset: str) -> Distribution:
        idx = self._asset_names.index(asset)
        return GaussianDistribution(mu=float(self._mu[idx]), sigma=float(np.sqrt(self._cov[idx, idx])))

    def joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.multivariate_normal(self._mu, self._cov, size=n)


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA vol; seeded from r_0^2 (converges after ~80 steps at lam=0.94)."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        raise ValueError("ewma_vol requires at least one return")
    sigma2 = float(r[0] ** 2)
    for ri in r[1:]:
        sigma2 = lam * sigma2 + (1.0 - lam) * float(ri) ** 2
    return float(np.sqrt(max(sigma2, 0.0)))


class ParametricGaussianForecaster:
    """EWMA vol per asset + sample correlation; joint Gaussian, zero-mean."""

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self._fitted = False
        self._asset_names: tuple[str, ...] = ()
        self._mu: np.ndarray = np.zeros(0)
        self._cov: np.ndarray = np.zeros((0, 0))

    def fit(self, returns: pd.DataFrame) -> None:
        self._asset_names = tuple(returns.columns)
        n_assets = returns.shape[1]
        sigmas = np.array(
            [ewma_vol(returns.iloc[:, i].values, lam=self.lam) for i in range(n_assets)]
        )
        corr = returns.corr().values
        self._mu = np.zeros(n_assets)
        self._cov = np.outer(sigmas, sigmas) * corr
        self._fitted = True

    def predict(self, weights: np.ndarray) -> PortfolioForecast:
        if not self._fitted:
            raise RuntimeError("ParametricGaussianForecaster has not been fitted")
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != len(self._asset_names):
            raise ValueError(
                f"weights length {weights.shape[0]} != {len(self._asset_names)} assets"
            )
        return GaussianPortfolioForecast(
            mu=self._mu, cov=self._cov, weights=weights, asset_names=self._asset_names,
        )

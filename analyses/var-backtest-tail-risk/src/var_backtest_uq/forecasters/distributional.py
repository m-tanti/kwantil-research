"""CCC-GARCH(1,1) portfolio forecast: per-asset GARCH marginals with a sample-correlation copula.

Innovations: Student-t (default) or Normal. Parameters refit every
`refit_every` steps; conditional variance is advanced daily via the
standard GARCH recursion between refits.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import multivariate_t, norm, t as student_t

from .base import Distribution, Forecaster, PortfolioForecast
from .historical import EmpiricalDistribution
from .parametric import GaussianDistribution, GaussianPortfolioForecast


# arch_model's optimizer is happier with returns scaled to percent.
_GARCH_SCALE = 100.0
# 5k samples -> ~0.7pp std on a 1% breach-rate estimate; bump if MC noise shows up.
_MC_SAMPLES_DEFAULT = 5_000


class StudentTDistribution:
    """Scaled Student-t with analytic ES. Vectorised via scipy.stats.t."""

    def __init__(self, mu: float, sigma: float, nu: float):
        if nu <= 1.0:
            raise ValueError(f"Student-t requires nu > 1 for finite mean; got nu={nu}")
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

    def quantile(self, p):
        z = student_t.ppf(p, df=self.nu)
        return self.mu + self.sigma * z

    def cdf(self, x):
        if self.sigma <= 0:
            arr = np.asarray(x, dtype=float)
            return np.where(arr >= self.mu, 1.0, 0.0)
        return student_t.cdf((np.asarray(x, dtype=float) - self.mu) / self.sigma, df=self.nu)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.mu + self.sigma * rng.standard_t(df=self.nu, size=n)

    def es(self, p: float) -> float:
        # McNeil/Frey/Embrechts 2015 S2.3.4.
        if self.nu <= 1.0:
            raise ValueError("ES not finite for Student-t with nu <= 1")
        z = float(student_t.ppf(p, df=self.nu))
        pdf_z = float(student_t.pdf(z, df=self.nu))
        return self.mu - self.sigma * (self.nu + z * z) / (self.nu - 1.0) * pdf_z / p


class MarginalsPortfolioForecast:
    """Per-asset marginals + a Gaussian or t-copula on the standardised residuals.

    portfolio_pnl() takes the Gaussian closed form when both marginals and
    copula are Gaussian; otherwise builds an EmpiricalDistribution by MC.
    """

    def __init__(
        self,
        marginals: list[Distribution],
        correlation: np.ndarray,
        weights: np.ndarray,
        asset_names: tuple[str, ...],
        n_mc_samples: int = _MC_SAMPLES_DEFAULT,
        rng_seed: int = 42,
        copula: Literal["gaussian", "t"] = "gaussian",
        copula_df: float | None = None,
    ):
        if len(marginals) != len(asset_names):
            raise ValueError("marginals and asset_names must align")
        if copula not in ("gaussian", "t"):
            raise ValueError(f"copula must be 'gaussian' or 't'; got {copula!r}")
        if copula == "t":
            if copula_df is None or copula_df <= 2.0:
                raise ValueError(
                    f"copula='t' requires copula_df > 2 for finite covariance; got {copula_df!r}"
                )
        self._marginals = list(marginals)
        self._correlation = np.asarray(correlation, dtype=float)
        self._weights = np.asarray(weights, dtype=float)
        self._asset_names = tuple(asset_names)
        self._n_mc = n_mc_samples
        self._rng_seed = rng_seed
        self._copula = copula
        self._copula_df = copula_df

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def asset_names(self) -> tuple[str, ...]:
        return self._asset_names

    @property
    def correlation(self) -> np.ndarray:
        return self._correlation.copy()

    def asset_marginal(self, asset: str) -> Distribution:
        return self._marginals[self._asset_names.index(asset)]

    def asset_pnl(self, asset: str) -> Distribution:
        return self.asset_marginal(asset)

    def _all_gaussian(self) -> bool:
        return all(isinstance(m, GaussianDistribution) for m in self._marginals)

    def portfolio_pnl(self) -> Distribution:
        # Closed-form only when marginals AND copula are Gaussian; under a
        # t-copula, sums of Gaussian marginals are no longer Gaussian.
        if self._all_gaussian() and self._copula == "gaussian":
            mu = np.array([m.mu for m in self._marginals], dtype=float)
            sigma = np.array([m.sigma for m in self._marginals], dtype=float)
            cov = self._correlation * np.outer(sigma, sigma)
            return GaussianDistribution(
                mu=float(self._weights @ mu),
                sigma=float(np.sqrt(max(self._weights @ cov @ self._weights, 0.0))),
            )
        samples = self._joint_sample(self._n_mc)
        return EmpiricalDistribution(samples @ self._weights)

    def _joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(self._rng_seed)
        d = len(self._marginals)
        if self._copula == "gaussian":
            z = rng.multivariate_normal(np.zeros(d), self._correlation, size=n)
            u = norm.cdf(z)
        else:
            df = self._copula_df
            assert df is not None  # checked in __init__
            z = multivariate_t(
                loc=np.zeros(d), shape=self._correlation, df=df, allow_singular=False,
            ).rvs(size=n, random_state=rng)
            u = student_t.cdf(z, df=df)
        out = np.empty_like(z)
        for k, marg in enumerate(self._marginals):
            out[:, k] = np.asarray(marg.quantile(u[:, k])).ravel()
        return out

    def joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._joint_sample(n, rng)


class DistributionalForecaster:
    """CCC-GARCH(1,1) joint forecast with periodic parameter refits.

    `dist`: "t" (default) or "normal" innovations. `copula`: "gaussian"
    (default, no upper-tail dependence) or "t" (shared `copula_df`,
    preserves joint extremes). `copula_df_floor` keeps MC variance finite
    when the fit drives df very low.
    """

    def __init__(
        self,
        refit_every: int = 20,
        p: int = 1,
        q: int = 1,
        dist: Literal["t", "normal"] = "t",
        copula: Literal["gaussian", "t"] = "gaussian",
        copula_df_floor: float = 4.0,
    ):
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1; got {refit_every}")
        if dist not in {"t", "normal"}:
            raise ValueError(f"dist must be 't' or 'normal'; got {dist!r}")
        if copula not in {"gaussian", "t"}:
            raise ValueError(f"copula must be 'gaussian' or 't'; got {copula!r}")
        if copula_df_floor <= 2.0:
            raise ValueError(f"copula_df_floor must be > 2; got {copula_df_floor}")
        self.refit_every = refit_every
        self.p = p
        self.q = q
        self.dist = dist
        self.copula = copula
        self.copula_df_floor = copula_df_floor
        self._steps_since_refit = 0
        self._fitted = False
        self._asset_names: tuple[str, ...] = ()
        self._params: list[dict[str, float]] = []
        self._sigma2_next: np.ndarray = np.zeros(0)   # scaled units
        self._corr: np.ndarray = np.zeros((0, 0))
        self._copula_df: float | None = None

    def fit(self, returns: pd.DataFrame) -> None:
        if not self._fitted or self._steps_since_refit >= self.refit_every:
            self._full_refit(returns)
            self._steps_since_refit = 0
        else:
            self._warm_advance(returns)
            self._steps_since_refit += 1

    def _full_refit(self, returns: pd.DataFrame) -> None:
        self._asset_names = tuple(returns.columns)
        n_assets = returns.shape[1]
        self._params = []
        sigma2_next = np.zeros(n_assets)
        std_resids = np.zeros((len(returns), n_assets))
        arch_dist = "t" if self.dist == "t" else "Normal"

        for k in range(n_assets):
            r_scaled = returns.iloc[:, k].to_numpy(dtype=float) * _GARCH_SCALE
            am = arch_model(
                r_scaled, vol="Garch", p=self.p, q=self.q,
                mean="Constant", dist=arch_dist, rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            params = {
                "mu": float(res.params["mu"]),
                "omega": float(res.params["omega"]),
                "alpha": float(res.params["alpha[1]"]),
                "beta": float(res.params["beta[1]"]),
            }
            if self.dist == "t":
                params["nu"] = float(res.params["nu"])
            self._params.append(params)
            sigma2_next[k] = float(
                res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
            )
            std_resids[:, k] = res.std_resid

        self._sigma2_next = sigma2_next
        self._corr = np.corrcoef(std_resids.T)
        if self._corr.ndim == 0:
            self._corr = np.array([[1.0]])
        if self.copula == "t":
            if self.dist == "t":
                # Mean per-asset fitted nu as a moment estimator for the joint df.
                avg_nu = float(np.mean([p["nu"] for p in self._params]))
            else:
                # kurt_excess = 6/(nu-4); cap at 0.5 so nu <= 16.
                kurt_excess = float(np.mean([
                    max(0.0, excess_kurtosis(std_resids[:, k])) for k in range(n_assets)
                ]))
                avg_nu = 4.0 + 6.0 / max(kurt_excess, 0.5)
            self._copula_df = max(avg_nu, self.copula_df_floor)
        self._fitted = True

    def _warm_advance(self, returns: pd.DataFrame) -> None:
        last_scaled = returns.iloc[-1].to_numpy(dtype=float) * _GARCH_SCALE
        for k, p in enumerate(self._params):
            resid = last_scaled[k] - p["mu"]
            sigma2_t = self._sigma2_next[k]
            self._sigma2_next[k] = p["omega"] + p["alpha"] * resid * resid + p["beta"] * sigma2_t

    def predict(self, weights: np.ndarray) -> PortfolioForecast:
        if not self._fitted:
            raise RuntimeError("DistributionalForecaster has not been fitted")
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != len(self._asset_names):
            raise ValueError(
                f"weights length {weights.shape[0]} != {len(self._asset_names)} assets"
            )
        sigma_raw = np.sqrt(np.maximum(self._sigma2_next, 0.0)) / _GARCH_SCALE
        if self.dist == "normal" and self.copula == "gaussian":
            D = np.diag(sigma_raw)
            cov = D @ self._corr @ D
            return GaussianPortfolioForecast(
                mu=np.zeros(len(self._asset_names)),
                cov=cov,
                weights=weights,
                asset_names=self._asset_names,
            )
        if self.dist == "t":
            marginals: list[Distribution] = [
                StudentTDistribution(mu=0.0, sigma=float(sigma_raw[k]), nu=self._params[k]["nu"])
                for k in range(len(self._asset_names))
            ]
        else:
            marginals = [
                GaussianDistribution(mu=0.0, sigma=float(sigma_raw[k]))
                for k in range(len(self._asset_names))
            ]
        return MarginalsPortfolioForecast(
            marginals=marginals,
            correlation=self._corr,
            weights=weights,
            asset_names=self._asset_names,
            copula=self.copula,
            copula_df=self._copula_df,
        )


def excess_kurtosis(x: np.ndarray) -> float:
    """Sample excess kurtosis. Returns 0 for too-small or zero-variance samples."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    m = x.mean()
    s2 = float(((x - m) ** 2).mean())
    if s2 <= 0:
        return 0.0
    m4 = float(((x - m) ** 4).mean())
    return m4 / (s2 * s2) - 3.0

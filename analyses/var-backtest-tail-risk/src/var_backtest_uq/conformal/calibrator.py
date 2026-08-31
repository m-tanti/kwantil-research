"""Conformal calibration: portfolio-level + per-asset controllers.

ES is read off the base distribution conditional on the calibrated VaR.
Controllers are seeded on the first fit(); subsequent updates flow through
observe_realized().
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from .protocols import ConformalMethod

from ..forecasters.base import Distribution, Forecaster, PortfolioForecast
from ..forecasters.distributional import StudentTDistribution
from ..forecasters.historical import EmpiricalDistribution
from ..forecasters.parametric import GaussianDistribution


def _es_from_base_below(base: Distribution, q: float) -> float:
    """E_base[X | X <= q] using whichever closed form fits the base type."""
    if isinstance(base, GaussianDistribution):
        if base.sigma <= 0:
            return q
        z = (q - base.mu) / base.sigma
        phi_z = float(norm.pdf(z))
        Phi_z = float(norm.cdf(z))
        if Phi_z < 1e-12:
            return q
        return float(base.mu - base.sigma * phi_z / Phi_z)
    if isinstance(base, StudentTDistribution):
        implied_p = float(base.cdf(q))
        if implied_p <= 0:
            return q
        return float(base.es(implied_p))
    if isinstance(base, EmpiricalDistribution):
        tail = base.samples[base.samples <= q]
        if tail.size == 0:
            return q
        return float(np.mean(tail))
    implied_p = float(base.cdf(q))
    if implied_p <= 0:
        return q
    return float(base.es(implied_p))


class CalibratedDistribution:
    """Base Distribution wrapped with per-alpha conformal controllers; uncalibrated p falls through to the base."""

    def __init__(
        self,
        base: Distribution,
        controllers: Mapping[float, ConformalMethod],
        forecast_mean: float,
        q_lo: float,
        q_hi: float,
    ):
        self._base = base
        self._controllers: dict[float, ConformalMethod] = dict(controllers)
        self._forecast_mean = forecast_mean
        self._q_lo = q_lo
        self._q_hi = q_hi
        # ctrl.predict() has per-step side effects; cache so es(alpha) doesn't fire it twice.
        self._quantile_cache: dict[float, float] = {}

    @property
    def mu(self) -> float | None:
        return getattr(self._base, "mu", None)

    @property
    def sigma(self) -> float | None:
        return getattr(self._base, "sigma", None)

    @property
    def samples(self) -> np.ndarray | None:
        return getattr(self._base, "samples", None)

    def quantile(self, p: float) -> float:
        if p in self._quantile_cache:
            return self._quantile_cache[p]
        ctrl = self._controllers.get(p)
        if ctrl is None or not ctrl.is_fitted:
            q = float(self._base.quantile(p))
        else:
            dist = ctrl.predict(
                np.array([self._forecast_mean]),
                quantile_lo=np.array([self._q_lo]),
                quantile_hi=np.array([self._q_hi]),
            )
            q = float(dist.quantile(p)[0])
        self._quantile_cache[p] = q
        return q

    def es(self, p: float) -> float:
        return _es_from_base_below(self._base, self.quantile(p))

    def cdf(self, x: float) -> float:
        return float(self._base.cdf(x))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._base.sample(n, rng)


class CalibratedPortfolioForecast:
    """Calibrated PortfolioForecast; public accessors let JointVarEsCalibrator wrap it without private-state access."""

    def __init__(
        self,
        base: PortfolioForecast,
        portfolio_controllers: Mapping[float, ConformalMethod],
        per_asset_controllers: Mapping[str, Mapping[float, ConformalMethod]],
        portfolio_signal: tuple[float, float, float],
        per_asset_signals: Mapping[str, tuple[float, float, float]],
    ):
        self._base = base
        self._portfolio_controllers = portfolio_controllers
        self._per_asset_controllers = per_asset_controllers
        self._portfolio_signal = portfolio_signal
        self._per_asset_signals = per_asset_signals

    @property
    def weights(self) -> np.ndarray:
        return self._base.weights

    @property
    def asset_names(self) -> tuple[str, ...]:
        return self._base.asset_names

    @property
    def base_forecast(self) -> PortfolioForecast:
        return self._base

    @property
    def portfolio_controllers(self) -> Mapping[float, ConformalMethod]:
        return self._portfolio_controllers

    @property
    def per_asset_controllers(self) -> Mapping[str, Mapping[float, ConformalMethod]]:
        return self._per_asset_controllers

    @property
    def portfolio_signal(self) -> tuple[float, float, float]:
        return self._portfolio_signal

    @property
    def per_asset_signals(self) -> Mapping[str, tuple[float, float, float]]:
        return self._per_asset_signals

    def portfolio_pnl(self) -> Distribution:
        mu, q_lo, q_hi = self._portfolio_signal
        return CalibratedDistribution(
            base=self._base.portfolio_pnl(),
            controllers=self._portfolio_controllers,
            forecast_mean=mu,
            q_lo=q_lo,
            q_hi=q_hi,
        )

    def asset_pnl(self, asset: str) -> Distribution:
        ctrls = self._per_asset_controllers.get(asset, {})
        if not ctrls:
            return self._base.asset_pnl(asset)
        mu, q_lo, q_hi = self._per_asset_signals[asset]
        return CalibratedDistribution(
            base=self._base.asset_pnl(asset),
            controllers=ctrls,
            forecast_mean=mu,
            q_lo=q_lo,
            q_hi=q_hi,
        )

    def joint_sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._base.joint_sample(n, rng)


class ConformalCalibrator:
    """Forecaster wrapper adding portfolio + per-asset conformal controllers.

    Adds observe_realized() (push realised values into controllers) and
    controller_state() (per-alpha and per-(alpha, asset) snapshot).
    """

    def __init__(
        self,
        base: Forecaster,
        portfolio_controllers: Mapping[float, ConformalMethod],
        per_asset_controller_factory: Callable[[float, str], ConformalMethod] | None = None,
    ):
        self._base = base
        self._portfolio_controllers = dict(portfolio_controllers)
        self._per_asset_factory = per_asset_controller_factory
        self._per_asset_controllers: dict[str, dict[float, ConformalMethod]] = {}
        self._warmup_done = False
        self._weights: np.ndarray | None = None
        self._latest_returns: pd.DataFrame | None = None
        self._asset_names: tuple[str, ...] = ()
        self._cached_portfolio_signal: tuple[float, float, float] | None = None
        self._cached_per_asset_signals: dict[str, tuple[float, float, float]] = {}

    @property
    def base_forecaster(self) -> Forecaster:
        return self._base

    @property
    def portfolio_controllers(self) -> Mapping[float, ConformalMethod]:
        return self._portfolio_controllers

    @property
    def per_asset_controllers(self) -> Mapping[str, Mapping[float, ConformalMethod]]:
        return self._per_asset_controllers

    @property
    def asset_names(self) -> tuple[str, ...]:
        return self._asset_names

    @property
    def portfolio_signal(self) -> tuple[float, float, float]:
        if self._cached_portfolio_signal is None:
            raise RuntimeError("portfolio_signal is undefined before the first predict()")
        return self._cached_portfolio_signal

    @property
    def per_asset_signals(self) -> Mapping[str, tuple[float, float, float]]:
        return self._cached_per_asset_signals

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done

    def fit(self, returns: pd.DataFrame) -> None:
        self._base.fit(returns)
        self._latest_returns = returns
        self._ensure_per_asset_controllers(returns.columns)

    def _ensure_per_asset_controllers(self, columns) -> None:
        if not self._asset_names:
            self._asset_names = tuple(columns)
            if self._per_asset_factory is not None:
                for asset in self._asset_names:
                    self._per_asset_controllers[asset] = {
                        float(alpha): self._per_asset_factory(float(alpha), asset)
                        for alpha in self._portfolio_controllers
                    }

    def predict(self, weights: np.ndarray) -> PortfolioForecast:
        self._weights = np.asarray(weights, dtype=float)
        if not self._warmup_done:
            self._warmup_controllers(self._latest_returns)
            self._warmup_done = True
        base_forecast = self._base.predict(self._weights)

        bp = base_forecast.portfolio_pnl()
        self._cached_portfolio_signal = (
            float(bp.quantile(0.5)), float(bp.quantile(0.05)), float(bp.quantile(0.95)),
        )
        self._cached_per_asset_signals = {}
        for asset in base_forecast.asset_names:
            ad = base_forecast.asset_pnl(asset)
            self._cached_per_asset_signals[asset] = (
                float(ad.quantile(0.5)), float(ad.quantile(0.05)), float(ad.quantile(0.95)),
            )

        return CalibratedPortfolioForecast(
            base=base_forecast,
            portfolio_controllers=self._portfolio_controllers,
            per_asset_controllers=self._per_asset_controllers,
            portfolio_signal=self._cached_portfolio_signal,
            per_asset_signals=dict(self._cached_per_asset_signals),
        )

    def _warmup_controllers(self, returns: pd.DataFrame) -> None:
        bf = self._base.predict(self._weights)
        bp = bf.portfolio_pnl()
        portfolio_mu = float(bp.quantile(0.5))
        portfolio_qlo = float(bp.quantile(0.05))
        portfolio_qhi = float(bp.quantile(0.95))

        n = len(returns)
        forecast_arr = np.full(n, portfolio_mu)
        realized_portfolio = (returns.to_numpy() @ self._weights).astype(float)
        q_lo_arr = np.full(n, portfolio_qlo)
        q_hi_arr = np.full(n, portfolio_qhi)
        for ctrl in self._portfolio_controllers.values():
            ctrl.fit(forecast_arr, realized_portfolio,
                     quantile_lo=q_lo_arr, quantile_hi=q_hi_arr)

        for k, asset in enumerate(self._asset_names):
            asset_ctrls = self._per_asset_controllers.get(asset)
            if not asset_ctrls:
                continue
            ad = bf.asset_pnl(asset)
            asset_mu = float(ad.quantile(0.5))
            asset_qlo = float(ad.quantile(0.05))
            asset_qhi = float(ad.quantile(0.95))
            forecast_a = np.full(n, asset_mu)
            realized_a = returns.iloc[:, k].to_numpy(dtype=float)
            qlo_a = np.full(n, asset_qlo)
            qhi_a = np.full(n, asset_qhi)
            for ctrl in asset_ctrls.values():
                ctrl.fit(forecast_a, realized_a, quantile_lo=qlo_a, quantile_hi=qhi_a)

    def observe_realized(
        self,
        realized: float,
        realized_per_asset: Mapping[str, float] | None = None,
    ) -> None:
        if self._cached_portfolio_signal is None:
            raise RuntimeError("observe_realized called before predict")
        mu, qlo, qhi = self._cached_portfolio_signal
        f_arr = np.array([mu]); a_arr = np.array([float(realized)])
        qlo_arr = np.array([qlo]); qhi_arr = np.array([qhi])
        for ctrl in self._portfolio_controllers.values():
            ctrl.update(f_arr, a_arr, quantile_lo=qlo_arr, quantile_hi=qhi_arr)

        if realized_per_asset and self._per_asset_controllers:
            for asset, ctrls in self._per_asset_controllers.items():
                if asset not in realized_per_asset:
                    continue
                a_mu, a_qlo, a_qhi = self._cached_per_asset_signals[asset]
                af_arr = np.array([a_mu])
                aa_arr = np.array([float(realized_per_asset[asset])])
                aqlo_arr = np.array([a_qlo]); aqhi_arr = np.array([a_qhi])
                for ctrl in ctrls.values():
                    ctrl.update(af_arr, aa_arr,
                               quantile_lo=aqlo_arr, quantile_hi=aqhi_arr)

    def controller_state(self) -> dict[float, dict[str, Any]]:
        """Per-alpha portfolio-level state. Per-asset state is in per_asset_controller_state()."""
        return {alpha: ctrl.state() for alpha, ctrl in self._portfolio_controllers.items()}

    def per_asset_controller_state(self) -> dict[tuple[float, str], dict[str, Any]]:
        out: dict[tuple[float, str], dict[str, Any]] = {}
        for asset, ctrls in self._per_asset_controllers.items():
            for alpha, ctrl in ctrls.items():
                if hasattr(ctrl, "state"):
                    out[(alpha, asset)] = ctrl.state()
        return out

    def warmup_walk(
        self,
        returns_full: pd.DataFrame,
        weights: np.ndarray,
        fit_window: int,
        n_steps: int,
    ) -> None:
        """OOS warmup: walk for n_steps without emitting output so the residual deques contain only OOS residuals."""
        self._weights = np.asarray(weights, dtype=float)
        self._ensure_per_asset_controllers(returns_full.columns)
        for i in range(fit_window, fit_window + n_steps):
            window = returns_full.iloc[i - fit_window : i]
            self._base.fit(window)
            bf = self._base.predict(self._weights)
            bp = bf.portfolio_pnl()
            portfolio_mu = float(bp.quantile(0.5))
            portfolio_qlo = float(bp.quantile(0.05))
            portfolio_qhi = float(bp.quantile(0.95))
            realized_portfolio = float(returns_full.iloc[i].to_numpy() @ self._weights)

            f_arr = np.array([portfolio_mu]); a_arr = np.array([realized_portfolio])
            qlo_arr = np.array([portfolio_qlo]); qhi_arr = np.array([portfolio_qhi])
            for ctrl in self._portfolio_controllers.values():
                if not ctrl.is_fitted:
                    ctrl.fit(f_arr, a_arr, quantile_lo=qlo_arr, quantile_hi=qhi_arr)
                else:
                    ctrl.update(f_arr, a_arr, quantile_lo=qlo_arr, quantile_hi=qhi_arr)

            for k, asset in enumerate(self._asset_names):
                asset_ctrls = self._per_asset_controllers.get(asset)
                if not asset_ctrls:
                    continue
                ad = bf.asset_pnl(asset)
                a_mu = float(ad.quantile(0.5))
                a_qlo = float(ad.quantile(0.05))
                a_qhi = float(ad.quantile(0.95))
                af = np.array([a_mu]); aa = np.array([float(returns_full.iloc[i, k])])
                aqlo = np.array([a_qlo]); aqhi = np.array([a_qhi])
                for ctrl in asset_ctrls.values():
                    if not ctrl.is_fitted:
                        ctrl.fit(af, aa, quantile_lo=aqlo, quantile_hi=aqhi)
                    else:
                        ctrl.update(af, aa, quantile_lo=aqlo, quantile_hi=aqhi)

        self._warmup_done = True

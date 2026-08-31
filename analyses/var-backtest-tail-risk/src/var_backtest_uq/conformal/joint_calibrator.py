"""ES feedback loop layered on top of the VaR conformal PID.

Decoupled from a strict Fissler-Ziegel (2016) joint update: VaR runs the
unchanged ConformalPIDController; ES applies a per-alpha multiplicative
scale driven by a clipped PID on the realised tail-mean residual on
breach days. Interface-compatible with a strict FZ updater if swapped
in later.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .protocols import ConformalMethod

from ..forecasters.base import Distribution, Forecaster, PortfolioForecast
from .calibrator import (
    CalibratedDistribution,
    CalibratedPortfolioForecast,
    ConformalCalibrator,
    _es_from_base_below,
)


class EsCorrector:
    """Per-alpha multiplicative ES scale, updated on breach days only.

    Residual e_t = realised/|ES_base| + 1, which is 0 when realised matches
    predicted ES and < 0 when realised is more extreme. PID drives a
    log-space scale, bounded to [s_min, s_max]. Update is gated on breach,
    matching the AS Z2 conditioning.
    """

    def __init__(
        self,
        alpha: float,
        Kp: float = 0.15,
        Ki: float = 0.01,
        Kd: float = 0.05,
        integral_clip: float = 25.0,
        scale_bounds: tuple[float, float] = (0.5, 3.0),
    ) -> None:
        if not 0.0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5); got {alpha}")
        self._lo, self._hi = scale_bounds
        if not 0.0 < self._lo < self._hi:
            raise ValueError(f"scale_bounds must satisfy 0 < lo < hi; got {scale_bounds}")
        self._alpha = alpha
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._integral_clip = integral_clip
        self._log_scale = 0.0
        self._integral = 0.0
        self._prev_error = 0.0
        self._n_updates = 0

    @property
    def scale(self) -> float:
        return float(np.clip(math.exp(self._log_scale), self._lo, self._hi))

    def update(self, realised: float, base_es: float, breached: bool) -> None:
        if not breached:
            return
        es_mag = abs(base_es)
        if es_mag < 1e-12:
            return
        # error < 0 (realised more extreme than predicted ES) grows the scale.
        error = realised / es_mag + 1.0
        self._integral = float(np.clip(
            self._integral + error, -self._integral_clip, self._integral_clip,
        ))
        derivative = error - self._prev_error
        update_signal = self._Kp * error + self._Ki * self._integral + self._Kd * derivative
        new_log = self._log_scale - update_signal
        self._log_scale = float(np.clip(new_log, math.log(self._lo), math.log(self._hi)))
        self._prev_error = error
        self._n_updates += 1

    def state(self) -> dict[str, Any]:
        return {
            "es_alpha_target": self._alpha,
            "es_scale": self.scale,
            "es_log_scale": self._log_scale,
            "es_integral": self._integral,
            "es_prev_error": self._prev_error,
            "es_n_updates": self._n_updates,
        }


class JointCalibratedDistribution(CalibratedDistribution):
    """Overrides es(p) with the per-alpha ES correction; VaR calibration is unchanged."""

    def __init__(
        self,
        base: Distribution,
        controllers: Mapping[float, ConformalMethod],
        es_correctors: Mapping[float, EsCorrector],
        forecast_mean: float,
        q_lo: float,
        q_hi: float,
    ):
        super().__init__(
            base=base, controllers=controllers,
            forecast_mean=forecast_mean, q_lo=q_lo, q_hi=q_hi,
        )
        self._es_correctors: dict[float, EsCorrector] = dict(es_correctors)

    def es(self, p: float) -> float:
        base_es = _es_from_base_below(self._base, self.quantile(p))
        corrector = self._es_correctors.get(p)
        if corrector is None:
            return base_es
        return corrector.scale * base_es

    def base_es(self, p: float) -> float:
        """Uncorrected ES at the calibrated VaR. Lets the corrector see the same number as the runner."""
        return _es_from_base_below(self._base, self.quantile(p))


class JointCalibratedPortfolioForecast(CalibratedPortfolioForecast):
    """Routes portfolio_pnl() through JointCalibratedDistribution; result is cached so runner and observe_realized share one instance."""

    def __init__(
        self,
        base: PortfolioForecast,
        portfolio_controllers: Mapping[float, ConformalMethod],
        per_asset_controllers: Mapping[str, Mapping[float, ConformalMethod]],
        portfolio_signal: tuple[float, float, float],
        per_asset_signals: Mapping[str, tuple[float, float, float]],
        es_correctors: Mapping[float, EsCorrector],
    ):
        super().__init__(
            base=base,
            portfolio_controllers=portfolio_controllers,
            per_asset_controllers=per_asset_controllers,
            portfolio_signal=portfolio_signal,
            per_asset_signals=per_asset_signals,
        )
        self._es_correctors = dict(es_correctors)
        self._cached_pnl: JointCalibratedDistribution | None = None

    def portfolio_pnl(self) -> Distribution:
        if self._cached_pnl is None:
            mu, q_lo, q_hi = self._portfolio_signal
            self._cached_pnl = JointCalibratedDistribution(
                base=self._base.portfolio_pnl(),
                controllers=self._portfolio_controllers,
                es_correctors=self._es_correctors,
                forecast_mean=mu,
                q_lo=q_lo,
                q_hi=q_hi,
            )
        return self._cached_pnl


class JointVarEsCalibrator:
    """ConformalCalibrator + per-alpha EsCorrector.

    observe_realized() reads (realised, base_es, breach) BEFORE updating
    the VaR controllers so the corrector sees the same numbers the runner
    did at predict time.
    """

    def __init__(
        self,
        base: Forecaster,
        portfolio_controllers: Mapping[float, ConformalMethod],
        per_asset_controller_factory: Callable[[float, str], ConformalMethod] | None = None,
        es_corrector_factory: Callable[[float], EsCorrector] | None = None,
    ):
        self._var_calibrator = ConformalCalibrator(
            base=base,
            portfolio_controllers=portfolio_controllers,
            per_asset_controller_factory=per_asset_controller_factory,
        )
        ec_factory = es_corrector_factory or (lambda a: EsCorrector(alpha=a))
        self._es_correctors: dict[float, EsCorrector] = {
            a: ec_factory(a) for a in portfolio_controllers
        }
        self._last_forecast: JointCalibratedPortfolioForecast | None = None

    def fit(self, returns: pd.DataFrame) -> None:
        self._var_calibrator.fit(returns)

    def predict(self, weights: np.ndarray) -> PortfolioForecast:
        base_calibrated: CalibratedPortfolioForecast = self._var_calibrator.predict(weights)
        forecast = JointCalibratedPortfolioForecast(
            base=base_calibrated.base_forecast,
            portfolio_controllers=base_calibrated.portfolio_controllers,
            per_asset_controllers=base_calibrated.per_asset_controllers,
            portfolio_signal=self._var_calibrator.portfolio_signal,
            per_asset_signals=self._var_calibrator.per_asset_signals,
            es_correctors=self._es_correctors,
        )
        self._last_forecast = forecast
        return forecast

    @property
    def current_forecast(self) -> JointCalibratedPortfolioForecast:
        if self._last_forecast is None:
            raise RuntimeError("current_forecast is undefined before the first predict()")
        return self._last_forecast

    @property
    def es_correctors(self) -> dict[float, EsCorrector]:
        return self._es_correctors

    def observe_realized(
        self,
        realized: float,
        realized_per_asset: Mapping[str, float] | None = None,
    ) -> None:
        # Read corrector inputs before updating VaR controllers; otherwise
        # the cached quantile shifts under us and the corrector sees post-update numbers.
        if self._last_forecast is None:
            raise RuntimeError("observe_realized called before predict")

        joint_pnl: JointCalibratedDistribution = self._last_forecast.portfolio_pnl()  # type: ignore[assignment]
        for alpha_f, corrector in self._es_correctors.items():
            var_now = joint_pnl.quantile(alpha_f)
            base_es_now = joint_pnl.base_es(alpha_f)
            breached = realized < var_now
            corrector.update(realized, base_es_now, breached)

        self._var_calibrator.observe_realized(realized, realized_per_asset)

    def controller_state(self) -> dict[float, dict[str, Any]]:
        var_state = self._var_calibrator.controller_state()
        return {
            alpha: {**var_st, **self._es_correctors[alpha].state()}
            for alpha, var_st in var_state.items()
        }

    def per_asset_controller_state(self) -> dict[tuple[float, str], dict[str, Any]]:
        # ES correction is portfolio-level only; per-asset state is unchanged.
        return self._var_calibrator.per_asset_controller_state()

    def warmup_walk(
        self,
        returns_full: pd.DataFrame,
        weights: np.ndarray,
        fit_window: int,
        n_steps: int,
    ) -> None:
        # ES correctors cold-start at scale=1.0; first breaches move them.
        self._var_calibrator.warmup_walk(returns_full, weights, fit_window, n_steps)

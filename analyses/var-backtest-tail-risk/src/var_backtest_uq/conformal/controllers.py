"""Online conformal-PID for a single lower-tail quantile.

Follows Angelopoulos, Candès & Tibshirani (2024), with vol-EMA
normalisation (Bhatnagar et al. 2023) and MDN-style width scaling.
Default gains were not tuned on the dashboard backtest.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .protocols import PredictiveDistribution


class OneSidedQuantileDistribution(PredictiveDistribution):
    """Single calibrated tail quantile; quantile(p) returns the stored bound for any p."""

    def __init__(self, quantile_value: float, alpha: float, forecast: float):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        self.quantile_value = quantile_value
        self.alpha = alpha
        self.forecast = forecast

    def quantile(self, p: float) -> np.ndarray:
        return np.array([self.quantile_value])

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.where(x < self.quantile_value, 0.0, 1.0)
        out = np.where(np.isclose(x, self.quantile_value), self.alpha, out)
        return out


class ConformalPIDController:
    """Drives P(actual < forecast + offset) toward a single target alpha."""

    def __init__(
        self,
        alpha: float = 0.025,
        window_size: int = 500,
        Kp: float = 0.03,
        Ki: float = 0.0005,
        Kd: float = 0.003,
        integral_clip: float = 50.0,
        use_vol_normalization: bool = True,
        vol_ema_span: float = 168.0,
        vol_floor: float = 1e-3,
        smooth_alpha: float = 1.0,
    ):
        if not 0.0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5); got {alpha}")
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError(f"smooth_alpha must be in (0, 1]; got {smooth_alpha}")
        self._alpha_target = alpha
        self._window_size = window_size
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._integral_clip = integral_clip

        self._use_vol = use_vol_normalization
        self._vol_ema_alpha = 2.0 / (vol_ema_span + 1.0)
        self._vol_floor = vol_floor
        self._vol_ema = 0.0
        self._vol_ema_initialized = False

        self._smooth_alpha = smooth_alpha
        self._smoothed_offset: float | None = None

        self._residuals: deque[float] = deque(maxlen=window_size)
        self._mdn_widths: deque[float] = deque(maxlen=window_size)

        self._base_quantile = self._alpha_target
        self._adjusted_quantile = self._base_quantile
        self._integral = 0.0
        self._prev_error = 0.0
        self._fitted = False

    def _update_vol_ema(self, residual: float) -> None:
        a = abs(residual)
        if not self._vol_ema_initialized:
            self._vol_ema = a
            self._vol_ema_initialized = True
        else:
            self._vol_ema = self._vol_ema_alpha * a + (1 - self._vol_ema_alpha) * self._vol_ema

    def _vol_divisor(self) -> float:
        return max(self._vol_ema, self._vol_floor) if self._use_vol else 1.0

    def _current_scale(self, mdn_lo, mdn_hi) -> tuple[float, float | None]:
        if mdn_lo is None or mdn_hi is None:
            return 1.0, None
        hi = float(np.asarray(mdn_hi).ravel()[0])
        lo = float(np.asarray(mdn_lo).ravel()[0])
        mdn_width = max(hi - lo, 1e-6)
        if len(self._mdn_widths) <= 10:
            return 1.0, mdn_width
        median_width = float(np.median(list(self._mdn_widths)))
        scale = max(0.3, min(3.0, mdn_width / max(median_width, 1e-6)))
        return scale, mdn_width

    def _raw_offset(self, mdn_lo, mdn_hi) -> tuple[float, float | None]:
        residuals_arr = np.array(self._residuals)
        base = np.percentile(residuals_arr, self._adjusted_quantile * 100)
        vol_out = self._vol_divisor()
        base = base * vol_out
        scale, mdn_width = self._current_scale(mdn_lo, mdn_hi)
        return float(base * scale), mdn_width

    def _smooth_and_store(self, raw_offset: float) -> float:
        a = self._smooth_alpha
        if self._smoothed_offset is None:
            self._smoothed_offset = raw_offset
        else:
            self._smoothed_offset = a * raw_offset + (1 - a) * self._smoothed_offset
        return self._smoothed_offset

    def fit(self, forecasts: np.ndarray, actuals: np.ndarray, **kwargs) -> None:
        forecasts = np.asarray(forecasts, dtype=float).ravel()
        actuals = np.asarray(actuals, dtype=float).ravel()
        residuals = actuals - forecasts
        for r in residuals:
            self._update_vol_ema(r)
            v = self._vol_divisor()
            self._residuals.append(r / v if self._use_vol else r)
        q_lo = kwargs.get("quantile_lo")
        q_hi = kwargs.get("quantile_hi")
        if q_lo is not None and q_hi is not None:
            q_lo = np.asarray(q_lo, dtype=float).ravel()
            q_hi = np.asarray(q_hi, dtype=float).ravel()
            for lo, hi in zip(q_lo, q_hi):
                self._mdn_widths.append(max(hi - lo, 1e-6))
        self._fitted = True

    def predict(self, forecast: np.ndarray, **kwargs) -> PredictiveDistribution:
        if not self._fitted:
            raise RuntimeError("ConformalPIDController has not been fitted yet")
        forecast_val = float(np.asarray(forecast).ravel()[0])
        raw_offset, mdn_width = self._raw_offset(
            kwargs.get("quantile_lo"), kwargs.get("quantile_hi"),
        )
        if mdn_width is not None:
            self._mdn_widths.append(mdn_width)
        smoothed = self._smooth_and_store(raw_offset)
        return OneSidedQuantileDistribution(
            quantile_value=forecast_val + smoothed,
            alpha=self._alpha_target,
            forecast=forecast_val,
        )

    def update(self, forecast: np.ndarray, actual: np.ndarray, **kwargs) -> None:
        forecast_val = float(np.asarray(forecast).ravel()[0])
        actual_val = float(np.asarray(actual).ravel()[0])
        residual = actual_val - forecast_val

        # PID targets the raw bound; smoothing is downstream only.
        raw_offset, mdn_width = self._raw_offset(
            kwargs.get("quantile_lo"), kwargs.get("quantile_hi"),
        )
        if mdn_width is not None:
            self._mdn_widths.append(mdn_width)
        var_bound = forecast_val + raw_offset

        self._update_vol_ema(residual)
        v_store = self._vol_divisor()
        self._residuals.append(residual / v_store if self._use_vol else residual)

        miss = 1.0 if actual_val < var_bound else 0.0

        # Positive error (under-coverage) shrinks adjusted_quantile, deepening VaR.
        error = miss - self._alpha_target
        self._integral = float(np.clip(
            self._integral + error, -self._integral_clip, self._integral_clip,
        ))
        derivative = error - self._prev_error

        self._adjusted_quantile = float(np.clip(
            self._base_quantile
            - (self._Kp * error + self._Ki * self._integral + self._Kd * derivative),
            0.001, 0.499,
        ))
        self._prev_error = error

    def state(self) -> dict:
        return {
            "alpha_target": self._alpha_target,
            "adjusted_quantile": self._adjusted_quantile,
            "base_quantile": self._base_quantile,
            "integral": self._integral,
            "prev_error": self._prev_error,
            "smoothed_offset": self._smoothed_offset,
            "vol_ema": self._vol_ema if self._vol_ema_initialized else None,
            "n_residuals": len(self._residuals),
        }

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_residuals(self) -> int:
        return len(self._residuals)

    @property
    def alpha_target(self) -> float:
        return self._alpha_target

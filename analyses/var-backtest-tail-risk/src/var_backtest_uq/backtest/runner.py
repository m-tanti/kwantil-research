"""Walk-forward VaR/ES backtest orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ..data.portfolio import log_returns, portfolio_returns
from ..forecasters.base import Distribution, Forecaster
from ..forecasters.distributional import excess_kurtosis


def _extract_loc_scale_kurt(dist: Distribution) -> tuple[float, float, float]:
    """(loc, scale, excess_kurtosis) for the AS bootstrap H0.

    Prefers stored samples; otherwise mu/sigma + zero kurtosis (Gaussian);
    otherwise a 4096-MC fallback.
    """
    mu = getattr(dist, "mu", None)
    sigma = getattr(dist, "sigma", None)
    samples = getattr(dist, "samples", None)
    if samples is not None and len(samples):
        s = np.asarray(samples, dtype=float)
        loc = float(s.mean()) if mu is None else float(mu)
        scale = float(s.std()) if sigma is None else float(sigma)
        return loc, scale, excess_kurtosis(s)
    if mu is not None and sigma is not None:
        return float(mu), float(sigma), 0.0
    s = np.asarray(dist.sample(4096), dtype=float)
    return float(s.mean()), float(s.std()), excess_kurtosis(s)


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame
    realized: pd.Series
    breaches: pd.DataFrame
    per_asset: pd.DataFrame = field(default_factory=pd.DataFrame)
    controller_state: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)


class BacktestRunner:
    """Walk-forward backtest of one or more methods on a single portfolio."""

    def __init__(
        self,
        methods: Mapping[str, Forecaster],
        portfolio_weights: Mapping[str, float],
        fit_window: int = 500,
        alpha_targets: tuple[float, ...] = (0.025, 0.01),
        warmup_steps: int = 0,
    ):
        if not methods:
            raise ValueError("BacktestRunner requires at least one method")
        if not portfolio_weights:
            raise ValueError("portfolio_weights cannot be empty")
        if fit_window < 30:
            raise ValueError(f"fit_window={fit_window} is too short for stable EWMA vol")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
        self._methods = dict(methods)
        self._weights_map = dict(portfolio_weights)
        self._fit_window = fit_window
        self._alpha_targets = tuple(alpha_targets)
        self._warmup_steps = warmup_steps

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        asset_order = list(self._weights_map.keys())
        missing = [a for a in asset_order if a not in prices.columns]
        if missing:
            raise ValueError(f"prices missing columns for assets: {missing}")
        prices = prices[asset_order]

        returns = log_returns(prices)
        weights = np.array([self._weights_map[a] for a in asset_order], dtype=float)
        realized = portfolio_returns(returns, weights)

        forecast_rows: list[dict] = []
        breach_rows: list[dict] = []
        controller_rows: list[dict] = []
        per_asset_rows: list[dict] = []

        n = len(returns)
        W = self._fit_window
        if n <= W:
            raise ValueError(
                f"have {n} returns but fit_window={W}; need at least W+1 to take one step"
            )

        # OOS warmup: stateful methods walk `warmup_walk` without emitting
        # output so reporting only sees post-warmup residuals.
        warmup_end = W + min(self._warmup_steps, n - W - 1)
        if self._warmup_steps > 0:
            for method in self._methods.values():
                if hasattr(method, "warmup_walk"):
                    method.warmup_walk(
                        returns_full=returns,
                        weights=weights,
                        fit_window=W,
                        n_steps=warmup_end - W,
                    )

        for i in range(warmup_end, n):
            window_returns = returns.iloc[i - W : i]
            date = returns.index[i]
            realized_pnl = float(realized.iloc[i])

            for method_name, method in self._methods.items():
                method.fit(window_returns)
                forecast = method.predict(weights)
                pnl_dist = forecast.portfolio_pnl()

                pnl_mu, pnl_sigma, pnl_kurt = _extract_loc_scale_kurt(pnl_dist)
                for alpha in self._alpha_targets:
                    var = float(pnl_dist.quantile(alpha))
                    es = float(pnl_dist.es(alpha))
                    breached = realized_pnl < var

                    forecast_rows.append({
                        "date": date,
                        "method": method_name,
                        "alpha": alpha,
                        "var": var,
                        "es": es,
                        "pnl_mu": pnl_mu,
                        "pnl_sigma": pnl_sigma,
                        "pnl_kurt_excess": pnl_kurt,
                    })
                    breach_rows.append({
                        "date": date,
                        "method": method_name,
                        "alpha": alpha,
                        "breached": bool(breached),
                    })

                # Per-asset diagnostic uses the base marginal; calibration is portfolio-level.
                for j, asset in enumerate(asset_order):
                    asset_dist = forecast.asset_pnl(asset)
                    realized_asset = float(returns.iloc[i, j])
                    for alpha in self._alpha_targets:
                        a_var = float(asset_dist.quantile(alpha))
                        a_es = float(asset_dist.es(alpha))
                        per_asset_rows.append({
                            "date": date,
                            "method": method_name,
                            "alpha": alpha,
                            "asset": asset,
                            "var": a_var,
                            "es": a_es,
                            "breached": bool(realized_asset < a_var),
                            "realized": realized_asset,
                        })

                if hasattr(method, "observe_realized"):
                    realized_per_asset = {
                        a: float(returns.iloc[i, j]) for j, a in enumerate(asset_order)
                    }
                    method.observe_realized(realized_pnl, realized_per_asset)
                if hasattr(method, "controller_state"):
                    state_per_alpha = method.controller_state()
                    for alpha, st in state_per_alpha.items():
                        controller_rows.append({
                            "date": date,
                            "method": method_name,
                            "alpha": float(alpha),
                            "asset": "__portfolio__",
                            **st,
                        })
                if hasattr(method, "per_asset_controller_state"):
                    per_asset_state = method.per_asset_controller_state()
                    for (alpha, asset), st in per_asset_state.items():
                        controller_rows.append({
                            "date": date,
                            "method": method_name,
                            "alpha": float(alpha),
                            "asset": asset,
                            **st,
                        })

        forecasts_df = pd.DataFrame(forecast_rows)
        breaches_df = pd.DataFrame(breach_rows)
        controller_df = pd.DataFrame(controller_rows)
        per_asset_df = pd.DataFrame(per_asset_rows)
        realized_eval = realized.iloc[warmup_end:].copy()
        realized_eval.name = "realized_pnl"

        metadata = {
            "weights": dict(zip(asset_order, weights.tolist())),
            "fit_window": W,
            "warmup_steps": self._warmup_steps,
            "alpha_targets": list(self._alpha_targets),
            "assets": asset_order,
            "methods": list(self._methods.keys()),
            "n_steps": len(realized_eval),
            "first_date": str(realized_eval.index[0].date()) if len(realized_eval) else None,
            "last_date": str(realized_eval.index[-1].date()) if len(realized_eval) else None,
        }
        return BacktestResult(
            forecasts=forecasts_df,
            realized=realized_eval,
            breaches=breaches_df,
            per_asset=per_asset_df,
            controller_state=controller_df,
            metadata=metadata,
        )

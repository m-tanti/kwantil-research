"""Pipeline entry points. `scripts/run_backtest.py` is an argparse shim over `run_backtest_pipeline()`."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .backtest.runner import BacktestResult, BacktestRunner
from .backtest.stats import compute_stats
from .conformal.calibrator import ConformalCalibrator
from .conformal.controllers import ConformalPIDController
from .conformal.joint_calibrator import EsCorrector, JointVarEsCalibrator
from .data.fetch import fetch_prices
from .forecasters.base import Forecaster
from .forecasters.distributional import DistributionalForecaster
from .forecasters.historical import HistoricalSimulationForecaster
from .forecasters.parametric import ParametricGaussianForecaster
from .reporting.pdf import compile_report
from .reporting.snapshot import write_dashboard_payload


logger = logging.getLogger(__name__)


def _make_conformal_pid_controller(alpha: float) -> ConformalPIDController:
    return ConformalPIDController(
        alpha=alpha, window_size=500,
        use_vol_normalization=True, smooth_alpha=1.0,
    )


def make_conformal_pid_calibrator(
    refit_every: int, alpha_targets: tuple[float, ...], copula: str = "t",
) -> ConformalCalibrator:
    """t-GARCH base wrapped by per-alpha and per-asset PID controllers.

    `copula` is exposed so the copula arm of the de-confounding factorial can
    be run from the CLI rather than by editing this factory.
    """
    return ConformalCalibrator(
        base=DistributionalForecaster(refit_every=refit_every, dist="t", copula=copula),
        portfolio_controllers={a: _make_conformal_pid_controller(a) for a in alpha_targets},
        per_asset_controller_factory=lambda a, _asset: _make_conformal_pid_controller(a),
    )


def make_joint_var_es_calibrator(
    refit_every: int, alpha_targets: tuple[float, ...], copula: str = "t",
) -> JointVarEsCalibrator:
    """Conformal-PID + per-alpha EsCorrector."""
    return JointVarEsCalibrator(
        base=DistributionalForecaster(refit_every=refit_every, dist="t", copula=copula),
        portfolio_controllers={a: _make_conformal_pid_controller(a) for a in alpha_targets},
        per_asset_controller_factory=lambda a, _asset: _make_conformal_pid_controller(a),
        es_corrector_factory=lambda a: EsCorrector(alpha=a),
    )


METHOD_FACTORIES: Mapping[str, "callable[..., Forecaster]"] = {
    "parametric": lambda **_: ParametricGaussianForecaster(lam=0.94),
    "historical": lambda **_: HistoricalSimulationForecaster(lookback=250),
    "distributional": lambda refit_every=20, copula="t", **_: DistributionalForecaster(
        refit_every=refit_every, copula=copula,
    ),
    "conformal_pid": lambda refit_every=20, alpha_targets=(0.025, 0.01), copula="t", **_:
        make_conformal_pid_calibrator(refit_every, alpha_targets, copula=copula),
    "conformal_pid_es": lambda refit_every=20, alpha_targets=(0.025, 0.01), copula="t", **_:
        make_joint_var_es_calibrator(refit_every, alpha_targets, copula=copula),
}


def resolve_methods(spec: str | list[str]) -> list[str]:
    """Resolve `"all"`, a single name, a comma-separated string, or a list."""
    if isinstance(spec, list):
        items = list(spec)
    elif spec == "all":
        return list(METHOD_FACTORIES.keys())
    else:
        items = [m.strip() for m in spec.split(",") if m.strip()]
    unknown = [m for m in items if m not in METHOD_FACTORIES]
    if unknown:
        raise ValueError(f"unknown methods: {unknown} (known: {list(METHOD_FACTORIES)})")
    return items


@dataclass
class PipelineResult:
    backtest_result: BacktestResult
    summary_df: pd.DataFrame
    details_df: pd.DataFrame
    artifacts: dict[str, Path] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def details_long(result: BacktestResult) -> pd.DataFrame:
    """Long-format per-(date, method, alpha) with realised P&L attached."""
    fc = result.forecasts.copy()
    br = result.breaches.set_index(["date", "method", "alpha"])["breached"]
    fc["breached"] = fc.set_index(["date", "method", "alpha"]).index.map(br).astype(bool)
    fc["realized_pnl"] = fc["date"].map(result.realized)
    cols = ["date", "method", "alpha", "var", "es", "breached", "realized_pnl"]
    # Carry the predicted-marginal moments when the forecaster supplies them.
    # Without these the AS null cannot be re-specified after the fact, which
    # makes the Z2 sensitivity in manual section 8 impossible to reproduce.
    cols += [c for c in ("pnl_mu", "pnl_sigma", "pnl_kurt_excess") if c in fc.columns]
    return fc[cols]


def run_backtest_pipeline(
    *,
    method: str | list[str] = "all",
    tickers: list[str] | None = None,
    weights: list[float] | None = None,
    start: str | None = "2018-01-01",
    end: str | None = None,
    fit_window: int = 500,
    alpha_targets: tuple[float, ...] = (0.025, 0.01),
    cache_dir: Path | str = Path("data/cache"),
    output_dir: Path | str = Path("output"),
    dashboard_dir: Path | str | None = None,
    write_dashboard: bool = True,
    write_pdf: bool = False,
    pdf_path: Path | str | None = None,
    refresh: bool = False,
    refit_every: int = 20,
    warmup_steps: int = 500,
    copula: str = "t",
) -> PipelineResult:
    """End-to-end var-backtest pipeline. Defaults match the dashboard config."""
    tickers = list(tickers) if tickers is not None else ["SPY", "TLT", "GLD"]
    weights = list(weights) if weights is not None else [1.0 / len(tickers)] * len(tickers)
    if len(weights) != len(tickers):
        raise ValueError(f"weights ({len(weights)}) != tickers ({len(tickers)})")
    if not np.isclose(sum(weights), 1.0, atol=1e-3):
        raise ValueError(f"weights must sum to 1.0; got {sum(weights):.4f}")
    weights_map = dict(zip(tickers, weights))
    method_names = resolve_methods(method)

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dashboard_dir is None:
        dashboard_dir = output_dir / "dashboard"
    else:
        dashboard_dir = Path(dashboard_dir)

    logger.info("Fetching %s (cache: %s)", tickers, cache_dir)
    prices = fetch_prices(tickers, cache_dir=cache_dir, refresh=refresh)
    logger.info(
        "Loaded %d daily rows %s -> %s",
        len(prices), prices.index[0].date(), prices.index[-1].date(),
    )

    methods = {
        name: METHOD_FACTORIES[name](
            refit_every=refit_every, alpha_targets=alpha_targets, copula=copula,
        )
        for name in method_names
    }
    runner = BacktestRunner(
        methods=methods,
        portfolio_weights=weights_map,
        fit_window=fit_window,
        alpha_targets=alpha_targets,
        warmup_steps=warmup_steps,
    )
    logger.info(
        "Walk-forward: methods=%s fit_window=%d alphas=%s",
        method_names, fit_window, alpha_targets,
    )
    t0 = time.time()
    result = runner.run(prices)
    elapsed = time.time() - t0
    logger.info("Walk-forward took %.1fs", elapsed)

    details = details_long(result)
    if start:
        details = details[details["date"] >= pd.Timestamp(start)]
    if end:
        details = details[details["date"] <= pd.Timestamp(end)]

    # Recompute stats on the trimmed window so headlines reflect the reporting period.
    trimmed_dates = pd.to_datetime(details["date"].unique())
    trimmed_per_asset = result.per_asset[result.per_asset["date"].isin(trimmed_dates)]
    trimmed_controller = result.controller_state
    if not trimmed_controller.empty:
        trimmed_controller = trimmed_controller[trimmed_controller["date"].isin(trimmed_dates)]
    # Preserve all forecast columns; compute_stats needs pnl_mu/pnl_sigma/pnl_kurt_excess.
    trim_mask = result.forecasts["date"].isin(trimmed_dates)
    trimmed_forecasts = result.forecasts[trim_mask].reset_index(drop=True)
    trimmed_breaches = result.breaches[result.breaches["date"].isin(trimmed_dates)].reset_index(drop=True)
    trimmed_result = BacktestResult(
        forecasts=trimmed_forecasts,
        breaches=trimmed_breaches,
        realized=result.realized.loc[trimmed_dates],
        per_asset=trimmed_per_asset,
        controller_state=trimmed_controller,
        metadata={**result.metadata, "start": start, "end": end},
    )
    summary = compute_stats(trimmed_result)

    controller_state = result.controller_state
    if not controller_state.empty:
        if start:
            controller_state = controller_state[controller_state["date"] >= pd.Timestamp(start)]
        if end:
            controller_state = controller_state[controller_state["date"] <= pd.Timestamp(end)]

    artifacts: dict[str, Path] = {}

    details_path = output_dir / "details.csv"
    summary_path = output_dir / "summary.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    artifacts["details_csv"] = details_path
    artifacts["summary_csv"] = summary_path
    logger.info("Wrote %d detail rows -> %s", len(details), details_path)
    logger.info("Wrote %d summary rows -> %s", len(summary), summary_path)

    if not controller_state.empty:
        cs_path = output_dir / "controller_state.csv"
        controller_state.to_csv(cs_path, index=False)
        artifacts["controller_state_csv"] = cs_path
        logger.info("Wrote %d controller-state rows -> %s", len(controller_state), cs_path)

    if write_dashboard:
        written = write_dashboard_payload(trimmed_result, dashboard_dir)
        artifacts.update({f"dashboard_{name}": path for name, path in written.items()})
        logger.info("Dashboard payload -> %s (%d artefacts)", dashboard_dir, len(written))

    if write_pdf:
        resolved_pdf_path = Path(pdf_path) if pdf_path else (output_dir / "validation_report.pdf")
        compile_report(trimmed_result, resolved_pdf_path)
        artifacts["validation_pdf"] = resolved_pdf_path
        logger.info("Validation PDF -> %s", resolved_pdf_path)

    return PipelineResult(
        backtest_result=trimmed_result,
        summary_df=summary,
        details_df=details,
        artifacts=artifacts,
        elapsed_seconds=elapsed,
    )


SUMMARY_PRINT_COLUMNS: tuple[str, ...] = (
    "method", "alpha", "n_obs", "n_breaches", "observed_rate",
    "kupiec_pvalue", "christoffersen_cc_pvalue",
    "basel_zone", "basel_n_breaches_window", "basel_total_multiplier",
    "as_z2_zone", "as_z2_statistic",
)

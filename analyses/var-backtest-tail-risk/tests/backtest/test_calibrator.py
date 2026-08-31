"""Tests for `ConformalCalibrator`, per-asset controllers, warmup, and
runner integration."""

import numpy as np

from var_backtest_uq.backtest.runner import BacktestRunner

from ..conftest import (
    make_conformal_calibrator,
    synthetic_prices,
    synthetic_returns,
)


def test_per_asset_controllers_distinct_from_portfolio():
    returns = synthetic_returns()
    cal = make_conformal_calibrator((0.025, 0.01))
    cal.fit(returns)
    cal.predict(np.array([1 / 3, 1 / 3, 1 / 3]))

    pf_state = cal.controller_state()
    pa_state = cal.per_asset_controller_state()

    assert set(pf_state.keys()) == {0.025, 0.01}
    # 3 assets × 2 alphas = 6 per-asset controllers.
    assert len(pa_state) == 6
    assert {k[1] for k in pa_state} == {"SPY", "TLT", "GLD"}


def test_warmup_walk_seeds_residual_deque_oos():
    returns = synthetic_returns(n=300)
    cal = make_conformal_calibrator((0.025,))
    cal.warmup_walk(
        returns_full=returns,
        weights=np.array([1 / 3, 1 / 3, 1 / 3]),
        fit_window=200,
        n_steps=50,
    )
    assert cal.warmup_done is True
    portfolio_ctrl = cal.portfolio_controllers[0.025]
    assert portfolio_ctrl.is_fitted is True
    assert portfolio_ctrl.n_residuals >= 50


def test_runner_warmup_steps_excludes_warmup_from_reporting():
    returns = synthetic_returns(n=600)
    prices = synthetic_prices(returns)
    cal = make_conformal_calibrator((0.025,))

    runner = BacktestRunner(
        methods={"conformal_pid": cal},
        portfolio_weights={"SPY": 1 / 3, "TLT": 1 / 3, "GLD": 1 / 3},
        fit_window=200,
        alpha_targets=(0.025,),
        warmup_steps=50,
    )
    result = runner.run(prices)

    # Reporting period: total returns is 599 (one less than prices); after
    # fit_window 200 + warmup 50, ~349 reporting steps remain.
    assert 300 <= len(result.realized) <= 360
    assert result.metadata["warmup_steps"] == 50


def test_runner_per_asset_breaches_recorded():
    returns = synthetic_returns(n=400)
    prices = synthetic_prices(returns)
    cal = make_conformal_calibrator((0.025,))
    runner = BacktestRunner(
        methods={"conformal_pid": cal},
        portfolio_weights={"SPY": 1 / 3, "TLT": 1 / 3, "GLD": 1 / 3},
        fit_window=200,
        alpha_targets=(0.025,),
        warmup_steps=20,
    )
    result = runner.run(prices)
    assert "asset" in result.per_asset.columns
    assert set(result.per_asset["asset"].unique()) == {"SPY", "TLT", "GLD"}

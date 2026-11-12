"""Smoke + behaviour tests for `JointVarEsCalibrator` and `EsCorrector`.

Cover the contract observed by `BacktestRunner` (fit / predict /
observe_realized / controller_state) and the directional behaviour of the
ES corrector under a deliberately-misspecified base distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from var_backtest_uq.conformal.joint_calibrator import (
    EsCorrector,
    JointCalibratedDistribution,
)

from ..conftest import make_joint_calibrator, synthetic_returns


# EsCorrector unit behaviours: directional, no-op, bound clamping.


def test_es_corrector_widens_when_realised_exceeds_predicted():
    """When realised tail loss is consistently MORE EXTREME than predicted ES,
    the corrector should drive its scale ABOVE 1, i.e. report a more
    negative ES on subsequent calls."""
    c = EsCorrector(alpha=0.025, Kp=0.2, Ki=0.02, Kd=0.05)
    # Realised = -3%, base_es = -1% → realised is 3× worse than predicted.
    for _ in range(10):
        c.update(realised=-0.03, stated_es=-0.01, breached=True)
    assert c.scale > 1.0, f"expected scale > 1 after under-estimated ES; got {c.scale}"


def test_es_corrector_tightens_when_realised_inside_predicted():
    """When realised tail loss is consistently LESS EXTREME than predicted ES,
    the corrector should drive its scale BELOW 1."""
    c = EsCorrector(alpha=0.025, Kp=0.2, Ki=0.02, Kd=0.05)
    for _ in range(10):
        c.update(realised=-0.005, stated_es=-0.02, breached=True)
    assert c.scale < 1.0, f"expected scale < 1 after over-estimated ES; got {c.scale}"


def test_es_corrector_no_op_when_no_breach():
    c = EsCorrector(alpha=0.025)
    for _ in range(50):
        c.update(realised=-0.05, stated_es=-0.02, breached=False)
    assert c.scale == 1.0
    assert c.state()["es_n_updates"] == 0


def test_es_corrector_respects_scale_bounds():
    c = EsCorrector(alpha=0.025, Kp=2.0, Ki=0.0, Kd=0.0, scale_bounds=(0.5, 2.5))
    # Hammer with extreme realised values; corrector should saturate at 2.5
    # rather than blow up.
    for _ in range(200):
        c.update(realised=-1.0, stated_es=-0.005, breached=True)
    assert 0.5 <= c.scale <= 2.5


# JointVarEsCalibrator integration behaviours.


def test_joint_calibrator_portfolio_pnl_is_cached():
    """Regression: an earlier draft re-created the `JointCalibratedDistribution`
    on every `portfolio_pnl()` call, which triggered `controller.predict`
    TWICE per alpha per step (visible as the mdn-widths EMA / smoothing-EMA
    state advancing twice as fast as the bare `ConformalCalibrator`). The
    fix is to cache `portfolio_pnl` on the joint forecast so the runner's
    call and `observe_realized`'s lookup return the same instance.
    """
    cal = make_joint_calibrator()
    cal.fit(synthetic_returns(n=600))
    forecast = cal.predict(np.full(3, 1.0 / 3))

    pnl_a = forecast.portfolio_pnl()
    pnl_b = forecast.portfolio_pnl()
    assert pnl_a is pnl_b, (
        "JointCalibratedPortfolioForecast.portfolio_pnl() must cache its result"
    )

    # And the cache the runner sees IS the cache observe_realized sees.
    pnl_c = cal.current_forecast.portfolio_pnl()
    assert pnl_c is pnl_a, (
        "observe_realized must reach the same JointCalibratedDistribution "
        "the runner used"
    )


def test_joint_calibrator_state_includes_both_var_and_es():
    cal = make_joint_calibrator()
    cal.fit(synthetic_returns(n=600, seed=1))
    cal.predict(np.full(3, 1.0 / 3))
    cal.observe_realized(
        realized=-0.04,
        realized_per_asset={"SPY": -0.04, "TLT": -0.04, "GLD": -0.04},
    )

    state = cal.controller_state()
    for st in state.values():
        # VaR-side keys (from ConformalPIDController.state()):
        assert "alpha_target" in st
        assert "adjusted_quantile" in st
        # ES-side keys (from EsCorrector.state()):
        assert "es_alpha_target" in st
        assert "es_scale" in st
        assert "es_n_updates" in st


def test_joint_calibrated_distribution_es_applies_corrector_scale():
    cal = make_joint_calibrator()
    cal.fit(synthetic_returns(n=600, seed=2))
    weights = np.full(3, 1.0 / 3)
    forecast = cal.predict(weights)
    pnl = forecast.portfolio_pnl()
    assert isinstance(pnl, JointCalibratedDistribution)

    base_es = pnl.base_es(0.025)
    es = pnl.es(0.025)
    # At t=0, scale = 1.0 → es ≈ base_es.
    assert es == pytest.approx(base_es, rel=1e-9)

    # A scale above 1 deepens the stated ES by scaling its excess over the VaR.
    # (The earlier version of this test forced ten 50% losses, which pushes the
    # calibrated VaR beyond every base sample; the base tail is then empty, the
    # excess is zero, and no scale can deepen it. That edge is coherent, ES
    # equals VaR, and is not what this test is about.)
    import math

    cal.es_correctors[0.025]._log_scale = math.log(2.0)
    var = pnl.quantile(0.025)
    es2 = pnl.es(0.025)
    assert es2 == pytest.approx(var + 2.0 * (base_es - var), rel=1e-9)
    assert es2 < base_es < var


# Coherence and closed-loop behaviour (0.2.0).


def test_stated_es_is_never_shallower_than_stated_var():
    """ES is the mean loss beyond the VaR. With the correction at its lower
    bound the stated ES must still be at least as deep as the stated VaR."""
    import math

    cal = make_joint_calibrator()
    cal.fit(synthetic_returns(n=600))
    for corrector in cal.es_correctors.values():
        corrector._log_scale = math.log(0.5)
    pnl = cal.predict(np.full(3, 1.0 / 3)).portfolio_pnl()
    for alpha in cal.es_correctors:
        assert pnl.es(alpha) <= pnl.quantile(alpha)
        assert pnl.es(alpha) > pnl.base_es(alpha)   # the correction did shrink the excess


def test_corrector_loop_is_closed_and_settles_inside_its_bounds():
    """Fed the ES it stated, the scale settles where stated ES matches the
    realised tail mean instead of drifting to a bound."""
    c = EsCorrector(alpha=0.025)
    var, base_es, realised = -0.010, -0.020, -0.016   # fixed point at scale 0.6
    for _ in range(600):
        stated = var + c.scale * (base_es - var)
        c.update(realised=realised, stated_es=stated, breached=True)
    stated = var + c.scale * (base_es - var)
    assert abs(stated - realised) < 5e-4
    assert 0.5 < c.scale < 3.0

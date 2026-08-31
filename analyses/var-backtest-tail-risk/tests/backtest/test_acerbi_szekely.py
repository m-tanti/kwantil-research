"""Tests for the Acerbi-Szekely Z1/Z2 ES backtests."""

import numpy as np
import pytest
from scipy.stats import norm

from var_backtest_uq.backtest.acerbi_szekely import (
    acerbi_szekely_test_1,
    acerbi_szekely_test_2,
    gaussian_h0_sampler,
)


def _gaussian_var_es(sigma: float, alpha: float) -> tuple[float, float]:
    """Closed-form VaR/ES for N(0, sigma)."""
    z = norm.ppf(alpha)
    var = sigma * z
    es = -sigma * norm.pdf(z) / alpha
    return var, es


def test_z2_correctly_specified_es_returns_near_zero():
    """When realised P&L is drawn from the predicted distribution, Z2 ~= 0."""
    rng = np.random.default_rng(0)
    n = 5000
    sigma = 0.01
    alpha = 0.025
    var, es = _gaussian_var_es(sigma, alpha)
    realized = rng.normal(0.0, sigma, size=n)
    z2 = acerbi_szekely_test_2(realized, np.full(n, var), np.full(n, es))["statistic"]
    assert abs(z2) < 0.2, f"Z2 should be near 0 under H0; got {z2}"


def test_z2_under_estimated_es_is_negative():
    """If predicted ES is too small in magnitude, realised tail mean exceeds it,
    Z2 < 0 (under-estimation of tail risk)."""
    rng = np.random.default_rng(1)
    n = 5000
    sigma = 0.01
    alpha = 0.025
    var_truth, es_truth = _gaussian_var_es(sigma, alpha)
    realized = rng.normal(0.0, sigma, size=n)
    es_under = es_truth * 0.5  # half magnitude → under-estimated
    z2 = acerbi_szekely_test_2(realized, np.full(n, var_truth), np.full(n, es_under))["statistic"]
    assert z2 < -0.4, f"Z2 should be sharply negative when ES under-estimated; got {z2}"


def test_z1_with_bootstrap_zones_classifies_correctly():
    rng = np.random.default_rng(2)
    n = 2000
    sigma = 0.01
    alpha = 0.025
    var, es = _gaussian_var_es(sigma, alpha)
    var_arr = np.full(n, var)
    es_arr = np.full(n, es)
    sigma_arr = np.full(n, sigma)
    sampler = gaussian_h0_sampler(np.zeros(n), sigma_arr)

    realized_h0 = rng.normal(0.0, sigma, size=n)
    res_h0 = acerbi_szekely_test_1(
        realized_h0, var_arr, es_arr, alpha=alpha,
        sample_under_h0=sampler, n_simulations=1000, seed=10,
    )
    assert res_h0["zone"] == "green"

    realized_bad = rng.normal(0.0, sigma * 1.6, size=n)
    res_bad = acerbi_szekely_test_1(
        realized_bad, var_arr, es_arr, alpha=alpha,
        sample_under_h0=sampler, n_simulations=1000, seed=11,
    )
    assert res_bad["zone"] in {"yellow", "red"}, (
        f"expected non-green zone for misspecified ES; got {res_bad['zone']}, "
        f"z1 = {res_bad['statistic']}, thresholds = "
        f"green {res_bad['green_threshold']}, yellow {res_bad['yellow_threshold']}"
    )


def test_z2_no_breaches_returns_nan():
    n = 100
    realized = np.full(n, 0.01)
    var = np.full(n, -0.05)
    es = np.full(n, -0.06)
    res = acerbi_szekely_test_2(realized, var, es)
    assert np.isnan(res["statistic"])


def test_input_alignment_validated():
    with pytest.raises(ValueError):
        acerbi_szekely_test_1(
            np.zeros(5), np.zeros(5), np.zeros(6), alpha=0.025,
        )

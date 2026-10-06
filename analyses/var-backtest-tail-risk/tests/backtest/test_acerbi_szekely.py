"""Tests for the Acerbi-Szekely Z1/Z2 ES backtests (names as in A&S 2014)."""

import numpy as np
import pytest
from scipy.stats import norm

from var_backtest_uq.backtest.acerbi_szekely import (
    acerbi_szekely_test_2,
    acerbi_szekely_test_1,
    gaussian_h0_sampler,
)


def _gaussian_var_es(sigma: float, alpha: float) -> tuple[float, float]:
    """Closed-form VaR/ES for N(0, sigma)."""
    z = norm.ppf(alpha)
    var = sigma * z
    es = -sigma * norm.pdf(z) / alpha
    return var, es


def test_z1_correctly_specified_es_returns_near_zero():
    """When realised P&L is drawn from the predicted distribution, Z1 ~= 0."""
    rng = np.random.default_rng(0)
    n = 5000
    sigma = 0.01
    alpha = 0.025
    var, es = _gaussian_var_es(sigma, alpha)
    realized = rng.normal(0.0, sigma, size=n)
    z1 = acerbi_szekely_test_1(realized, np.full(n, var), np.full(n, es))["statistic"]
    assert abs(z1) < 0.2, f"Z1 should be near 0 under H0; got {z1}"


def test_z1_under_estimated_es_is_negative():
    """If predicted ES is too small in magnitude, realised tail mean exceeds it,
    Z1 < 0 (under-estimation of tail risk)."""
    rng = np.random.default_rng(1)
    n = 5000
    sigma = 0.01
    alpha = 0.025
    var_truth, es_truth = _gaussian_var_es(sigma, alpha)
    realized = rng.normal(0.0, sigma, size=n)
    es_under = es_truth * 0.5  # half magnitude → under-estimated
    z1 = acerbi_szekely_test_1(realized, np.full(n, var_truth), np.full(n, es_under))["statistic"]
    assert z1 < -0.4, f"Z1 should be sharply negative when ES under-estimated; got {z1}"


def test_z2_with_bootstrap_zones_classifies_correctly():
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
    res_h0 = acerbi_szekely_test_2(
        realized_h0, var_arr, es_arr, alpha=alpha,
        sample_under_h0=sampler, n_simulations=1000, seed=10,
    )
    assert res_h0["zone"] == "green"

    realized_bad = rng.normal(0.0, sigma * 1.6, size=n)
    res_bad = acerbi_szekely_test_2(
        realized_bad, var_arr, es_arr, alpha=alpha,
        sample_under_h0=sampler, n_simulations=1000, seed=11,
    )
    assert res_bad["zone"] in {"yellow", "red"}, (
        f"expected non-green zone for misspecified ES; got {res_bad['zone']}, "
        f"z2 = {res_bad['statistic']}, thresholds = "
        f"green {res_bad['green_threshold']}, yellow {res_bad['yellow_threshold']}"
    )


def test_z1_no_breaches_returns_nan():
    n = 100
    realized = np.full(n, 0.01)
    var = np.full(n, -0.05)
    es = np.full(n, -0.06)
    res = acerbi_szekely_test_1(realized, var, es)
    assert np.isnan(res["statistic"])


def test_input_alignment_validated():
    with pytest.raises(ValueError):
        acerbi_szekely_test_2(
            np.zeros(5), np.zeros(5), np.zeros(6), alpha=0.025,
        )


def test_z1_z2_identity():
    """Z2 = 1 + N_breach / (T * alpha) * (Z1 - 1), exactly."""
    rng = np.random.default_rng(3)
    n, sigma, alpha = 1000, 0.01, 0.025
    var, es = _gaussian_var_es(sigma, alpha)
    realized = rng.standard_t(4, size=n) * sigma
    v, e = np.full(n, var), np.full(n, es)
    z1 = acerbi_szekely_test_1(realized, v, e)
    z2 = acerbi_szekely_test_2(realized, v, e, alpha=alpha)
    k = z1["n_breaches"] / (n * alpha)
    assert z2["statistic"] == pytest.approx(1 + k * (z1["statistic"] - 1), abs=1e-12)


def test_z1_blind_to_breach_frequency_z2_is_not():
    """VaR and ES both scaled by 0.8 on a correct model: breaches roughly
    double. Z1 barely moves; Z2 goes clearly negative."""
    rng = np.random.default_rng(4)
    n, sigma, alpha = 20000, 0.01, 0.025
    var, es = _gaussian_var_es(sigma, alpha)
    realized = rng.normal(0.0, sigma, size=n)
    v, e = np.full(n, 0.8 * var), np.full(n, 0.8 * es)
    z1 = acerbi_szekely_test_1(realized, v, e)["statistic"]
    z2 = acerbi_szekely_test_2(realized, v, e, alpha=alpha)["statistic"]
    assert abs(z1) < 0.1, f"Z1 should stay near 0; got {z1}"
    assert z2 < -0.5, f"Z2 should be sharply negative; got {z2}"

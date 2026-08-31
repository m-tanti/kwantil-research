"""Acerbi & Szekely (2014) Z1 / Z2 ES backtests with bootstrap zones.

P&L sign: positive=gain, negative=loss. VaR_alpha / ES_alpha are negative.
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

import numpy as np


Zone = Literal["green", "yellow", "red"]


class AcerbiSzekelyResult(TypedDict, total=False):
    """A&S Z1/Z2 result. zone/p_value/thresholds populated only when bootstrapped."""
    statistic: float
    n_obs: int
    n_breaches: int
    zone: Zone | None
    p_value: float
    green_threshold: float
    yellow_threshold: float
    n_simulations: int


def _z1_statistic(
    realized: np.ndarray,
    var_pred: np.ndarray,
    es_pred: np.ndarray,
    alpha: float,
) -> float:
    # Z1 = (1/(N*alpha)) * sum_t (X_t * 1[X_t<VaR] / |ES_t|) + 1.
    # Divide by |ES| (P&L sign convention) so the A&S derivation lines up.
    n = realized.size
    breach = realized < var_pred
    es_mag = np.abs(es_pred)
    es_safe = np.where(es_mag < 1e-12, 1e-12, es_mag)
    contributions = np.where(breach, realized / es_safe, 0.0)
    return float(contributions.sum() / (n * alpha) + 1.0)


def _z2_statistic(
    realized: np.ndarray,
    var_pred: np.ndarray,
    es_pred: np.ndarray,
) -> float:
    """Z2, conditional on breach. NaN when no breaches occurred."""
    breach = realized < var_pred
    n_breach = int(breach.sum())
    if n_breach == 0:
        return float("nan")
    es_mag = np.abs(es_pred)
    es_safe = np.where(es_mag < 1e-12, 1e-12, es_mag)
    return float((realized[breach] / es_safe[breach]).mean() + 1.0)


def acerbi_szekely_test_1(
    realized: np.ndarray,
    var_pred: np.ndarray,
    es_pred: np.ndarray,
    alpha: float,
    sample_under_h0: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    n_simulations: int = 5_000,
    seed: int = 42,
) -> AcerbiSzekelyResult:
    """A&S Test 1 (unconditional). Without `sample_under_h0`, returns the statistic only."""
    realized = np.asarray(realized, dtype=float).ravel()
    var_pred = np.asarray(var_pred, dtype=float).ravel()
    es_pred = np.asarray(es_pred, dtype=float).ravel()
    if not (realized.size == var_pred.size == es_pred.size):
        raise ValueError("realized / var_pred / es_pred must align in length")

    z1 = _z1_statistic(realized, var_pred, es_pred, alpha)
    out: dict = {
        "statistic": z1,
        "n_obs": realized.size,
        "n_breaches": int((realized < var_pred).sum()),
    }
    if sample_under_h0 is None:
        out.update({"zone": None, "p_value": float("nan"),
                    "green_threshold": float("nan"), "yellow_threshold": float("nan")})
        return out

    rng = np.random.default_rng(seed)
    null_z1s = np.empty(n_simulations)
    for i in range(n_simulations):
        x_sim = sample_under_h0(rng, realized.size)
        null_z1s[i] = _z1_statistic(x_sim, var_pred, es_pred, alpha)
    # Lower-tail: reject when Z1 falls below the 5th-percentile null.
    p_value = float(np.mean(null_z1s <= z1))
    green_threshold = float(np.percentile(null_z1s, 5.0))
    yellow_threshold = float(np.percentile(null_z1s, 0.1))
    if z1 >= green_threshold:
        zone = "green"
    elif z1 >= yellow_threshold:
        zone = "yellow"
    else:
        zone = "red"
    out.update({
        "zone": zone,
        "p_value": p_value,
        "green_threshold": green_threshold,
        "yellow_threshold": yellow_threshold,
        "n_simulations": n_simulations,
    })
    return out


def acerbi_szekely_test_2(
    realized: np.ndarray,
    var_pred: np.ndarray,
    es_pred: np.ndarray,
    sample_under_h0: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    n_simulations: int = 5_000,
    seed: int = 42,
) -> AcerbiSzekelyResult:
    """A&S Test 2 (conditional on breach)."""
    realized = np.asarray(realized, dtype=float).ravel()
    var_pred = np.asarray(var_pred, dtype=float).ravel()
    es_pred = np.asarray(es_pred, dtype=float).ravel()
    if not (realized.size == var_pred.size == es_pred.size):
        raise ValueError("realized / var_pred / es_pred must align in length")

    z2 = _z2_statistic(realized, var_pred, es_pred)
    out: dict = {
        "statistic": z2,
        "n_obs": realized.size,
        "n_breaches": int((realized < var_pred).sum()),
    }
    if sample_under_h0 is None or np.isnan(z2):
        out.update({"zone": None, "p_value": float("nan"),
                    "green_threshold": float("nan"), "yellow_threshold": float("nan")})
        return out

    rng = np.random.default_rng(seed)
    null_z2s = []
    for _ in range(n_simulations):
        x_sim = sample_under_h0(rng, realized.size)
        z = _z2_statistic(x_sim, var_pred, es_pred)
        if not np.isnan(z):
            null_z2s.append(z)
    if not null_z2s:
        out.update({"zone": None, "p_value": float("nan")})
        return out
    null_arr = np.array(null_z2s)
    p_value = float(np.mean(null_arr <= z2))
    green_threshold = float(np.percentile(null_arr, 5.0))
    yellow_threshold = float(np.percentile(null_arr, 0.1))
    if z2 >= green_threshold:
        zone = "green"
    elif z2 >= yellow_threshold:
        zone = "yellow"
    else:
        zone = "red"
    out.update({
        "zone": zone,
        "p_value": p_value,
        "green_threshold": green_threshold,
        "yellow_threshold": yellow_threshold,
        "n_simulations": len(null_arr),
    })
    return out


def gaussian_h0_sampler(mu: np.ndarray, sigma: np.ndarray) -> Callable:
    """H0 sampler drawing N(mu_t, sigma_t) per date.

    Under-states tail risk against fat-tailed predictions; for those use
    `student_t_h0_sampler`.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have the same shape")

    def _sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        if n != mu.shape[0]:
            raise ValueError(f"sampler expects n={mu.shape[0]}; got {n}")
        return mu + sigma * rng.standard_normal(n)

    return _sampler


def student_t_h0_sampler(
    mu: np.ndarray,
    sigma: np.ndarray,
    df: np.ndarray | float,
    df_floor: float = 4.5,
) -> Callable:
    """H0 sampler drawing a per-date scaled Student-t.

    `df` is scalar (broadcast) or aligned with mu/sigma. df_floor keeps
    kurtosis 6/(df-4) and the variance rescale finite.
    """
    from scipy.stats import t as student_t

    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    if mu.shape != sigma.shape:
        raise ValueError("mu and sigma must have the same shape")
    if np.isscalar(df):
        df_arr = np.full_like(mu, max(float(df), df_floor))
    else:
        df_arr = np.asarray(df, dtype=float).ravel()
        if df_arr.shape != mu.shape:
            raise ValueError("df must be scalar or align with mu/sigma")
        df_arr = np.maximum(df_arr, df_floor)
    # Scale so the per-date marginal variance equals sigma_t^2 exactly.
    scale_t = sigma * np.sqrt((df_arr - 2.0) / df_arr)

    def _sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        if n != mu.shape[0]:
            raise ValueError(f"sampler expects n={mu.shape[0]}; got {n}")
        z = student_t.rvs(df=df_arr, size=n, random_state=rng)
        return mu + scale_t * z

    return _sampler


def empirical_h0_sampler_from_residuals(residuals: np.ndarray) -> Callable:
    """H0 sampler resampling iid from a residual pool (for non-parametric bases)."""
    pool = np.asarray(residuals, dtype=float).ravel()
    if pool.size == 0:
        raise ValueError("residual pool is empty")

    def _sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(pool, size=n, replace=True)

    return _sampler

"""Kupiec POF and Christoffersen independence / conditional-coverage tests.

p-value methods: "mc" (default; bootstrap, robust at alpha=0.01) or "asymptotic"
(chi-square LR, breaks down when expected cell counts fall below ~5).
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

import numpy as np
from scipy.stats import chi2, norm


PValueMethod = Literal["mc", "asymptotic"]


class KupiecResult(TypedDict, total=False):
    """Kupiec POF result. CI keys present when n_obs>0; n_simulations in MC mode."""
    n_obs: int
    n_breaches: int
    observed_rate: float
    observed_rate_ci_lo: float
    observed_rate_ci_hi: float
    target_rate: float
    statistic: float
    p_value: float
    p_value_method: PValueMethod
    n_simulations: int | None


class ChristoffersenIndependenceResult(TypedDict, total=False):
    """Christoffersen independence result; n_simulations in MC mode."""
    statistic: float
    p_value: float
    p_value_method: PValueMethod
    n_simulations: int | None
    n_transitions: int
    n00: int
    n01: int
    n10: int
    n11: int


class ChristoffersenCCResult(TypedDict, total=False):
    """Christoffersen conditional-coverage result; n_simulations in MC mode."""
    statistic: float
    p_value: float
    p_value_method: PValueMethod
    n_simulations: int | None
    lr_pof: float
    lr_ind: float


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else 0.0


def wilson_score_interval(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Stays inside [0,1] for small p̂ where the normal approximation breaks."""
    if n <= 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(0.5 + level / 2))
    p_hat = x / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _kupiec_lr(breaches: np.ndarray, alpha: float) -> tuple[float, int, int]:
    n = int(breaches.size)
    x = int(breaches.sum())
    if n == 0:
        return float("nan"), 0, 0
    p_hat = x / n
    ll_null = x * _safe_log(alpha) + (n - x) * _safe_log(1.0 - alpha)
    ll_alt = x * _safe_log(p_hat) + (n - x) * _safe_log(1.0 - p_hat)
    return max(-2.0 * (ll_null - ll_alt), 0.0), n, x


def _independence_lr(breaches: np.ndarray) -> tuple[float, dict]:
    if breaches.size < 2:
        return float("nan"), {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    prev = breaches[:-1]
    curr = breaches[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    n_total = n00 + n01 + n10 + n11

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / n_total if n_total > 0 else 0.0

    ll_null = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1.0 - pi_01)
        + n01 * _safe_log(pi_01)
        + n10 * _safe_log(1.0 - pi_11)
        + n11 * _safe_log(pi_11)
    )
    return max(-2.0 * (ll_null - ll_alt), 0.0), {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
    }


def _mc_pvalue_kupiec(observed_lr: float, n: int, alpha: float,
                     n_simulations: int, seed: int) -> float:
    if math.isnan(observed_lr) or n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    null_lrs = np.zeros(n_simulations)
    sims = rng.random((n_simulations, n)) < alpha
    for i in range(n_simulations):
        lr_i, _, _ = _kupiec_lr(sims[i].astype(int), alpha)
        null_lrs[i] = lr_i
    return float(np.mean(null_lrs >= observed_lr))


def _mc_pvalue_independence(observed_lr: float, n: int, p_hat: float,
                           n_simulations: int, seed: int) -> float:
    # H0 here is independence, not rate=alpha, so resample at p_hat not alpha.
    if math.isnan(observed_lr) or n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    null_lrs = np.zeros(n_simulations)
    sims = rng.random((n_simulations, n)) < p_hat
    for i in range(n_simulations):
        lr_i, _ = _independence_lr(sims[i].astype(int))
        null_lrs[i] = 0.0 if math.isnan(lr_i) else lr_i
    return float(np.mean(null_lrs >= observed_lr))


def _mc_pvalue_cc(observed_lr: float, n: int, alpha: float,
                  n_simulations: int, seed: int) -> float:
    if math.isnan(observed_lr) or n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    null_lrs = np.zeros(n_simulations)
    sims = rng.random((n_simulations, n)) < alpha
    for i in range(n_simulations):
        sim_int = sims[i].astype(int)
        pof_lr, _, _ = _kupiec_lr(sim_int, alpha)
        ind_lr, _ = _independence_lr(sim_int)
        if math.isnan(pof_lr) or math.isnan(ind_lr):
            null_lrs[i] = 0.0
        else:
            null_lrs[i] = pof_lr + ind_lr
    return float(np.mean(null_lrs >= observed_lr))


def kupiec_pof(
    breaches: np.ndarray,
    alpha: float,
    *,
    method: PValueMethod = "mc",
    n_simulations: int = 10_000,
    seed: int = 42,
) -> KupiecResult:
    """Kupiec POF: H0 P(breach)=alpha. LR ~ chi-square(1) under asymptotic mode."""
    b = np.asarray(breaches, dtype=int).ravel()
    lr, n, x = _kupiec_lr(b, alpha)
    if n == 0:
        return {
            "n_obs": 0, "n_breaches": 0, "observed_rate": float("nan"),
            "target_rate": alpha, "statistic": float("nan"),
            "p_value": float("nan"), "p_value_method": method,
        }
    if method == "mc":
        p_value = _mc_pvalue_kupiec(lr, n, alpha, n_simulations, seed)
    else:
        p_value = float(1.0 - chi2.cdf(lr, df=1))
    ci_lo, ci_hi = wilson_score_interval(x, n, level=0.95)
    return {
        "n_obs": n,
        "n_breaches": x,
        "observed_rate": x / n,
        "observed_rate_ci_lo": ci_lo,
        "observed_rate_ci_hi": ci_hi,
        "target_rate": alpha,
        "statistic": lr,
        "p_value": p_value,
        "p_value_method": method,
        "n_simulations": n_simulations if method == "mc" else None,
    }


def christoffersen_independence(
    breaches: np.ndarray,
    *,
    method: PValueMethod = "mc",
    n_simulations: int = 10_000,
    seed: int = 42,
) -> ChristoffersenIndependenceResult:
    """Christoffersen independence: H0 P(b_t=1|b_{t-1}=0) = P(b_t=1|b_{t-1}=1)."""
    b = np.asarray(breaches, dtype=int).ravel()
    lr, transitions = _independence_lr(b)
    n_total = transitions["n00"] + transitions["n01"] + transitions["n10"] + transitions["n11"]
    if n_total == 0:
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "n_transitions": 0, "p_value_method": method,
            **transitions,
        }
    if method == "mc":
        p_value = _mc_pvalue_independence(lr, b.size, float(b.mean()), n_simulations, seed)
    else:
        p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {
        "statistic": lr,
        "p_value": p_value,
        "p_value_method": method,
        "n_simulations": n_simulations if method == "mc" else None,
        "n_transitions": n_total,
        **transitions,
    }


def christoffersen_conditional_coverage(
    breaches: np.ndarray,
    alpha: float,
    *,
    method: PValueMethod = "mc",
    n_simulations: int = 10_000,
    seed: int = 42,
) -> ChristoffersenCCResult:
    """Joint Kupiec + independence; LR ~ chi-square(2) under asymptotic mode."""
    b = np.asarray(breaches, dtype=int).ravel()
    pof_lr, n, _ = _kupiec_lr(b, alpha)
    ind_lr, _ = _independence_lr(b)
    if math.isnan(pof_lr) or math.isnan(ind_lr):
        return {
            "statistic": float("nan"), "p_value": float("nan"),
            "lr_pof": pof_lr, "lr_ind": ind_lr,
            "p_value_method": method,
        }
    cc_lr = pof_lr + ind_lr
    if method == "mc":
        p_value = _mc_pvalue_cc(cc_lr, n, alpha, n_simulations, seed)
    else:
        p_value = float(1.0 - chi2.cdf(cc_lr, df=2))
    return {
        "statistic": cc_lr,
        "p_value": p_value,
        "p_value_method": method,
        "n_simulations": n_simulations if method == "mc" else None,
        "lr_pof": pof_lr,
        "lr_ind": ind_lr,
    }

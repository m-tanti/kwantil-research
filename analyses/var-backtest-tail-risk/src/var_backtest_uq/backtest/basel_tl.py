"""Basel III backtesting traffic light (CRR Article 366)."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
import pandas as pd


Zone = Literal["green", "yellow", "red", "n/a"]


class BaselTrafficLightResult(TypedDict):
    """Empty input returns zone='n/a', total_multiplier=BASE_MULTIPLIER."""
    zone: Zone
    n_breaches: int
    addon: float
    total_multiplier: float
    window_used: int


# CRR Article 366 backtesting plus-factor table.
_ADDON_TABLE: dict[int, float] = {
    0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00,
    5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85,
}
BASE_MULTIPLIER = 3.0


def _addon_for(n: int) -> float:
    return _ADDON_TABLE.get(n, 1.00)


def _zone_for(n: int) -> Zone:
    if n <= 4:
        return "green"
    if n <= 9:
        return "yellow"
    return "red"


def basel_traffic_light(breaches: np.ndarray, window: int = 250) -> BaselTrafficLightResult:
    """Basel TL zone over the most recent `window` breaches."""
    b = np.asarray(breaches, dtype=int).ravel()
    if b.size == 0:
        return {
            "zone": "n/a", "n_breaches": 0, "addon": 0.0,
            "total_multiplier": BASE_MULTIPLIER, "window_used": 0,
        }
    used = b[-window:] if b.size >= window else b
    n = int(used.sum())
    addon = _addon_for(n)
    return {
        "zone": _zone_for(n),
        "n_breaches": n,
        "addon": addon,
        "total_multiplier": BASE_MULTIPLIER + addon,
        "window_used": used.size,
    }


def rolling_basel_multiplier(
    breaches: np.ndarray,
    window: int = 250,
) -> pd.DataFrame:
    """Per-step Basel multiplier and zone. Pre-`window` rows use the trailing-available count."""
    b = np.asarray(breaches, dtype=int).ravel()
    n = b.size
    if n == 0:
        return pd.DataFrame(columns=["n_breaches_window", "addon", "multiplier", "zone"])
    counts = np.empty(n, dtype=int)
    cumsum = np.concatenate([[0], np.cumsum(b)])
    for i in range(n):
        start = max(0, i + 1 - window)
        counts[i] = int(cumsum[i + 1] - cumsum[start])
    addons = np.array([_addon_for(c) for c in counts])
    zones = np.array([_zone_for(c) for c in counts])
    return pd.DataFrame({
        "n_breaches_window": counts,
        "addon": addons,
        "multiplier": BASE_MULTIPLIER + addons,
        "zone": zones,
    })


def ima_capital_path(
    var_pred: np.ndarray,
    breaches: np.ndarray,
    window: int = 250,
    var_60d_window: int = 60,
) -> pd.DataFrame:
    """Per-step IMA capital: m_c(t) * max(|VaR_t|, mean_60(|VaR|))."""
    var_abs = np.abs(np.asarray(var_pred, dtype=float).ravel())
    if var_abs.size == 0:
        return pd.DataFrame(columns=[
            "var_abs", "var_60d_mean", "multiplier", "zone", "ima_capital",
        ])
    var_60d = pd.Series(var_abs).rolling(var_60d_window, min_periods=1).mean().to_numpy()
    rolling = rolling_basel_multiplier(breaches, window=window)
    capital = rolling["multiplier"].to_numpy() * np.maximum(var_abs, var_60d)
    return pd.DataFrame({
        "var_abs": var_abs,
        "var_60d_mean": var_60d,
        "multiplier": rolling["multiplier"].to_numpy(),
        "zone": rolling["zone"].to_numpy(),
        "ima_capital": capital,
    })

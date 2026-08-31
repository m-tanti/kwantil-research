"""Portfolio return aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """`weights` align to `returns.columns` order."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length {weights.shape[0]} does not match {returns.shape[1]} return columns"
        )
    return pd.Series(returns.values @ weights, index=returns.index, name="portfolio_pnl")

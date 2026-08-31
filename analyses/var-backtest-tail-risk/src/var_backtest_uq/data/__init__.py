"""Price-data ingestion and portfolio aggregation.

- `fetch`, yfinance pull with on-disk Parquet cache.
- `portfolio`, log returns + portfolio aggregation helpers.
- `stress_windows`, named stress periods for dashboard / report overlays.
"""

from .fetch import fetch_prices
from .portfolio import log_returns, portfolio_returns
from .stress_windows import STRESS_WINDOWS, tag_dates

__all__ = [
    "fetch_prices",
    "log_returns",
    "portfolio_returns",
    "STRESS_WINDOWS",
    "tag_dates",
]

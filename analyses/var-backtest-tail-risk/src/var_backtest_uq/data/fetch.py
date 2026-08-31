"""Daily price fetch with on-disk parquet cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: Sequence[str],
    cache_dir: Path,
    start: str = "2010-01-01",
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted-close daily prices, parquet-cached per ticker. Drops dates not common across tickers."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.Series] = []
    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}.parquet"
        if cache_path.exists() and not refresh:
            df = pd.read_parquet(cache_path)
        else:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df.empty:
                raise RuntimeError(f"yfinance returned empty frame for {ticker}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_path)
        frames.append(df["Close"].rename(ticker))
    return pd.concat(frames, axis=1).dropna()

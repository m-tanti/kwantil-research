"""Named stress windows used by dashboard overlays and the PDF stress tables."""

from __future__ import annotations

import pandas as pd


STRESS_WINDOWS: list[tuple[str, str, str]] = [
    ("COVID crash",            "2020-02-19", "2020-04-30"),
    ("UK gilt / Fed terminal", "2022-09-01", "2022-11-15"),
    ("SVB / CS",               "2023-03-08", "2023-03-31"),
    ("Yen carry unwind",       "2024-08-01", "2024-08-15"),
    ("Tariff vol",             "2025-04-02", "2025-04-30"),
]


def tag_dates(dates: pd.DatetimeIndex) -> pd.Series:
    """Per-date stress-window label (empty string outside any window)."""
    out = pd.Series([""] * len(dates), index=dates, name="stress_window", dtype=object)
    for label, start, end in STRESS_WINDOWS:
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        out.iloc[mask] = label
    return out

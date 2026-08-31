"""Assemble raw ENTSO-E pulls into one tidy 15-min analysis table + missing-data audit.

Handles the resolution mix: load/generation are 15-min; day-ahead prices are hourly
until Sep 2025 and 15-min after the Oct 2025 MTU change (hourly prices are forward-filled
onto the 15-min grid within each hour - a labelled, not silent, transformation).
Output: artifacts/panel.parquet (UTC index) + audit printout.
"""
import glob
from pathlib import Path

import pandas as pd
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
OUT = ROOT / "artifacts"


def load_series(name: str) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(str(RAW / name / "*.parquet")))]
    df = pd.concat(parts)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def main() -> None:
    load_fc = load_series("load_forecast")
    load_act = load_series("load_actual")
    ws_fc = load_series("wind_solar_forecast")
    gen = load_series("generation_actual")
    price = load_series("day_ahead_prices")

    print("columns:")
    for n, d in [("load_fc", load_fc), ("load_act", load_act), ("ws_fc", ws_fc),
                 ("gen", gen), ("price", price)]:
        print(f"  {n}: {list(d.columns)[:8]}")

    grid = pd.date_range(load_act.index.min(), load_act.index.max(), freq="15min", tz="UTC")
    panel = pd.DataFrame(index=grid)

    def put(col: str, s: pd.Series) -> None:
        panel[col] = s.reindex(grid)

    put("load_fc", load_fc.iloc[:, 0])
    put("load_act", load_act.iloc[:, 0])

    def pick(df: pd.DataFrame, key: str) -> pd.Series | None:
        cols = [c for c in df.columns if key.lower() in c.lower()]
        return df[cols[0]] if cols else None

    for key, tag in [("Solar", "solar"), ("Wind Onshore", "wind_on"), ("Wind Offshore", "wind_off")]:
        s = pick(ws_fc, key)
        if s is not None:
            put(f"{tag}_fc", s)
        g = pick(gen, key)
        if g is not None:
            put(f"{tag}_act", g)

    p = price.iloc[:, 0]
    p15 = p.reindex(grid)
    # hourly stretches: forward-fill at most 3 steps (45 min) to complete each hour
    panel["da_price"] = p15.ffill(limit=3)
    panel["da_price_native_15min"] = ~p15.isna()

    panel.to_parquet(OUT / "panel.parquet")

    print(f"\npanel: {len(panel):,} rows x {len(panel.columns)} cols, "
          f"{panel.index.min()} .. {panel.index.max()}")
    print("\nmissing-data audit (% NaN by column, by quarter):")
    q = panel.drop(columns=["da_price_native_15min"]).isna().groupby(
        panel.index.to_period("Q")).mean() * 100
    print(q.round(2).to_string())


if __name__ == "__main__":
    main()

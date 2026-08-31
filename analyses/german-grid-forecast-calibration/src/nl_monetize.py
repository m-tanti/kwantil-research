"""NL monetization leg: price the Gaussian '90%' band's misses at imbalance-vs-DA spreads.

Framing (stated in the article): a desk that reserves flexibility equal to the stated
band and scales like a 100 MW-mean-load portfolio (errors proportional to system errors -
an assumption, stated). For every 15-min where the outcome escaped the trailing Gaussian
90% band, the excess MW beyond the band edge is exposure the stated band said not to
carry; it is priced at the ADVERSE imbalance spread for that interval's direction
(shortage -> Short price minus DA; surplus -> DA minus Long price; favorable spreads
clipped at zero - conservative, risk-side only). Same for the adaptive band, for contrast.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw_nl"
WIN, MINP, GAMMA = 90, 60, 0.02
LEVEL = 0.90
PORTFOLIO_MW = 100.0


def load_series(name: str) -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(str(RAW / name / "*.parquet")))]
    df = pd.concat(parts)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df[~df.index.duplicated(keep="first")].sort_index()


def main() -> None:
    load_fc = load_series("load_forecast").iloc[:, 0]
    load_act = load_series("load_actual").iloc[:, 0]
    imb = load_series("imbalance_prices")
    da = load_series("day_ahead_prices").iloc[:, 0]

    grid = pd.date_range(load_act.index.min(), load_act.index.max(), freq="15min", tz="UTC")
    p = pd.DataFrame(index=grid)
    p["fc"] = load_fc.reindex(grid)
    p["act"] = load_act.reindex(grid)
    p["long"] = imb["Long"].reindex(grid) if "Long" in imb else np.nan
    p["short"] = imb["Short"].reindex(grid) if "Short" in imb else np.nan
    p["da"] = da.reindex(grid).ffill(limit=3)
    p = p.dropna(subset=["fc", "act", "da"])
    print(f"panel rows: {len(p):,}; imbalance coverage: "
          f"long {p['long'].notna().mean():.1%} short {p['short'].notna().mean():.1%}")

    idx = p.index.tz_convert("Europe/Amsterdam")
    p["date"] = idx.date
    p["slot"] = idx.hour * 4 + idx.minute // 15
    p["err"] = p["act"] - p["fc"]

    wide = p.pivot_table(index="date", columns="slot", values="err")
    E = wide.to_numpy()
    dates = wide.index
    n_dates, n_slots = E.shape
    alpha = 1 - LEVEL
    z = norm.ppf(1 - alpha / 2)
    a_t = np.full(n_slots, alpha)

    scale = PORTFOLIO_MW / p["act"].mean()
    lookup = p.set_index(["date", "slot"])

    res = {"gauss": 0.0, "aci": 0.0, "miss_g": 0, "miss_a": 0, "n": 0}
    spreads_on_miss, spreads_all = [], []
    for t in range(MINP, n_dates):
        w0 = max(0, t - WIN)
        d = dates[t]
        for s in range(n_slots):
            window = E[w0:t, s]
            window = window[~np.isnan(window)]
            y = E[t, s]
            if len(window) < MINP or np.isnan(y):
                continue
            try:
                row = lookup.loc[(d, s)]
            except KeyError:
                continue
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            adverse = (
                max(float(row["short"]) - float(row["da"]), 0.0)
                if y > 0
                else max(float(row["da"]) - float(row["long"]), 0.0)
            ) if pd.notna(row["short"]) and pd.notna(row["long"]) else np.nan
            res["n"] += 1
            if not np.isnan(adverse):
                spreads_all.append(adverse)
            mu, sd = window.mean(), window.std(ddof=1)
            g_lo, g_hi = mu - z * sd, mu + z * sd
            aa = float(np.clip(a_t[s], 0.001, 0.45))
            a_lo = np.quantile(window, aa / 2)
            a_hi = np.quantile(window, 1 - aa / 2)
            miss_a = 0.0 if a_lo <= y <= a_hi else 1.0
            a_t[s] = a_t[s] + GAMMA * (alpha - miss_a)
            for key, lo, hi, mkey in [("gauss", g_lo, g_hi, "miss_g"), ("aci", a_lo, a_hi, "miss_a")]:
                if y < lo or y > hi:
                    res[mkey] += 1
                    excess = (y - hi) if y > hi else (lo - y)
                    if not np.isnan(adverse):
                        res[key] += excess * scale * adverse * 0.25
                        if key == "gauss":
                            spreads_on_miss.append(adverse)

    print(f"\nintervals evaluated: {res['n']:,}")
    print(f"gaussian 90% band: misses {res['miss_g']:,} ({res['miss_g']/res['n']:.1%}) "
          f"-> un-reserved adverse exposure EUR {res['gauss']:,.0f} per {PORTFOLIO_MW:.0f} MW portfolio / 13 mo")
    print(f"adaptive 90% band: misses {res['miss_a']:,} ({res['miss_a']/res['n']:.1%}) "
          f"-> EUR {res['aci']:,.0f}")
    print(f"mean adverse spread, all intervals: EUR {np.mean(spreads_all):.1f}/MWh; "
          f"on gaussian-miss intervals: EUR {np.mean(spreads_on_miss):.1f}/MWh")


if __name__ == "__main__":
    main()

"""Core calibration analysis on the DE-LU panel.

For each target (load, solar, wind onshore/offshore), day-ahead forecast errors are
evaluated out-of-sample with two interval constructions, both per quarter-hour slot,
both using a trailing 90-day window (min 60 obs), always shifted one day (no lookahead):
  gaussian  - mu +/- z * sigma of trailing errors (the parametric habit under test)
  conformal - trailing empirical error quantiles (rolling split-conformal)
Reports: bias/MAE headline, conditional bias by hour, empirical coverage vs nominal
(overall and by season), and mean interval width (sharpness).
Writes artifacts/coverage_results.parquet + artifacts/conditional_bias.parquet.
"""
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
ART = ROOT / "artifacts"
TARGETS = {
    "load": ("load_fc", "load_act"),
    "solar": ("solar_fc", "solar_act"),
    "wind_on": ("wind_on_fc", "wind_on_act"),
    "wind_off": ("wind_off_fc", "wind_off_act"),
}
LEVELS = [0.80, 0.90, 0.95]
WIN, MINP = 90, 60


def main() -> None:
    panel = pd.read_parquet(ART / "panel.parquet")
    idx = panel.index.tz_convert("Europe/Berlin")
    panel["date"] = idx.date
    panel["slot"] = idx.hour * 4 + idx.minute // 15

    cov_rows, bias_rows = [], []
    for name, (fc, act) in TARGETS.items():
        err = (panel[act] - panel[fc]).rename("err")
        d = pd.DataFrame({"err": err, "date": panel["date"], "slot": panel["slot"]})
        d = d.dropna()
        print(f"\n=== {name}: n={len(d):,}  bias={d.err.mean():+.1f} MW  "
              f"MAE={d.err.abs().mean():.1f} MW ===")

        hb = d.groupby(d.slot // 4).err.mean()
        bias_rows.extend(
            {"target": name, "hour": h, "bias_mw": b} for h, b in hb.items()
        )
        print("worst-bias hours:",
              ", ".join(f"h{h}:{b:+.0f}MW" for h, b in
                        hb.reindex(hb.abs().sort_values(ascending=False).index[:4]).items()))

        wide = d.pivot_table(index="date", columns="slot", values="err")
        season = pd.Series(pd.DatetimeIndex(wide.index).month.map(
            lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5)
            else "JJA" if m in (6, 7, 8) else "SON"), index=wide.index)

        roll = wide.rolling(WIN, min_periods=MINP)
        mu, sd = roll.mean().shift(1), roll.std().shift(1)
        for lv in LEVELS:
            a = (1 - lv) / 2
            z = norm.ppf(1 - a)
            g_lo, g_hi = mu - z * sd, mu + z * sd
            c_lo = roll.quantile(a).shift(1)
            c_hi = roll.quantile(1 - a).shift(1)
            for method, lo, hi in [("gaussian", g_lo, g_hi), ("conformal", c_lo, c_hi)]:
                inside = (wide >= lo) & (wide <= hi)
                valid = lo.notna() & hi.notna() & wide.notna()
                covered = inside[valid].sum().sum() / valid.sum().sum()
                width = (hi - lo)[valid].mean().mean()
                cov_rows.append({"target": name, "level": lv, "method": method,
                                 "coverage": covered, "mean_width_mw": width})
                by_season = {
                    s: inside[valid & (season == s).to_numpy()[:, None]].sum().sum()
                    / max(valid[(season == s).to_numpy()[:, None] & valid].sum().sum(), 1)
                    for s in ["DJF", "MAM", "JJA", "SON"]
                }
                seas = " ".join(f"{s}:{v:.3f}" for s, v in by_season.items())
                print(f"  {lv:.0%} {method:9s} cov={covered:.3f} "
                      f"width={width:7.0f} MW   {seas}")

    pd.DataFrame(cov_rows).to_parquet(ART / "coverage_results.parquet", index=False)
    pd.DataFrame(bias_rows).to_parquet(ART / "conditional_bias.parquet", index=False)


if __name__ == "__main__":
    main()

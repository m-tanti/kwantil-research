"""Compare two backtest runs: the May-2026 baseline against the extended series.

The question this answers is the only one that matters when a backtest is re-run on
fresh data: did the conclusions survive contact with data nobody had seen when the
conclusions were drawn?

Usage:
    python scripts/compare_runs.py output_baseline_may2026 output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from var_backtest_uq.data.stress_windows import STRESS_WINDOWS, tag_dates  # noqa: E402


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ALPHA = 0.01
line = "=" * 100


def load(d: Path):
    return (pd.read_csv(d / "summary.csv"),
            pd.read_csv(d / "details.csv", parse_dates=["date"]))


def main(base_dir: str, new_dir: str) -> None:
    base_s, base_d = load(Path(base_dir))
    new_s, new_d = load(Path(new_dir))

    b_end, n_end = base_d["date"].max(), new_d["date"].max()
    added = new_d[new_d["date"] > b_end]
    n_added_steps = added[added["alpha"] == ALPHA]["date"].nunique()

    print(f"{line}\nRUN COMPARISON\n{line}")
    print(f"  baseline  {base_d['date'].min():%Y-%m-%d} -> {b_end:%Y-%m-%d}   "
          f"{base_d[base_d['alpha']==ALPHA]['date'].nunique():,} steps")
    print(f"  extended  {new_d['date'].min():%Y-%m-%d} -> {n_end:%Y-%m-%d}   "
          f"{new_d[new_d['alpha']==ALPHA]['date'].nunique():,} steps")
    print(f"  new, previously unseen data: {n_added_steps} trading days "
          f"({b_end:%Y-%m-%d} to {n_end:%Y-%m-%d})")

    # ---------------------------------------------------------------- headline table
    print(f"\n{line}\nDID THE VERDICTS SURVIVE?  alpha = {ALPHA}\n{line}")
    bs = base_s[base_s["alpha"] == ALPHA].set_index("method")
    ns = new_s[new_s["alpha"] == ALPHA].set_index("method")
    print(f"  {'method':<18}{'rate was':>10}{'rate now':>10}{'Kupiec was':>12}{'Kupiec now':>12}"
          f"{'zone was':>11}{'zone now':>11}{'AS Z2 was':>11}{'AS Z2 now':>11}")
    for m in ns.index:
        if m not in bs.index:
            continue
        b, n = bs.loc[m], ns.loc[m]
        flag = ""
        if b["basel_zone"] != n["basel_zone"] or b["as_z2_zone"] != n["as_z2_zone"]:
            flag = "  <-- ZONE CHANGED"
        if (b["kupiec_pvalue"] >= 0.05) != (n["kupiec_pvalue"] >= 0.05):
            flag += "  <-- KUPIEC VERDICT FLIPPED"
        print(f"  {m:<18}{100*b['observed_rate']:>9.2f}%{100*n['observed_rate']:>9.2f}%"
              f"{b['kupiec_pvalue']:>12.4f}{n['kupiec_pvalue']:>12.4f}"
              f"{b['basel_zone']:>11}{n['basel_zone']:>11}"
              f"{b['as_z2_zone']:>11}{n['as_z2_zone']:>11}{flag}")

    # ---------------------------------------------------------------- new window only
    print(f"\n{line}\nTHE NEW WINDOW ALONE - {n_added_steps} unseen days\n{line}")
    a = added[added["alpha"] == ALPHA]
    exp = ALPHA * n_added_steps
    print(f"  expected breaches over {n_added_steps} days at {100*ALPHA:.0f}%: {exp:.2f}\n")
    print(f"  {'method':<18}{'breaches':>10}{'rate':>9}{'vs expected':>14}")
    for m, g in a.groupby("method"):
        nb = int(g["breached"].sum())
        print(f"  {m:<18}{nb:>10}{100*g['breached'].mean():>8.2f}%"
              f"{(nb - exp):>+13.2f}")
    print("\n  Note: with this few days, 0 or 1 breaches are both consistent with a correct")
    print("  model. This window is a sanity check, not a verdict.")

    # ---------------------------------------------------------------- regimes
    print(f"\n{line}\nCONDITIONAL COVERAGE, EXTENDED SERIES  alpha = {ALPHA}\n{line}")
    nd = new_d[new_d["alpha"] == ALPHA].copy()
    nd["sw"] = tag_dates(pd.DatetimeIndex(nd["date"])).values
    labels = [w[0] for w in STRESS_WINDOWS]
    print(f"  {'method':<18}{'calm':>9}" + "".join(f"{l[:11]:>13}" for l in labels))
    for m, g in nd.groupby("method"):
        row = f"  {m:<18}{100*g[g['sw']=='']['breached'].mean():>8.2f}%"
        for l in labels:
            s = g[g["sw"] == l]
            row += f"{(100*s['breached'].mean() if len(s) else float('nan')):>12.1f}%"
        print(row)
    days = nd[nd["method"] == "parametric"]
    print(f"  {'days':<18}{len(days[days['sw']=='']):>9}"
          + "".join(f"{len(days[days['sw']==l]):>13}" for l in labels))

    # ---------------------------------------------------------------- capital
    print(f"\n{line}\nBASEL MULTIPLIER, EXTENDED SERIES\n{line}")
    for m in ns.index:
        n = ns.loc[m]
        print(f"  {m:<18} zone {n['basel_zone']:<8} multiplier {n['basel_total_multiplier']:.2f}"
              f"   breaches in trailing window: {int(n['basel_n_breaches_window'])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output_baseline_may2026",
         sys.argv[2] if len(sys.argv) > 2 else "output")

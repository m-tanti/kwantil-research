"""What the intervals are worth: tool-change policies, simulated insert by insert.

A coverage number does not sell an audit. This turns the intervals into the two counts
a shop already tracks, as a function of where the change threshold is set:

  cuts of life thrown away  -- the insert was changed while it still had usable life
  cuts run past the limit   -- the insert stayed in past the wear limit

Policies, all evaluated on inserts the model never saw:
  fixed-N   change after N cuts regardless of anything. The folklore rule. N is swept,
            which traces out the trade-off curve a shop is implicitly sitting on.
  point     change at the first cut where predicted wear reaches the threshold.
  upper     change when the TOP of the 90% interval reaches the limit. Acts on the
            possibility of being over -- early, safe, expensive.
  lower     change only when the BOTTOM of the interval reaches the limit. Acts on
            near-certainty -- late, cheap, exposed.

No currency appears anywhere in this file. The reader supplies their own insert cost
and their own cost of a part cut with an over-limit tool; the counts are ours, the
rates are theirs. That is the difference between an arithmetic they can check and a
number they can dismiss.

TWO CORRECTIONS (2026-08-23 review, memo A2.2):
  * Counts are now indexed on cut_index, the PHYSICAL cut number within the insert's
    record, not on the position of a labelled row. Wear was measured at irregular
    intervals (146 labels over 167 cuts), so the old counts were "per measured cut"
    and understated the real spans by an unknown factor.
  * fixed-N is only swept over N that actually bite. The median insert carries 8 cuts,
    so at N = 18 the rule was inert for 15 of 16 inserts and "throws away nothing" was
    arithmetic rather than a finding. N values leaving more than half the inserts
    unchanged are computed but excluded from the best-N choice, and the inert count is
    reported for every N.

Consumes artifacts/interim/calibrated.csv. Outputs artifacts/interim/decision_curve.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "artifacts" / "interim"

LIMITS = [0.40, 0.50, 0.60, 0.70]
THRESHOLDS = np.round(np.arange(0.10, 1.01, 0.05), 2)
VARIANT = "meta"


def outcomes(track: pd.DataFrame, change_at: int | None, limit: float) -> tuple[int, int]:
    """Return (cuts of life thrown away, cuts run past the limit) for one insert.

    Everything is counted in physical cut numbers (cut_index), not in labelled rows.
    """
    over = track.loc[track["VB"] >= limit, "cut_index"]
    first_over = int(over.iloc[0]) if len(over) else None
    last = int(track["cut_index"].iloc[-1])

    if change_at is None:                       # never changed within the record
        return (0, 0 if first_over is None else last - first_over)
    if first_over is None:                      # never actually reached the limit
        return (max(0, last - change_at), 0)
    if change_at <= first_over:
        return (first_over - change_at, 0)
    return (0, change_at - first_over)


def first_cut_where(track: pd.DataFrame, mask) -> int | None:
    """Physical cut number of the first row satisfying mask, or None."""
    hit = track.loc[mask, "cut_index"]
    return int(hit.iloc[0]) if len(hit) else None


def policy_curve(df: pd.DataFrame, limit: float) -> list[dict]:
    """Sweep the point-prediction threshold; also evaluate the interval policies."""
    rows = []
    for seed, g_seed in df.groupby("seed"):
        tracks = [g.sort_values("cut_index").reset_index(drop=True)
                  for _, g in g_seed.groupby("case")]
        # denominator: physical cuts on record, not labelled rows
        total_cuts = sum(int(g["cut_index"].iloc[-1]) for g in tracks)

        for T in THRESHOLDS:
            waste = late = 0
            for g in tracks:
                w, l = outcomes(g, first_cut_where(g, g["pred"] >= T), limit)
                waste += w
                late += l
            rows.append({"seed": seed, "limit": limit, "policy": "point",
                         "param": T, "waste": waste, "late": late, "cuts": total_cuts,
                         "inert": 0})

        for name, col, sign in [("upper", "q_C-grp", +1), ("lower", "q_C-grp", -1),
                                ("upper-norm", "q_C-norm", +1), ("lower-norm", "q_C-norm", -1)]:
            waste = late = 0
            for g in tracks:
                bound = g["pred"] + sign * g[col]
                w, l = outcomes(g, first_cut_where(g, bound >= limit), limit)
                waste += w
                late += l
            rows.append({"seed": seed, "limit": limit, "policy": name,
                         "param": np.nan, "waste": waste, "late": late,
                         "cuts": total_cuts, "inert": 0})

        for N in range(2, 21):
            waste = late = inert = 0
            for g in tracks:
                last = int(g["cut_index"].iloc[-1])
                change = N if last >= N else None       # rule never fires on a short life
                inert += change is None
                w, l = outcomes(g, change, limit)
                waste += w
                late += l
            rows.append({"seed": seed, "limit": limit, "policy": "fixed-N",
                         "param": N, "waste": waste, "late": late,
                         "cuts": total_cuts, "inert": inert})
    return rows


def main() -> None:
    res = pd.read_csv(INTERIM / "calibrated.csv")
    df = res[res["variant"] == VARIANT].copy()

    rows = []
    for L in LIMITS:
        rows += policy_curve(df, L)
    out = pd.DataFrame(rows)

    agg = (out.groupby(["limit", "policy", "param"], dropna=False)
              [["waste", "late", "cuts", "inert"]].mean().reset_index())
    agg["waste_per_100"] = 100 * agg["waste"] / agg["cuts"]
    agg["late_per_100"] = 100 * agg["late"] / agg["cuts"]
    agg.to_csv(INTERIM / "decision_curve.csv", index=False)

    line = "=" * 84
    print(f"{line}\nTHE TRADE-OFF, PER 100 CUTS  (model '{VARIANT}', averaged over seeds)\n{line}")

    for L in LIMITS:
        a = agg[agg["limit"] == L]
        print(f"\n--- wear limit {L:.2f} mm " + "-" * 58)
        print(f"{'policy':<14} {'param':>6} {'life thrown away':>18} {'run past limit':>16}")
        named = a[a["policy"] != "point"]
        named = named[named["policy"] != "fixed-N"]
        for _, r in named.iterrows():
            print(f"{r['policy']:<14} {'':>6} {r['waste_per_100']:>17.1f} "
                  f"{r['late_per_100']:>16.1f}")

        pt = a[a["policy"] == "point"]
        best = pt.iloc[(pt["waste_per_100"] + pt["late_per_100"]).argmin()]
        print(f"{'point (best T)':<14} {best['param']:>6.2f} {best['waste_per_100']:>17.1f} "
              f"{best['late_per_100']:>16.1f}")

        fx = a[(a["policy"] == "fixed-N") & (a["inert"] <= 8)]
        if len(fx):
            bestn = fx.iloc[(fx["waste_per_100"] + fx["late_per_100"]).argmin()]
            print(f"{'fixed-N (best)':<14} {bestn['param']:>6.0f} {bestn['waste_per_100']:>17.1f} "
                  f"{bestn['late_per_100']:>16.1f}   [inert on {bestn['inert']:.0f}/16 inserts]")
        allfx = a[a["policy"] == "fixed-N"]
        excluded = allfx[allfx["inert"] > 8]["param"]
        if len(excluded):
            print(f"{'':<14} {'':>6} N >= {excluded.min():.0f} excluded: the rule never fires "
                  f"on more than half the inserts")

    print(f"\n{line}\nPOINT-THRESHOLD CURVE AT {LIMITS[2]:.2f} mm - the dial in figure 3\n{line}")
    a = agg[(agg["limit"] == LIMITS[2]) & (agg["policy"] == "point")]
    print(f"{'threshold':>10} {'life thrown away':>18} {'run past limit':>16}")
    for _, r in a.iterrows():
        print(f"{r['param']:>10.2f} {r['waste_per_100']:>17.1f} {r['late_per_100']:>16.1f}")

    print(f"\nwrote {INTERIM / 'decision_curve.csv'}")


if __name__ == "__main__":
    main()

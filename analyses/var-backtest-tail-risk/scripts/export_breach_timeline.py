"""Export breach timing and the rolling Basel window.

Two things the payload does not currently carry, both needed to SHOW findings the
tables can only assert:

  breach_events   the date of every breach, per method and level. Christoffersen tests
                  whether breaches cluster; a rug of the actual dates lets a reader see
                  the clustering rather than take a p-value on trust.

  rolling250      the trailing 250-day breach count through time, which is the quantity
                  the Basel traffic light actually thresholds on. Plotting it against the
                  zone boundaries shows the light changing colour while the model does
                  not change at all.

Both are small: breaches are rare by construction, and the rolling series is downsampled
to weekly, which is far finer than the zone boundaries move.

Writes breach_timeline.json into the dashboard payload directories.
"""

from __future__ import annotations

import json
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
WINDOW = 250
# Basel traffic-light boundaries on the trailing-250 breach count, at the 99% level
ZONES = {"green_max": 4, "yellow_max": 9}


def main() -> None:
    d = pd.read_csv(ROOT / "output" / "details.csv", parse_dates=["date"])

    events, rolling = [], []
    for (method, alpha), g in d.groupby(["method", "alpha"]):
        g = g.sort_values("date").reset_index(drop=True)

        br = g.loc[g["breached"] == 1]
        for _, r in br.iterrows():
            events.append({
                "method": method,
                "alpha": float(alpha),
                "date": r["date"].strftime("%Y-%m-%d"),
                # how far past the VaR the loss went, as a multiple of the VaR itself.
                # a breach that is barely a breach is not the same event as one that is
                # three times the number the model quoted.
                "excess_ratio": float(abs(r["realized_pnl"]) / abs(r["var"]))
                if r["var"] else None,
            })

        roll = g["breached"].rolling(WINDOW, min_periods=1).sum()
        weekly = g.iloc[::5]
        rolling.append({
            "method": method,
            "alpha": float(alpha),
            "dates": [t.strftime("%Y-%m-%d") for t in weekly["date"]],
            "count": [int(x) for x in roll.iloc[::5]],
        })

    payload = {
        "window": WINDOW,
        "zones": ZONES,
        "events": events,
        "rolling": rolling,
    }

    out = json.dumps(payload, separators=(",", ":"))
    for dirp in [ROOT / "output" / "dashboard", ROOT.parent / "dashboard" / "static" / "data"]:
        if dirp.exists():
            (dirp / "breach_timeline.json").write_text(out, encoding="utf-8")
            print(f"wrote {dirp / 'breach_timeline.json'}  ({len(out)/1024:.0f} KB)")

    print(f"\n  breach events: {len(events)}")
    a1 = [e for e in events if e["alpha"] == 0.01]
    print(f"  at alpha=0.01: {len(a1)}")
    for m in sorted({e['method'] for e in a1}):
        ev = [e for e in a1 if e["method"] == m]
        worst = max((e["excess_ratio"] or 0) for e in ev) if ev else 0
        print(f"    {m:<18}{len(ev):>4} breaches, worst loss {worst:.2f}x the quoted VaR")

    # how often does the trailing window put each method in each zone?
    print(f"\n  time spent in each Basel zone (alpha=0.01, trailing {WINDOW}d):")
    for r in [x for x in rolling if x["alpha"] == 0.01]:
        c = np.array(r["count"])
        g = 100 * (c <= ZONES["green_max"]).mean()
        y = 100 * ((c > ZONES["green_max"]) & (c <= ZONES["yellow_max"])).mean()
        rd = 100 * (c > ZONES["yellow_max"]).mean()
        print(f"    {r['method']:<18} green {g:5.1f}%   yellow {y:5.1f}%   red {rd:5.1f}%")


if __name__ == "__main__":
    main()

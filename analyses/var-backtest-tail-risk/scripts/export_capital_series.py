"""Export a compact IMA capital series for the dashboard.

The full capital path is 21,720 rows and roughly 4.7 MB of JSON. The dashboard needs a
line, not a ledger, so this downsamples to weekly and keeps only the fields a chart
reads. The full path stays in the payload for anyone who wants it.

Why this matters commercially: the capital multiplier is the quantity a Head of Market
Risk is measured on. A model that breaches more often is pushed up the Basel multiplier
ladder, and the capital held is m_c(t) x max(|VaR_t|, mean_60(|VaR|)). Two models with
similar VaR levels can therefore carry materially different capital purely because one
of them keeps tripping the traffic light.

Writes capital_series.json into the dashboard payload directories.
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
STEP = 5  # weekly


def main() -> None:
    src = ROOT / "output" / "dashboard" / "ima_capital_path.json"
    rows = pd.DataFrame(json.loads(src.read_text(encoding="utf-8")))
    rows["date"] = pd.to_datetime(rows["date"])

    series, summary = [], []
    for (method, alpha), g in rows.groupby(["method", "alpha"]):
        g = g.sort_values("date")
        w = g.iloc[::STEP]
        series.append({
            "method": method,
            "alpha": float(alpha),
            "dates": [t.strftime("%Y-%m-%d") for t in w["date"]],
            "capital": [round(float(x), 6) for x in w["ima_capital"]],
            "multiplier": [round(float(x), 2) for x in w["multiplier"]],
        })
        summary.append({
            "method": method,
            "alpha": float(alpha),
            "mean_capital": float(g["ima_capital"].mean()),
            "peak_capital": float(g["ima_capital"].max()),
            "mean_multiplier": float(g["multiplier"].mean()),
            "pct_time_above_base": float(100 * (g["multiplier"] > 3.0).mean()),
        })

    # capital relative to the cheapest method at each level, which is the comparison a
    # risk manager actually makes when choosing between them
    for a in {s["alpha"] for s in summary}:
        at = [s for s in summary if s["alpha"] == a]
        base = min(s["mean_capital"] for s in at)
        for s in at:
            s["vs_cheapest_pct"] = float(100 * (s["mean_capital"] / base - 1))

    payload = {"step_days": STEP, "series": series, "summary": summary}
    out = json.dumps(payload, separators=(",", ":"))
    for d in [ROOT / "output" / "dashboard", ROOT.parent / "dashboard" / "static" / "data"]:
        if d.exists():
            (d / "capital_series.json").write_text(out, encoding="utf-8")
            print(f"wrote {d / 'capital_series.json'}  ({len(out)/1024:.0f} KB)")

    print("\n  mean IMA capital, alpha = 0.01 (units: fraction of book value)")
    for s in sorted([x for x in summary if x["alpha"] == 0.01],
                    key=lambda x: x["mean_capital"]):
        print(f"    {s['method']:<18}{s['mean_capital']:.5f}   "
              f"mult {s['mean_multiplier']:.2f}   "
              f"{s['pct_time_above_base']:5.1f}% of days above base   "
              f"{s['vs_cheapest_pct']:+6.1f}% vs cheapest")


if __name__ == "__main__":
    main()

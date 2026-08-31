"""Per-month out-of-sample coverage + width distribution for all three methods.

Same construction as calibration.py/aci.py (trailing 90d per slot, shift 1 day,
gaussian / rolling-conformal / ACI) but records per-observation coverage and width,
then aggregates: monthly coverage per (target, level, method) and width stats
(mean, P95) per (target, level, method). Exports JSON for the article figure.
"""
import json
import os
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
OUTJ = Path(os.environ.get("FIGURE_DATA_DIR", ART / "figures"))
TARGETS = {
    "load": ("load_fc", "load_act"),
    "solar": ("solar_fc", "solar_act"),
    "wind_on": ("wind_on_fc", "wind_on_act"),
    "wind_off": ("wind_off_fc", "wind_off_act"),
}
LEVELS = [0.80, 0.90, 0.95]
WIN, MINP, GAMMA = 90, 60, 0.02


def main() -> None:
    panel = pd.read_parquet(ART / "panel.parquet")
    idx = panel.index.tz_convert("Europe/Berlin")
    panel["date"] = idx.date
    panel["slot"] = idx.hour * 4 + idx.minute // 15

    monthly_rows, width_rows = [], []
    for name, (fc, act) in TARGETS.items():
        err = panel[act] - panel[fc]
        wide = pd.DataFrame({"err": err, "date": panel["date"], "slot": panel["slot"]}) \
            .dropna().pivot_table(index="date", columns="slot", values="err")
        E = wide.to_numpy()
        dates = pd.DatetimeIndex(wide.index)
        months = dates.to_period("M").astype(str)
        n_dates, n_slots = E.shape

        for lv in LEVELS:
            alpha = 1 - lv
            z = norm.ppf(1 - alpha / 2)
            a_t = np.full(n_slots, alpha)
            rec = {m: {"g": [0, 0], "c": [0, 0], "a": [0, 0]} for m in np.unique(months)}
            widths = {"g": [], "c": [], "a": []}
            for t in range(MINP, n_dates):
                w0 = max(0, t - WIN)
                m = months[t]
                for s in range(n_slots):
                    window = E[w0:t, s]
                    window = window[~np.isnan(window)]
                    y = E[t, s]
                    if len(window) < MINP or np.isnan(y):
                        continue
                    mu, sd = window.mean(), window.std(ddof=1)
                    g_lo, g_hi = mu - z * sd, mu + z * sd
                    c_lo = np.quantile(window, alpha / 2)
                    c_hi = np.quantile(window, 1 - alpha / 2)
                    aa = float(np.clip(a_t[s], 0.001, 0.45))
                    a_lo = np.quantile(window, aa / 2)
                    a_hi = np.quantile(window, 1 - aa / 2)
                    miss_a = 0.0 if a_lo <= y <= a_hi else 1.0
                    a_t[s] = a_t[s] + GAMMA * (alpha - miss_a)
                    for key, lo, hi in [("g", g_lo, g_hi), ("c", c_lo, c_hi), ("a", a_lo, a_hi)]:
                        hit = 1 if lo <= y <= hi else 0
                        rec[m][key][0] += hit
                        rec[m][key][1] += 1
                        widths[key].append(hi - lo)
            for m, d in sorted(rec.items()):
                if d["g"][1] == 0:
                    continue
                monthly_rows.append(
                    {"target": name, "level": lv, "month": m,
                     "gaussian": round(d["g"][0] / d["g"][1], 4),
                     "conformal": round(d["c"][0] / d["c"][1], 4),
                     "aci": round(d["a"][0] / d["a"][1], 4),
                     "n": d["g"][1]}
                )
            for key, label in [("g", "gaussian"), ("c", "conformal"), ("a", "aci")]:
                w = np.array(widths[key])
                width_rows.append(
                    {"target": name, "level": lv, "method": label,
                     "mean_w": round(float(w.mean()), 0),
                     "p95_w": round(float(np.quantile(w, 0.95)), 0),
                     "max_w": round(float(w.max()), 0)}
                )
        print(f"{name} done", flush=True)

    pd.DataFrame(monthly_rows).to_parquet(ART / "monthly_coverage.parquet", index=False)
    pd.DataFrame(width_rows).to_parquet(ART / "width_stats.parquet", index=False)
    OUTJ.mkdir(parents=True, exist_ok=True)
    (OUTJ / "monthly_coverage.json").write_text(json.dumps(monthly_rows))
    (OUTJ / "width_stats.json").write_text(json.dumps(width_rows))
    ws = pd.DataFrame(width_rows)
    print(ws[ws.level == 0.9].to_string(index=False))


if __name__ == "__main__":
    main()

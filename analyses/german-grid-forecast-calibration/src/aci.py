"""Adaptive conformal inference (Gibbs & Candes 2021) as the third method.

Per (target, slot): trailing-90d empirical quantiles as in calibration.py, but the
miscoverage level alpha_t is adapted online: alpha_{t+1} = alpha_t + gamma*(alpha - miss_t),
clipped to [0.001, 0.45]. Under drift this restores near-nominal coverage by construction
(the guarantee conformal's exchangeability assumption loses when the climate shifts).
Appends results to artifacts/coverage_results.parquet (method='aci').
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
ART = ROOT / "artifacts"
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

    rows = []
    for name, (fc, act) in TARGETS.items():
        err = panel[act] - panel[fc]
        wide = pd.DataFrame({"err": err, "date": panel["date"], "slot": panel["slot"]}) \
            .dropna().pivot_table(index="date", columns="slot", values="err")
        E = wide.to_numpy()
        n_dates, n_slots = E.shape
        for lv in LEVELS:
            alpha = 1 - lv
            a_t = np.full(n_slots, alpha)
            covered = widths = valid = 0
            for t in range(MINP, n_dates):
                w0 = max(0, t - WIN)
                for s in range(n_slots):
                    window = E[w0:t, s]
                    window = window[~np.isnan(window)]
                    y = E[t, s]
                    if len(window) < MINP or np.isnan(y):
                        continue
                    a = float(np.clip(a_t[s], 0.001, 0.45))
                    lo = np.quantile(window, a / 2)
                    hi = np.quantile(window, 1 - a / 2)
                    miss = 0.0 if lo <= y <= hi else 1.0
                    a_t[s] = a_t[s] + GAMMA * (alpha - miss)
                    covered += 1 - miss
                    widths += hi - lo
                    valid += 1
            cov = covered / valid
            width = widths / valid
            rows.append({"target": name, "level": lv, "method": "aci",
                         "coverage": cov, "mean_width_mw": width})
            print(f"{name:8s} {lv:.0%} aci cov={cov:.3f} width={width:7.0f} MW")

    res = pd.read_parquet(ART / "coverage_results.parquet")
    res = pd.concat([res[res.method != "aci"], pd.DataFrame(rows)], ignore_index=True)
    res.to_parquet(ART / "coverage_results.parquet", index=False)
    print("\nfull comparison (coverage @ nominal):")
    piv = res.pivot_table(index=["target", "level"], columns="method", values="coverage")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()

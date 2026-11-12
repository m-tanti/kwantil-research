"""One row per forged part: the CMM dimensions, the stated tolerances, and a
small set of process features summarised from the machine's sensor trace.

The feature list is fixed here, chosen on physical grounds before any of them
was compared with a measured dimension: billet temperature, how long the part
waited between heating and forging, how hard and how long the hammers worked,
and where the axes ended up. With 81 parts a model can memorise almost any
feature set, so the list is short and is not tuned.

Reads artifacts/raw/ (see DATA.md); writes artifacts/interim/parts.csv and
artifacts/interim/tolerances.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
OUT = ROOT / "artifacts" / "interim"

FORCE_ON_KN = 50.0          # hammer force above this counts as forging
HOT_C = 600.0               # induction pyrometer above this counts as heating

USE = [
    "Timer Tick [ms]", "Power [kW]", "Force [kN]", "A_ACT_Force [kN]", "DB_ACT_Force [kN]",
    "A_ACTpos [mm]", "L_ACTpos [mm]", "R_ACTpos [mm]", "ForgingBox_Temp",
]
FEATURES = [
    "temp_peak_u1", "temp_peak_f", "temp_at_forge_start", "transfer_s",
    "forge_duration_s", "force_mean", "force_p95", "power_mean",
    "chuck_force_mean_abs", "mandrel_force_mean_abs",
    "a_pos_min", "l_pos_min", "r_pos_min", "box_temp_mean",
]


def read_cmm(path: Path) -> tuple[pd.DataFrame, dict]:
    d = pd.read_excel(path, header=None)
    names = [str(n).strip() for n in d.iloc[0, 4:]]
    nominal = d.iloc[1, 4:].astype(float).to_numpy()
    upper = d.iloc[2, 4:].astype(float).to_numpy()
    lower = d.iloc[3, 4:].astype(float).to_numpy()
    body = d.iloc[4:].reset_index(drop=True)
    parts = pd.DataFrame({
        "part": body.iloc[:, 3].astype(int),
        "run_id": body.iloc[:, 2].astype(int),
        "date": pd.to_datetime(body.iloc[:, 0].astype(str), dayfirst=True).dt.strftime("%Y-%m-%d"),
        "time": body.iloc[:, 1].astype(str),
    })
    for j, n in enumerate(names):
        parts[n] = body.iloc[:, 4 + j].astype(float)
    tol = {n: {"nominal": float(nominal[j]), "lower": float(nominal[j] + lower[j]),
               "upper": float(nominal[j] + upper[j])} for j, n in enumerate(names)}
    return parts, tol


def trace_features(path: Path) -> dict:
    head = pd.read_csv(path, nrows=0, encoding="latin-1").columns
    temp_cols = [c for c in head if c.startswith("TMP_Ind_")]
    d = pd.read_csv(path, encoding="latin-1", usecols=USE + temp_cols)
    t = (d["Timer Tick [ms]"].to_numpy(float) - d["Timer Tick [ms]"].iloc[0]) / 1000.0
    force = d["Force [kN]"].to_numpy(float)
    on = force > FORCE_ON_KN
    if not on.any():
        raise ValueError(f"{path.name}: hammer force never exceeds {FORCE_ON_KN} kN")
    i0, i1 = int(np.argmax(on)), int(len(on) - 1 - np.argmax(on[::-1]))
    u1 = d[[c for c in temp_cols if "U1" in c][0]].to_numpy(float)
    tf = d[[c for c in temp_cols if c.endswith("F [°C]") or "Ind_F" in c][0]].to_numpy(float)
    hot = np.where(u1[: i0 + 1] > HOT_C)[0]
    transfer = float(t[i0] - t[hot[-1]]) if hot.size else float("nan")
    seg = slice(i0, i1 + 1)
    return {
        "n_rows": int(len(d)),
        "temp_peak_u1": float(np.max(u1)),
        "temp_peak_f": float(np.max(tf)),
        "temp_at_forge_start": float(max(u1[i0], tf[i0])),
        "transfer_s": transfer,
        "forge_duration_s": float(t[i1] - t[i0]),
        "force_mean": float(force[seg][on[seg]].mean()),
        "force_p95": float(np.percentile(force[seg][on[seg]], 95)),
        "power_mean": float(d["Power [kW]"].to_numpy(float)[seg].mean()),
        "chuck_force_mean_abs": float(np.abs(d["A_ACT_Force [kN]"].to_numpy(float)[seg]).mean()),
        "mandrel_force_mean_abs": float(np.abs(d["DB_ACT_Force [kN]"].to_numpy(float)[seg]).mean()),
        "a_pos_min": float(d["A_ACTpos [mm]"].to_numpy(float)[seg].min()),
        "l_pos_min": float(d["L_ACTpos [mm]"].to_numpy(float)[seg].min()),
        "r_pos_min": float(d["R_ACTpos [mm]"].to_numpy(float)[seg].min()),
        "box_temp_mean": float(d["ForgingBox_Temp"].to_numpy(float)[seg].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts, tol = read_cmm(RAW / "CMMData.xlsx")
    rows = []
    for p in parts["part"]:
        f = RAW / f"Scope{p:04d}.csv"
        feats = trace_features(f)
        feats["part"] = int(p)
        rows.append(feats)
        print(f"  {f.name}  rows {feats['n_rows']:>6}  forge {feats['forge_duration_s']:.1f}s")
    out = parts.merge(pd.DataFrame(rows), on="part")
    out["order_in_day"] = out.groupby("date").cumcount()
    out.to_csv(OUT / "parts.csv", index=False, lineterminator="\n")
    (OUT / "tolerances.json").write_bytes(json.dumps(tol, indent=1).encode("utf-8"))
    print(f"wrote {len(out)} parts, {len(FEATURES)} features")


if __name__ == "__main__":
    main()

"""Write the figure data for the KW-2026-04 article from the shipped sweep files.

Reads experiments/factorial/{t,gaussian}/as_null_sensitivity.csv and writes
factorial.json and null_sweep.json. Nothing is recomputed here; the numbers are
the sweep's, reshaped for the two figures.

Usage:
    python scripts/export_article_figures.py --out <article>/figures/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.01

NULL_ORDER = [
    ("gaussian", "Gaussian"),
    ("t_df15", "Student-t, df 15"),
    ("t_moment", "Moment-matched t (shipped)"),
    ("t_df8", "Student-t, df 8"),
    ("t_df5", "Student-t, df 5"),
    ("empirical", "Empirical bootstrap"),
]
PANELS = [
    ("conformal_pid", "Conformal-PID"),
    ("parametric", "Parametric EWMA"),
    ("broken_scaled_0.8", "Broken: VaR × 0.8"),
    ("broken_gauss_matched", "Broken: Gaussian tail"),
]
REAL = ["conformal_pid", "conformal_pid_es", "distributional", "historical", "parametric"]


def _null_rows(g: pd.DataFrame, tag: str) -> list[dict]:
    g = g.set_index("null")
    out = []
    for key, label in NULL_ORDER:
        r = g.loc[key]
        df = r["df_used"]
        out.append({
            "null": key,
            "label": label,
            "green": float(r[f"{tag}_green_threshold"]),
            "yellow": float(r[f"{tag}_yellow_threshold"]),
            "zone": r[f"{tag}_zone"],
            "p": float(r[f"{tag}_p_value"]),
            "df": None if pd.isna(df) else float(df),
        })
    return out


def null_sweep(t: pd.DataFrame) -> dict:
    panels = []
    for method, label in PANELS:
        g = t[(t.method == method) & (t.alpha == ALPHA)]
        first = g.iloc[0]
        panels.append({
            "method": method,
            "label": label,
            "alpha": ALPHA,
            "kurt": float(first["kurt_median"]),
            "n_breaches": int(first["n_breaches"]),
            "kupiec_p": float(first["kupiec_pvalue"]),
            "stats": {
                tag: {"obs": float(first[tag]), "nulls": _null_rows(g, tag)}
                for tag in ("z1", "z2")
            },
        })
    counts = {}
    for tag in ("z1", "z2"):
        real = t[t.method.isin(REAL)]
        counts[tag] = []
        for key, label in NULL_ORDER:
            z = real[real.null == key][f"{tag}_zone"].value_counts()
            counts[tag].append({"label": label, **{c: int(z.get(c, 0)) for c in ("green", "yellow", "red")}})
    return {"panels": panels, "counts": counts}


def factorial(t: pd.DataFrame, g: pd.DataFrame) -> dict:
    cells = []
    for cop, frame in (("t", t), ("gaussian", g)):
        for nul in ("gaussian", "t_moment"):
            r = frame[(frame.method == "conformal_pid") & (frame.alpha == ALPHA)
                      & (frame.null == nul)].iloc[0]
            cells.append({"copula": cop, "null": nul, "z1": float(r["z1"]),
                          "green": float(r["z1_green_threshold"]), "zone": r["z1_zone"]})
    c = {(x["copula"], x["null"]): x for x in cells}
    return {
        "cells": cells,
        "copula_effect": round(c[("t", "t_moment")]["z1"] - c[("gaussian", "t_moment")]["z1"], 4),
        "null_effect": round(c[("t", "t_moment")]["green"] - c[("t", "gaussian")]["green"], 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(ROOT / "experiments/factorial/t/as_null_sensitivity.csv")
    g = pd.read_csv(ROOT / "experiments/factorial/gaussian/as_null_sensitivity.csv")
    for name, payload in (("null_sweep.json", null_sweep(t)), ("factorial.json", factorial(t, g))):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

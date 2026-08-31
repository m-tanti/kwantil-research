"""Export figure data as JSON for the article's Svelte components.

The site is Svelte 5 + D3 modules with a passing test asserting zero third-party
requests, so figures are components reading precomputed JSON -- no plotting library
output, no notebook embeds. This writes exactly what the three figures need and
nothing else; anything a figure does not read does not belong in these files.

  fig1_insert_life.json  one insert's whole life: measured wear, prediction, band
  fig2_coverage.json     coverage by construction and wear regime, with Wilson bounds
  fig3_decision.json     the change-threshold dial: two counts per 100 cuts

Numbers are rounded at export. A figure that renders four decimal places of a
quantity estimated from sixteen inserts is lying about its own precision.
"""

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
INTERIM = ROOT / "artifacts" / "interim"
OUT = ROOT / "artifacts" / "figures"

CONSTR = ["G", "G-cal", "C-row", "C-grp", "C-norm", "C-mond"]
LABELS = {
    "G": "Gaussian, in-sample sigma",
    "G-cal": "Gaussian, held-out sigma",
    "C-row": "Conformal, rows split",
    "C-grp": "Conformal, inserts split",
    "C-norm": "Conformal, normalised",
    "C-mond": "Conformal, Mondrian",
}
WEAR_LIMIT = 0.60
MATERIAL = {1: "cast iron", 2: "stainless J45"}


def cluster_bootstrap(sub, col: str, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Resample INSERTS, not rows.

    146 cuts come from 16 inserts, so a row-level binomial interval (Wilson) assumes an
    independence this data does not have. The article spends a whole section on that
    point; using Wilson in its own figures would be the loudest possible own goal.
    """
    rng = np.random.default_rng(seed)
    by_case = {k: v[col].to_numpy() for k, v in sub.groupby("case")}
    cases = np.array(list(by_case.keys()))
    if len(cases) < 2:
        return (float("nan"), float("nan"))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(cases, size=len(cases), replace=True)
        boot[i] = np.concatenate([by_case[p] for p in pick]).mean()
    return tuple(np.percentile(boot, [2.5, 97.5]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def fig1(res: pd.DataFrame) -> dict:
    """One insert's life. Averaged over seeds so the band is the typical one."""
    g = res[res["variant"] == "meta"]
    agg = (g.groupby(["case", "cut_index"])
             .agg(VB=("VB", "first"), pred=("pred", "mean"),
                  q_grp=("q_C-grp", "mean"), q_norm=("q_C-norm", "mean"),
                  q_gauss=("q_G", "mean"), q_gcal=("q_G-cal", "mean"),
                  material=("material", "first"),
                  DOC=("DOC", "first"), feed=("feed", "first"),
                  clip=("smcDC_clip_frac", "first"))
             .reset_index())

    inserts = []
    for case, gg in agg.groupby("case"):
        gg = gg.sort_values("cut_index")
        inserts.append({
            "case": int(case),
            "material": MATERIAL.get(int(gg["material"].iloc[0]), "?"),
            "doc": r(gg["DOC"].iloc[0], 2),
            "feed": r(gg["feed"].iloc[0], 2),
            "cuts": [int(v) for v in gg["cut_index"]],
            "vb": [r(v) for v in gg["VB"]],
            "pred": [r(v) for v in gg["pred"]],
            "grp": [r(v) for v in gg["q_grp"]],
            "norm": [r(v) for v in gg["q_norm"]],
            "gauss": [r(v) for v in gg["q_gauss"]],
            "gcal": [r(v) for v in gg["q_gcal"]],
            "clip": [r(v, 2) for v in gg["clip"]],
        })
    return {"wearLimit": WEAR_LIMIT, "inserts": inserts,
            "note": "prediction and band averaged over calibration draws; "
                    "every point is from an insert the model never saw"}


def fig2(res: pd.DataFrame) -> dict:
    """Coverage by construction and regime, with the Wilson bounds shown, not hidden."""
    out = {"nominal": 90, "wearLimit": WEAR_LIMIT, "interval": "cluster bootstrap over inserts, 2000 resamples", "variants": {}}
    for variant, g in res.groupby("variant"):
        rows = []
        buckets = [("all", g),
                   ("fresh", g[g["VB"] < 0.30]),
                   ("mid", g[(g["VB"] >= 0.30) & (g["VB"] < WEAR_LIMIT)]),
                   ("worn", g[g["VB"] >= WEAR_LIMIT])]
        for name, gg in buckets:
            if gg.empty:
                continue
            n = int(gg.groupby("seed").size().mean())
            n_inserts = int(gg["case"].nunique())
            for c in CONSTR:
                cov = gg.groupby("seed")[f"cov_{c}"].mean()
                blo, bhi = cluster_bootstrap(gg, f"cov_{c}")
                wlo, whi = wilson(int(round(cov.mean() * n)), n)
                rows.append({
                    "regime": name, "constr": c, "label": LABELS[c], "n": n,
                    "nInserts": n_inserts,
                    "cov": r(100 * cov.mean(), 1),
                    "lo": r(100 * blo, 1), "hi": r(100 * bhi, 1),
                    "wilsonLo": r(100 * wlo, 1), "wilsonHi": r(100 * whi, 1),
                    "seedLo": r(100 * cov.quantile(0.05), 1),
                    "seedHi": r(100 * cov.quantile(0.95), 1),
                    "width": r(gg.groupby("seed")[f"wid_{c}"].median().mean()),
                })
        out["variants"][variant] = rows
    return out


def fig3(curve: pd.DataFrame) -> dict:
    """The dial. Counts only -- the reader brings the rates."""
    out = {"limits": {}, "policies": {}}
    for L, a in curve.groupby("limit"):
        pt = a[a["policy"] == "point"].sort_values("param")
        out["limits"][f"{L:.2f}"] = {
            "threshold": [r(v, 2) for v in pt["param"]],
            "waste": [r(v, 1) for v in pt["waste_per_100"]],
            "late": [r(v, 1) for v in pt["late_per_100"]],
        }
        named = a[~a["policy"].isin(["point", "fixed-N"])]
        out["policies"][f"{L:.2f}"] = [
            {"policy": p, "waste": r(w, 1), "late": r(l, 1)}
            for p, w, l in zip(named["policy"], named["waste_per_100"], named["late_per_100"])
        ]
        fx = a[a["policy"] == "fixed-N"].sort_values("param")
        out["limits"][f"{L:.2f}"]["fixedN"] = {
            "n": [int(v) for v in fx["param"]],
            "waste": [r(v, 1) for v in fx["waste_per_100"]],
            "late": [r(v, 1) for v in fx["late_per_100"]],
        }
    out["note"] = ("counts per 100 cuts, averaged over calibration draws; "
                   "rates are supplied by the reader")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(INTERIM / "calibrated.csv")
    curve = pd.read_csv(INTERIM / "decision_curve.csv")

    for name, payload in [("fig1_insert_life", fig1(res)),
                          ("fig2_coverage", fig2(res)),
                          ("fig3_decision", fig3(curve))]:
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {p.name:<24} {p.stat().st_size/1024:>7.1f} KB")


if __name__ == "__main__":
    main()

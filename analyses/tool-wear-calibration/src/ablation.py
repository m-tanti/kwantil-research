"""Ablations: does the conclusion survive the choices we made?

Three choices were made by hand and each one deserves a sweep rather than a defence:

  1. WEAR LIMIT. 0.60 mm is a conventional carbide change point, not a property of the
     dataset. If the headline ("the honest interval is wider than the limit it polices",
     "coverage collapses in the worn regime") only holds at 0.60, it is an artefact and
     must not be published. Swept 0.30 -> 1.00 mm.
  2. DAQ-CORRUPT RUN. idx 17 is kept and flagged (decision 2026-08-22). Every headline is
     recomputed with it excluded, and the difference reported.
  3. CALIBRATION SIZE. 4 inserts held for calibration was a guess. Reported as the
     realised calibration count and its effect on width.

Consumes artifacts/interim/calibrated.csv. Outputs artifacts/interim/ablation_wear_limit.csv
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
CONSTR = ["G", "G-cal", "C-row", "C-grp", "C-norm", "C-mond"]
LIMITS = np.round(np.arange(0.30, 1.001, 0.05), 2)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sweep(res: pd.DataFrame, variant: str) -> pd.DataFrame:
    g = res[res["variant"] == variant]
    rows = []
    for L in LIMITS:
        worn = g[g["VB"] >= L]
        n_per_seed = int(worn.groupby("seed").size().mean()) if len(worn) else 0
        row = {"wear_limit": L, "n_worn_per_seed": n_per_seed,
               "pct_of_data_worn": 100 * len(worn) / len(g)}
        for c in CONSTR:
            row[f"cov_all_{c}"] = 100 * g.groupby("seed")[f"cov_{c}"].mean().mean()
            if len(worn):
                cov = worn.groupby("seed")[f"cov_{c}"].mean()
                row[f"cov_worn_{c}"] = 100 * cov.mean()
                k = int(round(cov.mean() * n_per_seed))
                lo, hi = wilson(k, n_per_seed)
                row[f"wilson_lo_{c}"], row[f"wilson_hi_{c}"] = 100 * lo, 100 * hi
            else:
                row[f"cov_worn_{c}"] = np.nan
                row[f"wilson_lo_{c}"] = row[f"wilson_hi_{c}"] = np.nan
            wid = g.groupby("seed")[f"wid_{c}"].median().mean()
            row[f"width_{c}"] = wid
            row[f"width_over_limit_{c}"] = wid / L
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    res = pd.read_csv(INTERIM / "calibrated.csv")
    line = "=" * 92

    print(f"{line}\nABLATION 1 - WEAR LIMIT SWEEP  (variant 'meta', the fixed-rule model)\n{line}")
    tab = sweep(res, "meta")
    tab.to_csv(INTERIM / "ablation_wear_limit.csv", index=False)

    print(f"\n{'limit':>6} {'n worn':>7} {'%data':>6}  "
          + "".join(f"{c+' worn':>12}" for c in CONSTR))
    for _, r in tab.iterrows():
        print(f"{r['wear_limit']:>6.2f} {r['n_worn_per_seed']:>7.0f} "
              f"{r['pct_of_data_worn']:>5.0f}%  "
              + "".join(f"{r[f'cov_worn_{c}']:>11.1f}%" for c in CONSTR))

    print(f"\nInterval width as a multiple of the wear limit "
          f"(>1.00 means the 90% interval is wider than the limit it polices):")
    print(f"{'limit':>6}  " + "".join(f"{c:>12}" for c in CONSTR))
    for _, r in tab.iterrows():
        print(f"{r['wear_limit']:>6.2f}  "
              + "".join(f"{r[f'width_over_limit_{c}']:>11.2f}x" for c in CONSTR))

    print(f"\n{line}\nIs the headline an artefact of choosing 0.60 mm?\n{line}")
    for c in ("G", "G-cal", "C-grp", "C-norm"):
        sub = tab[tab["n_worn_per_seed"] >= 10]
        print(f"  {c:<7} worn-regime coverage across limits with n>=10: "
              f"{sub[f'cov_worn_{c}'].min():.1f}% to {sub[f'cov_worn_{c}'].max():.1f}% "
              f"(nominal 90%)")
    over = tab[tab["width_over_limit_C-grp"] > 1.0]["wear_limit"]
    if len(over):
        print(f"  C-grp interval exceeds the limit for every limit <= {over.max():.2f} mm")

    print(f"\n{line}\nABLATION 2 - DROP THE DAQ-CORRUPT RUN (idx 17, kept+flagged)\n{line}")
    print(f"{'variant':<9} {'constr':<8} {'with (kept)':>12} {'without':>10} {'delta':>8}")
    for variant in res["variant"].unique():
        g = res[res["variant"] == variant]
        gd = g[g["flag_corrupt"] == 0]
        for c in CONSTR:
            a = 100 * g.groupby("seed")[f"cov_{c}"].mean().mean()
            b = 100 * gd.groupby("seed")[f"cov_{c}"].mean().mean()
            print(f"{variant:<9} {c:<8} {a:>11.1f}% {b:>9.1f}% {b-a:>+7.2f}pp")

    print(f"\n{line}\nABLATION 3 - FEATURE SET (does measuring buy anything?)\n{line}")
    print(f"{'variant':<10} {'RMSE mm':>9} {'C-grp cov':>10} {'C-grp width':>12} "
          f"{'C-norm cov':>11} {'C-norm width':>13}")
    for variant, g in res.groupby("variant", sort=False):
        rmse = np.sqrt(((g["VB"] - g["pred"]) ** 2).groupby(g["seed"]).mean()).mean()
        print(f"{variant:<10} {rmse:>9.4f} "
              f"{100*g.groupby('seed')['cov_C-grp'].mean().mean():>9.1f}% "
              f"{g.groupby('seed')['wid_C-grp'].median().mean():>12.3f} "
              f"{100*g.groupby('seed')['cov_C-norm'].mean().mean():>10.1f}% "
              f"{g.groupby('seed')['wid_C-norm'].median().mean():>13.3f}")

    print(f"\nwrote {INTERIM / 'ablation_wear_limit.csv'}")


if __name__ == "__main__":
    main()

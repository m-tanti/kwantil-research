"""A1.1 - characterise the NASA Milling Data Set.

Answers the three questions that gate the build:
  1. how many groups (cases) are there, and how are runs distributed across them
  2. how complete is the flank-wear (VB) label
  3. what does the raw signal look like, and what has to happen to get per-cut features

Writes artifacts/interim/index.csv (one row per run, metadata + label) and prints a report.
"""

import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
def _find_mill_mat(root: pathlib.Path) -> pathlib.Path:
    """Locate mill.mat under artifacts/raw, wherever the user extracted it.

    The distributed zip contains a "3. Milling" folder, but extracting the file
    on its own is at least as common. Both work; anything else fails with the
    download link rather than a bare traceback.
    """
    raw = root / "artifacts" / "raw"
    for candidate in (raw / "3. Milling" / "mill.mat", raw / "mill.mat"):
        if candidate.exists():
            return candidate
    found = sorted(raw.rglob("mill.mat")) if raw.exists() else []
    if found:
        return found[0]
    raise SystemExit(
        f"mill.mat not found under {raw}\n"
        "  This repository ships no data. Download the NASA Milling Data Set:\n"
        "    https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip\n"
        "  and extract mill.mat to artifacts/raw/ (the enclosing folder is optional).\n"
        "  Full guide: DATA.md"
    )


RAW = _find_mill_mat(ROOT)
INTERIM = ROOT / "artifacts" / "interim"

SIGNALS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
SCALARS = ["case", "run", "VB", "time", "DOC", "feed", "material"]


def load() -> np.ndarray:
    mat = loadmat(str(RAW), struct_as_record=True)
    key = [k for k in mat if not k.startswith("__")]
    print(f"top-level keys: {key}")
    mill = mat["mill"]
    print(f"mill shape {mill.shape}, dtype fields: {list(mill.dtype.names)}")
    return mill[0]


def scalar(entry, field):
    """Scalars are stored as 1x1 arrays; missing values come back empty or NaN."""
    v = entry[field]
    a = np.asarray(v).ravel()
    if a.size == 0:
        return np.nan
    try:
        return float(a[0])
    except (TypeError, ValueError):
        return np.nan


def build_index(mill) -> pd.DataFrame:
    rows = []
    for i, entry in enumerate(mill):
        row = {f: scalar(entry, f) for f in SCALARS}
        row["idx"] = i
        for s in SIGNALS:
            sig = np.asarray(entry[s]).ravel()
            row[f"{s}_n"] = sig.size
            row[f"{s}_std"] = float(np.std(sig)) if sig.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    line = "=" * 72
    print(f"\n{line}\nSHAPE\n{line}")
    print(f"entries (runs) total     : {len(df)}")
    print(f"distinct cases (groups)  : {df['case'].nunique()}")

    print(f"\n{line}\nLABEL COMPLETENESS - VB (flank wear)\n{line}")
    lab = df["VB"].notna()
    print(f"runs with a VB value     : {lab.sum()} / {len(df)}  ({100*lab.mean():.1f}%)")
    print(f"runs with VB missing     : {(~lab).sum()}")
    print(f"VB range                 : {df['VB'].min():.4f} .. {df['VB'].max():.4f}")

    print(f"\n{line}\nGROUP STRUCTURE - runs and labelled runs per case\n{line}")
    g = df.groupby("case").agg(
        runs=("run", "size"),
        labelled=("VB", "count"),
        material=("material", "first"),
        DOC=("DOC", "first"),
        feed=("feed", "first"),
        VB_max=("VB", "max"),
    )
    g["label_pct"] = (100 * g["labelled"] / g["runs"]).round(0)
    print(g.to_string())

    print(f"\nlabelled runs per case: min {g['labelled'].min():.0f}, "
          f"median {g['labelled'].median():.0f}, max {g['labelled'].max():.0f}")

    print(f"\n{line}\nEXPERIMENTAL DESIGN - the conditions that vary\n{line}")
    cond = df.groupby(["material", "DOC", "feed"]).agg(
        cases=("case", "nunique"), runs=("run", "size"), labelled=("VB", "count")
    )
    print(cond.to_string())

    print(f"\n{line}\nSIGNALS\n{line}")
    for s in SIGNALS:
        n = df[f"{s}_n"]
        print(f"{s:<14} samples/cut: min {n.min():.0f} median {n.median():.0f} max {n.max():.0f}"
              f"   | std across runs: {df[f'{s}_std'].min():.4f} .. {df[f'{s}_std'].max():.4f}")

    zero = [s for s in SIGNALS if (df[f"{s}_std"] < 1e-9).any()]
    if zero:
        print(f"\nWARNING flat/dead channel in some runs: {zero}")

    print(f"\n{line}\nWHAT THIS MEANS FOR THE SPLIT\n{line}")
    n_groups = df["case"].nunique()
    lab_per_case = g["labelled"]
    print(f"group-aware splitting has {n_groups} groups to work with.")
    for frac, name in [(0.3, "30% calibration"), (0.25, "25% calibration")]:
        k = max(1, int(round(n_groups * frac)))
        approx_n = int(lab_per_case.median() * k)
        print(f"  {name}: ~{k} cases held out -> ~{approx_n} labelled points in calibration")
    print("\nCoverage estimated on that many points carries a wide Wilson interval.")
    print("That width is a finding to publish, not a defect to hide (S3.1 constraint 1).")


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    mill = load()
    df = build_index(mill)
    df.to_csv(INTERIM / "index.csv", index=False)
    report(df)
    print(f"\nwrote {INTERIM / 'index.csv'}")

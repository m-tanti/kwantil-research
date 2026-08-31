"""A1.1 - diagnose the signal anomalies found by characterise.py.

Two things looked wrong:
  1. per-run signal std ranging up to ~1e32 (physically impossible for volts)
  2. signal length varying: 9000 samples for most runs, 15360 for some

Establish which runs, which channels, and whether it is corruption, saturation,
or a header/scaling issue -- before any feature extraction is designed.
"""

import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

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
SIGNALS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

mill = loadmat(str(RAW), struct_as_record=True)["mill"][0]

rows = []
for i, e in enumerate(mill):
    r = {
        "idx": i,
        "case": float(np.asarray(e["case"]).ravel()[0]),
        "run": float(np.asarray(e["run"]).ravel()[0]),
    }
    for s in SIGNALS:
        x = np.asarray(e[s], dtype=float).ravel()
        r[f"{s}_n"] = x.size
        r[f"{s}_absmax"] = np.abs(x).max() if x.size else np.nan
        r[f"{s}_nonfinite"] = int((~np.isfinite(x)).sum())
    rows.append(r)

df = pd.DataFrame(rows)
line = "=" * 72

print(f"{line}\n1. SIGNAL LENGTH\n{line}")
for s in SIGNALS:
    vc = df[f"{s}_n"].value_counts().sort_index()
    print(f"{s:<14} {dict(vc)}")
odd = df[df["smcAC_n"] != 9000][["idx", "case", "run", "smcAC_n"]]
print(f"\nruns not 9000 samples: {len(odd)}")
print(odd.to_string(index=False))

print(f"\n{line}\n2. NON-FINITE VALUES\n{line}")
for s in SIGNALS:
    bad = df[df[f"{s}_nonfinite"] > 0]
    print(f"{s:<14} runs containing NaN/Inf: {len(bad)}"
          + (f"  -> idx {list(bad['idx'])}" if len(bad) else ""))

print(f"\n{line}\n3. MAGNITUDE - plausible range for a +/-10 V DAQ channel\n{line}")
for s in SIGNALS:
    a = df[f"{s}_absmax"]
    finite = a[np.isfinite(a)]
    print(f"{s:<14} |max| median {np.median(finite):>10.3f}   "
          f"p95 {np.percentile(finite, 95):>10.3f}   max {finite.max():>12.4g}")
    over = df[a > 100]
    if len(over):
        print(f"{'':<14} runs with |max| > 100: {len(over)} -> "
              f"idx {list(over['idx'])[:12]}{'...' if len(over) > 12 else ''}")

print(f"\n{line}\n4. THE WORST RUN, CHANNEL BY CHANNEL\n{line}")
worst_idx = int(df["vib_spindle_absmax"].idxmax())
e = mill[worst_idx]
print(f"idx {worst_idx}  case {df.loc[worst_idx, 'case']:.0f}  run {df.loc[worst_idx, 'run']:.0f}")
for s in SIGNALS:
    x = np.asarray(e[s], dtype=float).ravel()
    q = np.percentile(np.abs(x), [50, 99, 99.9])
    print(f"  {s:<13} n={x.size:<6} median|x| {q[0]:>10.4g}  p99 {q[1]:>10.4g}  "
          f"p99.9 {q[2]:>10.4g}  max {np.abs(x).max():>12.4g}")

print(f"\n{line}\n5. HOW MANY RUNS ARE AFFECTED\n{line}")
suspect = df[(df[[f'{s}_absmax' for s in SIGNALS]] > 100).any(axis=1)]
print(f"runs with any channel |max| > 100 : {len(suspect)} / {len(df)}")
print(f"cases affected                    : {sorted(suspect['case'].unique().astype(int))}")
df.to_csv(ROOT / "artifacts" / "interim" / "signal_diagnostics.csv", index=False)
print(f"\nwrote artifacts/interim/signal_diagnostics.csv")

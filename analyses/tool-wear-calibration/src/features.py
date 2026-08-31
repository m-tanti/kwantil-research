"""Feature extraction: mill.mat -> one row per run.

Design notes (from the NASA readme, Goebel & Agogino 2007):
  * AE and vibration channels are already RMS-conditioned (dT = 8 ms) and sampled
    at 250 Hz, so this is windowing and summarising, not high-frequency DSP.
  * Each 36 s cut contains an entry ramp, a steady plateau and an exit ramp
    (readme figs 7-8). Features are computed on the plateau only -- entry and exit
    transients are a property of the experiment, not of tool wear.
  * Two runs (idx 17, 94) carry impossible values in the tail: AE and vibration were
    amplified into a +/-5 V range and DC current tops out near 10 V, so samples beyond
    +/-20 V are DAQ corruption. They are masked, not silently averaged in, and the
    masked count is reported so the article can state it.
  * smcDC CLIPS at the 10 V converter ceiling in a minority of runs, and it does so at
    high flank wear -- a worn insert draws more torque, more current, and the channel
    rails. Clipping is therefore not noise to be smoothed away: it is a wear signal, and
    it is simultaneously a loss of resolution in the exact regime where the tool-change
    decision is hardest. Every channel gets an explicit clip_frac feature so the model
    can use it and the article can show it.
  * The steady window is located on the AC current envelope, NOT the DC channel: the DC
    channel is the one that rails, and a railed channel has no envelope to find.

Outputs artifacts/interim/features.csv
"""

import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
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
PLAUSIBLE_V = 20.0          # beyond this is DAQ corruption, not signal
PLATEAU_FRAC = 0.5          # plateau = above this fraction of the run's p95 level
TRIM = 0.10                 # drop this much from each end of the detected plateau
FS = 250.0                  # Hz
SMOOTH_N = 125              # 0.5 s boxcar for the AC envelope
# converter ceilings: current channels run to +/-10 V, AE and vibration to +/-5 V
CLIP_V = {"smcAC": 10.0, "smcDC": 10.0,
          "vib_table": 5.0, "vib_spindle": 5.0, "AE_table": 5.0, "AE_spindle": 5.0}


def steady_window(ac: np.ndarray) -> slice:
    """Locate the steady cutting plateau from the AC current envelope.

    Deliberately not the DC channel: DC is the one that rails at the converter
    ceiling, and a railed channel has a flat envelope with no detectable plateau.
    """
    x = np.abs(np.nan_to_num(ac, nan=0.0))
    if x.max() <= 0:
        n = len(ac)
        return slice(n // 4, 3 * n // 4)
    env = np.convolve(x, np.ones(SMOOTH_N) / SMOOTH_N, mode="same")
    lvl = np.percentile(env, 95)
    above = np.where(env > PLATEAU_FRAC * lvl)[0]
    if above.size < 100:                       # no clear plateau -> use the middle half
        n = len(ac)
        return slice(n // 4, 3 * n // 4)
    lo, hi = above[0], above[-1]
    span = hi - lo
    return slice(int(lo + TRIM * span), int(hi - TRIM * span) + 1)


def summarise(x: np.ndarray, prefix: str) -> dict:
    """Plain, defensible summaries. Nothing here needs a paragraph to justify."""
    keys = ("mean", "std", "rms", "max", "p90", "kurt", "skew", "slope", "clip_frac")
    x = x[np.isfinite(x)]
    if x.size < 10:
        return {f"{prefix}_{k}": np.nan for k in keys}

    ceiling = CLIP_V[prefix]
    clip_frac = float(np.mean(np.abs(x) >= 0.999 * ceiling))

    t = np.arange(x.size) / FS
    slope = np.polyfit(t, x, 1)[0] if x.size > 2 else np.nan
    if np.std(x) < 1e-9:          # railed flat: higher moments are undefined, not zero
        return {f"{prefix}_{k}": np.nan for k in keys} | {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": 0.0,
            f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
            f"{prefix}_max": float(np.max(np.abs(x))),
            f"{prefix}_p90": float(np.percentile(np.abs(x), 90)),
            f"{prefix}_slope": 0.0,
            f"{prefix}_clip_frac": clip_frac,
        }
    return {f"{prefix}_clip_frac": clip_frac} | {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=1)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
        f"{prefix}_max": float(np.max(np.abs(x))),
        f"{prefix}_p90": float(np.percentile(np.abs(x), 90)),
        f"{prefix}_kurt": float(kurtosis(x)),
        f"{prefix}_skew": float(skew(x)),
        f"{prefix}_slope": float(slope),
    }


def scalar(entry, field):
    a = np.asarray(entry[field]).ravel()
    if a.size == 0:
        return np.nan
    try:
        return float(a[0])
    except (TypeError, ValueError):
        return np.nan


def main() -> None:
    mill = loadmat(str(RAW), struct_as_record=True)["mill"][0]
    rows, masked_total, masked_runs = [], 0, []

    for i, e in enumerate(mill):
        row = {
            "idx": i,
            "case": scalar(e, "case"),
            "run": scalar(e, "run"),
            "VB": scalar(e, "VB"),
            "time": scalar(e, "time"),
            "DOC": scalar(e, "DOC"),
            "feed": scalar(e, "feed"),
            "material": scalar(e, "material"),
        }

        chans, masked_here = {}, 0
        for s in SIGNALS:
            x = np.asarray(e[s], dtype=float).ravel().copy()
            bad = ~np.isfinite(x) | (np.abs(x) > PLAUSIBLE_V)
            masked_here += int(bad.sum())
            x[bad] = np.nan
            chans[s] = x
        if masked_here:
            masked_total += masked_here
            masked_runs.append((i, int(row["case"]), int(row["run"]), masked_here))

        win = steady_window(chans["smcAC"])
        row["window_start"] = win.start
        row["window_stop"] = win.stop
        row["window_n"] = win.stop - win.start
        row["window_sec"] = (win.stop - win.start) / FS
        row["masked_samples"] = masked_here

        for s in SIGNALS:
            row.update(summarise(chans[s][win], s))
        rows.append(row)

    df = pd.DataFrame(rows)

    # DAQ corruption: kept, not dropped (decision 2026-08-22), but flagged so every
    # downstream result can be re-run without them and the difference reported.
    df["corrupt_frac"] = df["masked_samples"] / 9000.0
    df["flag_corrupt"] = (df["corrupt_frac"] > 0.01).astype(int)

    # cut index within case, and cumulative cutting time -- what a shop actually knows
    df = df.sort_values(["case", "run"]).reset_index(drop=True)
    df["cut_index"] = df.groupby("case").cumcount() + 1
    df["cuts_in_case"] = df.groupby("case")["case"].transform("size")

    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM / "features.csv", index=False)

    line = "=" * 72
    print(f"{line}\nFEATURE EXTRACTION\n{line}")
    print(f"runs processed        : {len(df)}")
    print(f"feature columns       : {sum(c.count('_') > 0 and c.split('_')[0] in ('smcAC','smcDC','vib','AE') or c.startswith(tuple(SIGNALS)) for c in df.columns)}")
    print(f"labelled (VB present) : {df['VB'].notna().sum()}")
    print(f"\nsteady window length  : median {df['window_sec'].median():.1f} s "
          f"(min {df['window_sec'].min():.1f}, max {df['window_sec'].max():.1f}) "
          f"out of {9000/FS:.0f} s recorded")
    print(f"\nDAQ-corrupt samples masked: {masked_total}")
    for idx, case, run, n in masked_runs:
        print(f"  idx {idx:>3}  case {case:>2}  run {run:>2}  -> {n} samples "
              f"({100*n/9000:.2f}% of the run)")
    print(f"\n{line}\nCONVERTER CLIPPING - signal lost at the ceiling\n{line}")
    for s in SIGNALS:
        cf = df[f"{s}_clip_frac"]
        n_any = int((cf > 0).sum())
        n_bad = int((cf > 0.5).sum())
        print(f"{s:<13} runs with any clipped samples: {n_any:>3}   "
              f"runs >50% clipped: {n_bad:>3}   max clip fraction {cf.max():.2f}")

    hi = df[df["smcDC_clip_frac"] > 0.5]
    if len(hi):
        lab = hi["VB"].dropna()
        rest = df.loc[df["smcDC_clip_frac"] <= 0.5, "VB"].dropna()
        print(f"\nsmcDC heavily clipped in {len(hi)} runs, cases "
              f"{sorted(hi['case'].unique().astype(int))}")
        print(f"  median VB where clipped     : {lab.median():.3f} mm  (n={len(lab)})")
        print(f"  median VB where not clipped : {rest.median():.3f} mm  (n={len(rest)})")
        print("  -> the load channel saturates in the high-wear regime: resolution is")
        print("     lost exactly where the tool-change decision is hardest.")

    nan_feat = df.filter(regex="_(mean|rms|kurt)$").isna().sum().sum()
    print(f"\nNaN feature cells after masking: {nan_feat}")
    print(f"\nwrote {INTERIM / 'features.csv'}  shape {df.shape}")


if __name__ == "__main__":
    main()

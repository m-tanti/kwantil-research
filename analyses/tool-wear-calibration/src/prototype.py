"""Prototype: three ways to promise 90%, evaluated honestly on unseen inserts.

Two models, because the point of the article is what a shop gains by measuring:
  M0  "the fixed rule"  -- cut count + cutting conditions only. This is the folklore
      interval every shop already runs, fitted. No sensors.
  M1  "sensor-informed" -- M0 plus the per-cut signal features.

Three interval constructions, all nominally 90%:
  G     Gaussian  : point +/- 1.645 * training RMSE. The standard habit.
  C-row Conformal, rows split at random. THE TRAP: cuts from one insert land in both
        fit and calibration, so the calibration residuals are not out-of-sample.
  C-grp Conformal, split by insert (case). The honest construction.

Evaluation is leave-one-insert-out throughout: every reported number is measured on an
insert the model has never seen, which is the only question a shop actually asks.

Outputs artifacts/interim/predictions.csv and a coverage report.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "artifacts" / "interim"

ALPHA = 0.10                       # nominal 90% intervals
SEED = 42
CAL_CASES = 4                      # inserts held out for calibration inside each fold
META = ["DOC", "feed", "material", "cut_index"]
WEAR_LIMIT = 0.60                  # mm; a common flank-wear change point for carbide


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. With 16 inserts, this is not optional decoration."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=-1)),
    ])


def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(INTERIM / "features.csv")
    df = df[df["VB"].notna()].copy()
    sensor = [c for c in df.columns if any(
        c.startswith(p) for p in
        ("smcAC_", "smcDC_", "vib_table_", "vib_spindle_", "AE_table_", "AE_spindle_"))]
    return df, sensor


def evaluate(df: pd.DataFrame, feats: list[str], label: str, rng: np.random.Generator):
    """Leave-one-insert-out. Returns a row per test point with all three intervals."""
    cases = np.sort(df["case"].unique())
    out = []

    for held in cases:
        te = df[df["case"] == held]
        tr = df[df["case"] != held]
        if len(tr) < 20 or te.empty:
            continue

        other = np.sort(tr["case"].unique())
        cal_cases = rng.choice(other, size=min(CAL_CASES, len(other) - 1), replace=False)

        # --- group-aware calibration: whole inserts held out -----------------
        fit_g = tr[~tr["case"].isin(cal_cases)]
        cal_g = tr[tr["case"].isin(cal_cases)]
        m_g = make_model().fit(fit_g[feats], fit_g["VB"])
        r_g = np.abs(cal_g["VB"].to_numpy() - m_g.predict(cal_g[feats]))
        n_g = len(r_g)
        q_g = np.quantile(r_g, min(1.0, np.ceil((n_g + 1) * (1 - ALPHA)) / n_g),
                          method="higher")

        # --- row-level calibration: same size, but rows drawn at random ------
        idx = rng.permutation(len(tr))
        n_cal = len(cal_g)
        cal_r = tr.iloc[idx[:n_cal]]
        fit_r = tr.iloc[idx[n_cal:]]
        m_r = make_model().fit(fit_r[feats], fit_r["VB"])
        r_r = np.abs(cal_r["VB"].to_numpy() - m_r.predict(cal_r[feats]))
        n_r = len(r_r)
        q_r = np.quantile(r_r, min(1.0, np.ceil((n_r + 1) * (1 - ALPHA)) / n_r),
                          method="higher")

        # --- Gaussian: 1.645 * in-training RMSE ------------------------------
        m_all = make_model().fit(tr[feats], tr["VB"])
        resid_tr = tr["VB"].to_numpy() - m_all.predict(tr[feats])
        q_gauss = 1.645 * np.sqrt(np.mean(resid_tr**2))

        pred = m_all.predict(te[feats])
        for i, (_, row) in enumerate(te.iterrows()):
            out.append({
                "model": label, "case": held, "run": row["run"], "cut_index": row["cut_index"],
                "material": row["material"], "DOC": row["DOC"], "feed": row["feed"],
                "VB": row["VB"], "pred": pred[i],
                "smcDC_clip_frac": row["smcDC_clip_frac"],
                "q_gauss": q_gauss, "q_row": q_r, "q_grp": q_g,
                "n_cal_grp": n_g, "n_cal_row": n_r,
            })

    res = pd.DataFrame(out)
    for name, q in [("G", "q_gauss"), ("C-row", "q_row"), ("C-grp", "q_grp")]:
        res[f"cov_{name}"] = (np.abs(res["VB"] - res["pred"]) <= res[q]).astype(int)
        res[f"wid_{name}"] = 2 * res[q]
    return res


def regime(v: float) -> str:
    if v < 0.30:
        return "1 fresh   (VB<0.30)"
    if v < WEAR_LIMIT:
        return "2 mid     (0.30-0.60)"
    return "3 worn    (VB>=0.60)"


def report(res: pd.DataFrame) -> None:
    line = "=" * 78
    for label, g in res.groupby("model", sort=False):
        rmse = np.sqrt(np.mean((g["VB"] - g["pred"]) ** 2))
        mae = np.mean(np.abs(g["VB"] - g["pred"]))
        print(f"\n{line}\nMODEL {label}   (leave-one-insert-out, n={len(g)})\n{line}")
        print(f"point accuracy: RMSE {rmse:.4f} mm   MAE {mae:.4f} mm   "
              f"VB range {g['VB'].min():.2f}-{g['VB'].max():.2f} mm")
        print(f"\n{'construction':<8} {'coverage':>9} {'95% Wilson':>16} "
              f"{'mean width':>11} {'median width':>13}")
        print("-" * 62)
        for name in ("G", "C-row", "C-grp"):
            k, n = int(g[f"cov_{name}"].sum()), len(g)
            lo, hi = wilson(k, n)
            print(f"{name:<8} {100*k/n:>8.1f}% {f'[{100*lo:.1f}, {100*hi:.1f}]':>16} "
                  f"{g[f'wid_{name}'].mean():>10.3f} {g[f'wid_{name}'].median():>13.3f}")
        print("            nominal 90%")

        print(f"\ncoverage by wear regime (the regime that matters is the last one):")
        g = g.assign(regime=g["VB"].map(regime))
        print(f"{'regime':<24} {'n':>4}  " + "  ".join(f"{c:>18}" for c in ("G", "C-row", "C-grp")))
        for reg, gg in g.groupby("regime"):
            cells = []
            for name in ("G", "C-row", "C-grp"):
                k, n = int(gg[f"cov_{name}"].sum()), len(gg)
                lo, hi = wilson(k, n)
                cells.append(f"{100*k/n:>5.0f}% [{100*lo:>3.0f},{100*hi:>3.0f}]")
            print(f"{reg:<24} {len(gg):>4}  " + "  ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    df, sensor = load()
    print(f"labelled runs {len(df)} across {df['case'].nunique()} inserts; "
          f"{len(sensor)} sensor features")

    res = pd.concat([
        evaluate(df, META, "M0 fixed-rule", np.random.default_rng(SEED)),
        evaluate(df, META + sensor, "M1 sensor-informed", np.random.default_rng(SEED)),
    ], ignore_index=True)

    res.to_csv(INTERIM / "predictions.csv", index=False)
    report(res)
    print(f"\nwrote {INTERIM / 'predictions.csv'}")

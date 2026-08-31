"""Five ways to promise 90%, four feature sets, repeated over seeds.

Supersedes prototype.py. Everything is leave-one-insert-out: every number is measured
on an insert the model has never seen, which is the only question a shop asks.

CONSTRUCTIONS
  G       Gaussian, point +/- 1.645 * IN-SAMPLE training RMSE. The naive habit, and a
          random forest's in-sample residuals are optimistically small because trees
          memorise; kept because practitioners really do this, not as the fair comparison.
  G-cal   Gaussian, point +/- 1.645 * sd of the SAME held-out calibration residuals the
          conformal constructions use. This is the like-for-like Gaussian: the only thing
          separating it from C-grp is the normal-shape assumption rather than the data.
  C-row   Conformal, rows split at random. THE TRAP -- cuts from one insert land in
          both fit and calibration, so calibration residuals are not out-of-sample.
  C-grp   Conformal, split by insert. Honest, but only marginally valid: it promises
          90% on average over inserts, not 90% in the worn regime.
  C-norm  Normalized conformal, split by insert. Conformity score |y-mu|/sigma(x),
          with sigma learned from the fit set's OUT-OF-BAG residuals (in-sample
          residuals would be optimistically small and would poison the scale).
          Widens the interval where the model is locally unreliable.
  C-mond  Mondrian conformal, split by insert, bucketed by PREDICTED wear against the
          limit. Separate quantile each side. Directly targets conditional coverage in
          the regime the decision lives in -- at the cost of splitting an already small
          calibration set, which is itself worth showing.

FEATURE SETS (the ablation)
  meta     cut count + cutting conditions. The folklore change-interval, fitted.
  +clip    meta + the six clip_frac channels only. Tests whether the converter railing
           is by itself a wear signal -- cheap to deploy, and if it works it is a
           finding a shop can act on with no new instrumentation.
  +compact meta + mean/std/rms per channel (18 features).
  +all     meta + all 54 signal features.

Feature sets are FIXED, not selected per fold: selecting inside the loop on this many
features with this little data would leak and would not survive review.

Outputs artifacts/interim/calibrated.csv
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

ALPHA = 0.10
N_SEEDS = 25
CAL_CASES = 4
WEAR_LIMIT = 0.60           # mm -- swept in ablation.py, fixed here only for Mondrian
EPS = 1e-3
CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
META = ["DOC", "feed", "material", "cut_index"]


def rf(**kw):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     n_jobs=-1, **kw)),
    ])


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample conformal quantile: ceil((n+1)(1-alpha))/n, taken from above."""
    n = len(scores)
    if n == 0:
        return np.inf
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, lvl, method="higher"))


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    clip = [f"{c}_clip_frac" for c in CHANNELS]
    compact = [f"{c}_{s}" for c in CHANNELS for s in ("mean", "std", "rms")]
    allsig = [c for c in df.columns if any(c.startswith(f"{ch}_") for ch in CHANNELS)]
    return {
        "meta": META,
        "+clip": META + clip,
        "+compact": META + compact,
        "+all": META + allsig,
    }


def one_fold(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str],
             rng: np.random.Generator) -> pd.DataFrame:
    other = np.sort(tr["case"].unique())
    cal_cases = rng.choice(other, size=min(CAL_CASES, len(other) - 1), replace=False)

    # ---- insert-level calibration -------------------------------------------------
    fit_g = tr[~tr["case"].isin(cal_cases)]
    cal_g = tr[tr["case"].isin(cal_cases)]
    m_g = rf(random_state=0, oob_score=True, bootstrap=True).fit(fit_g[feats], fit_g["VB"])

    mu_cal = m_g.predict(cal_g[feats])
    res_cal = np.abs(cal_g["VB"].to_numpy() - mu_cal)
    q_grp = conformal_q(res_cal)

    # sigma model trained on OUT-OF-BAG residuals of the fit set
    oob = m_g.named_steps["rf"].oob_prediction_
    fit_X = m_g.named_steps["impute"].transform(fit_g[feats])
    s_model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                    random_state=0, n_jobs=-1)
    s_model.fit(fit_X, np.abs(fit_g["VB"].to_numpy() - oob))

    def sigma(X_df):
        return np.maximum(s_model.predict(m_g.named_steps["impute"].transform(X_df)), EPS)

    q_norm = conformal_q(res_cal / sigma(cal_g[feats]))

    # Mondrian: separate quantile either side of the limit, on PREDICTED wear
    hot = mu_cal >= WEAR_LIMIT
    q_m_hi = conformal_q(res_cal[hot]) if hot.sum() >= 5 else q_grp
    q_m_lo = conformal_q(res_cal[~hot]) if (~hot).sum() >= 5 else q_grp

    # ---- row-level calibration: same size, rows drawn at random (the trap) ---------
    idx = rng.permutation(len(tr))
    n_cal = len(cal_g)
    cal_r, fit_r = tr.iloc[idx[:n_cal]], tr.iloc[idx[n_cal:]]
    m_r = rf(random_state=0).fit(fit_r[feats], fit_r["VB"])
    q_row = conformal_q(np.abs(cal_r["VB"].to_numpy() - m_r.predict(cal_r[feats])))

    # ---- full-data model for prediction + the Gaussian habit ----------------------
    m_all = rf(random_state=0).fit(tr[feats], tr["VB"])
    q_gauss = 1.645 * np.sqrt(np.mean((tr["VB"].to_numpy() - m_all.predict(tr[feats])) ** 2))
    # like-for-like Gaussian: same held-out calibration inserts as the conformal methods
    q_gauss_cal = 1.645 * float(np.std(cal_g["VB"].to_numpy() - mu_cal, ddof=1))

    pred = m_all.predict(te[feats])
    sig_te = sigma(te[feats])
    out = te[["case", "run", "cut_index", "material", "DOC", "feed", "VB",
              "smcDC_clip_frac", "flag_corrupt"]].copy()
    out["pred"] = pred
    out["sigma"] = sig_te
    out["q_G"] = q_gauss
    out["q_G-cal"] = q_gauss_cal
    out["q_C-row"] = q_row
    out["q_C-grp"] = q_grp
    out["q_C-norm"] = q_norm * sig_te
    out["q_C-mond"] = np.where(pred >= WEAR_LIMIT, q_m_hi, q_m_lo)
    out["n_cal"] = len(cal_g)
    return out


def run(df: pd.DataFrame) -> pd.DataFrame:
    sets = feature_sets(df)
    frames = []
    for name, feats in sets.items():
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + seed)
            for held in np.sort(df["case"].unique()):
                te, tr = df[df["case"] == held], df[df["case"] != held]
                if te.empty or len(tr) < 20:
                    continue
                f = one_fold(tr, te, feats, rng)
                f["variant"], f["seed"] = name, seed
                frames.append(f)
        print(f"  done {name} ({len(feats)} features)")
    res = pd.concat(frames, ignore_index=True)
    for c in ("G", "G-cal", "C-row", "C-grp", "C-norm", "C-mond"):
        res[f"cov_{c}"] = (np.abs(res["VB"] - res["pred"]) <= res[f"q_{c}"]).astype(int)
        res[f"wid_{c}"] = 2 * res[f"q_{c}"]
    return res


CONSTR = ["G", "G-cal", "C-row", "C-grp", "C-norm", "C-mond"]


def summarise(res: pd.DataFrame) -> None:
    line = "=" * 86
    print(f"\n{line}\nCOVERAGE AT NOMINAL 90% - mean over {N_SEEDS} seeds [5th, 95th pct across seeds]\n{line}")
    for variant, g in res.groupby("variant", sort=False):
        per_seed = g.groupby("seed")
        rmse = np.sqrt(((g["VB"] - g["pred"]) ** 2).groupby(g["seed"]).mean()).mean()
        print(f"\n{variant:<9} RMSE {rmse:.4f} mm")
        print(f"  {'constr':<8} {'coverage':>9} {'across seeds':>16} {'med width':>11} {'width/limit':>12}")
        for c in CONSTR:
            cov = per_seed[f"cov_{c}"].mean()
            wid = per_seed[f"wid_{c}"].median()
            print(f"  {c:<8} {100*cov.mean():>8.1f}% "
                  f"{f'[{100*cov.quantile(.05):.1f}, {100*cov.quantile(.95):.1f}]':>16} "
                  f"{wid.mean():>10.3f} {wid.mean()/WEAR_LIMIT:>11.2f}x")

    print(f"\n{line}\nCOVERAGE IN THE WORN REGIME (VB >= {WEAR_LIMIT} mm) - where the decision is\n{line}")
    worn = res[res["VB"] >= WEAR_LIMIT]
    print(f"  n = {worn.groupby(['variant','seed']).size().iloc[0]} points per seed\n")
    print(f"  {'variant':<9} " + "".join(f"{c:>14}" for c in CONSTR))
    for variant, g in worn.groupby("variant", sort=False):
        cells = []
        for c in CONSTR:
            cov = g.groupby("seed")[f"cov_{c}"].mean()
            cells.append(f"{100*cov.mean():>8.1f}%     ")
        print(f"  {variant:<9} " + "".join(cells))


if __name__ == "__main__":
    df = pd.read_csv(INTERIM / "features.csv")
    df = df[df["VB"].notna()].copy()
    print(f"labelled runs {len(df)} across {df['case'].nunique()} inserts")
    print(f"DAQ-flagged runs kept and flagged: {int(df['flag_corrupt'].sum())}")
    res = run(df)
    res.to_csv(INTERIM / "calibrated.csv", index=False)
    summarise(res)
    print(f"\nwrote {INTERIM / 'calibrated.csv'}  ({len(res):,} rows)")

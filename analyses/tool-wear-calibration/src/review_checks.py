"""Adversarial review of our own results. Every check here is one a hostile reviewer would run.

CHECK 1  Is the Gaussian baseline a strawman? It currently takes sigma from IN-SAMPLE random
         forest residuals, which are optimistically small because trees memorise. Conformal gets
         a held-out calibration set. Refit the Gaussian on the SAME held-out set and see how much
         of the 68.5% survives.
CHECK 2  Are the Wilson intervals wrong? 146 cuts come from 16 inserts, so the observations are
         clustered and Wilson assumes they are not. Compare against a cluster bootstrap over
         inserts. This matters because the article lectures the reader about grouping.
CHECK 3  Does the random forest ever predict high wear? If predictions are compressed, threshold
         policies above some value never fire and the decision curve saturates for a reason that
         has nothing to do with the wear limit.
CHECK 4  Is "fixed-N" really a fixed-N policy? It indexes labelled cuts, and inserts with fewer
         than N labelled cuts are never changed at all.
CHECK 5  How often does the Mondrian construction silently fall back to the pooled quantile?
CHECK 6  Is the sensor result an artefact of leaving max_features at its default? With 58
         features and all of them considered at every split, added noise cannot be averaged out.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "artifacts" / "interim"
ALPHA, CAL_CASES, SEEDS = 0.10, 4, 8
META = ["DOC", "feed", "material", "cut_index"]
CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
LIMIT = 0.60
line = "=" * 88


def rf(max_features=1.0, **kw):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     max_features=max_features, n_jobs=-1,
                                     random_state=0, **kw)),
    ])


def cq(scores, alpha=ALPHA):
    n = len(scores)
    return np.inf if n == 0 else float(
        np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


df = pd.read_csv(INTERIM / "features.csv")
df = df[df["VB"].notna()].copy()
res = pd.read_csv(INTERIM / "calibrated.csv")


# ---------------------------------------------------------------- CHECK 1
print(f"{line}\nCHECK 1 - is the Gaussian baseline a strawman?\n{line}")
rows = []
for seed in range(SEEDS):
    rng = np.random.default_rng(1000 + seed)
    for held in np.sort(df["case"].unique()):
        te, tr = df[df["case"] == held], df[df["case"] != held]
        if te.empty or len(tr) < 20:
            continue
        other = np.sort(tr["case"].unique())
        cal_cases = rng.choice(other, size=min(CAL_CASES, len(other) - 1), replace=False)
        fit_g, cal_g = tr[~tr["case"].isin(cal_cases)], tr[tr["case"].isin(cal_cases)]

        m_fit = rf().fit(fit_g[META], fit_g["VB"])
        r_cal = cal_g["VB"].to_numpy() - m_fit.predict(cal_g[META])

        m_all = rf().fit(tr[META], tr["VB"])
        r_in = tr["VB"].to_numpy() - m_all.predict(tr[META])

        pred = m_all.predict(te[META])
        for i, (_, row) in enumerate(te.iterrows()):
            rows.append({
                "seed": seed, "VB": row["VB"], "err": abs(row["VB"] - pred[i]),
                "q_gauss_insample": 1.645 * np.sqrt(np.mean(r_in**2)),
                "q_gauss_heldout": 1.645 * np.std(r_cal, ddof=1),
                "q_conf_heldout": cq(np.abs(r_cal)),
            })
g = pd.DataFrame(rows)
for name in ["q_gauss_insample", "q_gauss_heldout", "q_conf_heldout"]:
    g[f"cov_{name}"] = (g["err"] <= g[name]).astype(int)

print(f"{'construction':<34} {'all cuts':>10} {'worn':>10} {'median width':>14}")
for name, lab in [("q_gauss_insample", "Gaussian, in-sample sigma"),
                  ("q_gauss_heldout", "Gaussian, held-out sigma"),
                  ("q_conf_heldout", "Conformal, held-out (same set)")]:
    allc = 100 * g.groupby("seed")[f"cov_{name}"].mean().mean()
    worn = 100 * g[g["VB"] >= LIMIT].groupby("seed")[f"cov_{name}"].mean().mean()
    print(f"{lab:<34} {allc:>9.1f}% {worn:>9.1f}% {2*g[name].median():>13.3f}")
print("\nIf the held-out Gaussian recovers most of the gap, the published 68.5% is measuring")
print("random-forest overfitting, not the Gaussian assumption, and the claim must be rewritten.")


# ---------------------------------------------------------------- CHECK 2
print(f"\n{line}\nCHECK 2 - Wilson vs cluster bootstrap over inserts\n{line}")
meta = res[res["variant"] == "meta"]
rng = np.random.default_rng(7)
print(f"{'subset':<10} {'constr':<8} {'cov':>7} {'Wilson (row-level)':>22} {'cluster bootstrap':>22}")
for label, sub in [("all", meta), ("worn", meta[meta["VB"] >= LIMIT])]:
    per_case = sub.groupby(["case", "seed"])
    cases = sub["case"].unique()
    for c in ["G", "C-grp", "C-norm"]:
        pooled = sub.groupby("seed")[f"cov_{c}"].mean().mean()
        n = int(sub.groupby("seed").size().mean())
        lo, hi = wilson(int(round(pooled * n)), n)
        boot = []
        by_case = {k: v[f"cov_{c}"].to_numpy() for k, v in sub.groupby("case")}
        for _ in range(2000):
            pick = rng.choice(cases, size=len(cases), replace=True)
            vals = np.concatenate([by_case[p] for p in pick])
            boot.append(vals.mean())
        b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
        print(f"{label:<10} {c:<8} {100*pooled:>6.1f}% "
              f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>22} {f'[{100*b_lo:.1f}, {100*b_hi:.1f}]':>22}")


# ---------------------------------------------------------------- CHECK 3
print(f"\n{line}\nCHECK 3 - does the model ever predict high wear?\n{line}")
print(f"true VB     : min {meta['VB'].min():.2f}  median {meta['VB'].median():.2f}  "
      f"p95 {meta['VB'].quantile(0.95):.2f}  max {meta['VB'].max():.2f}")
print(f"predicted   : min {meta['pred'].min():.2f}  median {meta['pred'].median():.2f}  "
      f"p95 {meta['pred'].quantile(0.95):.2f}  max {meta['pred'].max():.2f}")
for t in [0.5, 0.6, 0.7, 0.75, 0.8]:
    frac = 100 * (meta["pred"] >= t).mean()
    print(f"  predictions reaching {t:.2f} mm: {frac:>5.2f}%  "
          f"(true VB reaching it: {100*(meta['VB']>=t).mean():>5.2f}%)")


# ---------------------------------------------------------------- CHECK 4
print(f"\n{line}\nCHECK 4 - is 'fixed-N' a real policy?\n{line}")
counts = df.groupby("case").size()
print(f"labelled cuts per insert: min {counts.min()}, median {counts.median():.0f}, max {counts.max()}")
for N in [8, 12, 18, 20]:
    never = int((counts <= N - 1).sum())
    print(f"  N={N:>2}: {never} of {len(counts)} inserts are NEVER changed "
          f"(they have fewer than {N} labelled cuts)")
print("\nAlso note the unit: the simulation counts LABELLED cuts, not physical cuts")
print(f"(146 labelled out of 167 recorded), so 'per 100 cuts' is really 'per 100 measured cuts'.")


# ---------------------------------------------------------------- CHECK 5
print(f"\n{line}\nCHECK 5 - how often does Mondrian fall back to the pooled quantile?\n{line}")
same = np.isclose(meta["q_C-mond"], meta["q_C-grp"])
print(f"rows where the Mondrian quantile equals the pooled one: {100*same.mean():.1f}%")
hi_pred = meta["pred"] >= LIMIT
print(f"rows whose PREDICTED wear is at or above {LIMIT}: {100*hi_pred.mean():.1f}%")
print("A bucket needs 5 calibration points to get its own quantile; if predictions rarely")
print("reach the limit (check 3), the upper bucket is usually empty and Mondrian is not")
print("actually being tested.")


# ---------------------------------------------------------------- CHECK 6
print(f"\n{line}\nCHECK 6 - is the sensor result an artefact of max_features?\n{line}")
sensor = [c for c in df.columns if any(c.startswith(f"{ch}_") for ch in CHANNELS)]
sets = {"meta": META, "+all": META + sensor}
print(f"{'features':<8} {'max_features':<14} {'RMSE':>8}")
for name, feats in sets.items():
    for mf in [1.0, "sqrt", 0.33]:
        errs = []
        for seed in range(3):
            for held in np.sort(df["case"].unique()):
                te, tr = df[df["case"] == held], df[df["case"] != held]
                if te.empty or len(tr) < 20:
                    continue
                m = rf(max_features=mf).fit(tr[feats], tr["VB"])
                errs.append((te["VB"].to_numpy() - m.predict(te[feats])) ** 2)
        print(f"{name:<8} {str(mf):<14} {np.sqrt(np.concatenate(errs).mean()):>8.4f}")

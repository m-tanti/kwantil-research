"""Seeded simulation for the Clustered Data background article's figure 2.

Demonstrates the split-design failure on synthetic data where the truth is known, so
the article can show the mechanism rather than assert it. The setup mirrors the shape
of real grouped data without borrowing any particular dataset's numbers:

    y_ij = 0.05 * t_ij + a_i + e_ij

  a_i ~ N(0, tau)   a per-group offset: what makes one insert, batch or operator
                    systematically different from another
  e_ij ~ N(0, sigma) within-group noise
  t_ij               a within-group index (cut number, part number, sequence position)
  c_i                a group-level covariate, unique per group

c_i is the mechanism. It is an ordinary continuous feature, not an identifier anybody
would flag, but because it is constant within a group and unique across groups, a
flexible model can use it to recover a_i whenever other rows from the same group are in
the training set. That is what leakage looks like in practice: not a column called
"group_id", just a covariate that happens to name the group.

The intraclass correlation rho = tau^2 / (tau^2 + sigma^2) is swept from 0 to 0.8.
At rho = 0 the groups carry no signal and both split designs agree; as rho rises the
row-level split drifts away from its promise while the group-level split holds.

Writes background_clustered.json next to the other figure exports.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "figures"

ALPHA = 0.10          # nominal 90% intervals
N_GROUPS = 24
M_PER_GROUP = 10
N_TEST_GROUPS = 6
N_CAL_GROUPS = 6
N_REPS = 200
SIGMA = 1.0
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, lvl, method="higher"))


def simulate(rho: float, rng: np.random.Generator) -> dict:
    tau = np.sqrt(rho / (1 - rho)) * SIGMA if rho > 0 else 0.0

    g = np.repeat(np.arange(N_GROUPS), M_PER_GROUP)
    t = np.tile(np.arange(1, M_PER_GROUP + 1), N_GROUPS).astype(float)
    c = np.repeat(rng.uniform(0, 1, N_GROUPS), M_PER_GROUP)
    a = np.repeat(rng.normal(0, tau, N_GROUPS), M_PER_GROUP)
    y = 0.05 * t + a + rng.normal(0, SIGMA, len(t))
    X = np.column_stack([t, c])

    groups = rng.permutation(N_GROUPS)
    te_g, cal_g, fit_g = groups[:N_TEST_GROUPS], \
        groups[N_TEST_GROUPS:N_TEST_GROUPS + N_CAL_GROUPS], \
        groups[N_TEST_GROUPS + N_CAL_GROUPS:]
    te = np.isin(g, te_g)
    tr = ~te

    # ---- split by group: no row from a calibration group is ever fitted on ----
    fit = np.isin(g, fit_g)
    cal = np.isin(g, cal_g)
    m = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                              random_state=0, n_jobs=-1).fit(X[fit], y[fit])
    q_grp = conformal_q(np.abs(y[cal] - m.predict(X[cal])))
    pred_grp = m.predict(X[te])

    # ---- split by row: same calibration size, rows drawn at random from training ----
    tr_idx = np.where(tr)[0]
    perm = rng.permutation(len(tr_idx))
    n_cal = int(cal.sum())
    cal_r, fit_r = tr_idx[perm[:n_cal]], tr_idx[perm[n_cal:]]
    m_r = RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                random_state=0, n_jobs=-1).fit(X[fit_r], y[fit_r])
    q_row = conformal_q(np.abs(y[cal_r] - m_r.predict(X[cal_r])))

    # both are scored on the SAME unseen groups, using the same predictions, so the
    # only thing that differs between them is where the calibration rows came from
    err = np.abs(y[te] - pred_grp)
    return {
        "cov_grp": float(np.mean(err <= q_grp)), "wid_grp": 2 * q_grp,
        "cov_row": float(np.mean(err <= q_row)), "wid_row": 2 * q_row,
    }


def main() -> None:
    rows = []
    for rho in RHOS:
        rng = np.random.default_rng(20260823)
        reps = [simulate(rho, rng) for _ in range(N_REPS)]
        row = {"rho": rho}
        for k in ("cov_grp", "cov_row", "wid_grp", "wid_row"):
            vals = np.array([r[k] for r in reps])
            row[k] = round(float(vals.mean()) * (100 if k.startswith("cov") else 1), 2)
            if k.startswith("cov"):
                row[k + "_lo"] = round(float(np.percentile(vals, 5)) * 100, 1)
                row[k + "_hi"] = round(float(np.percentile(vals, 95)) * 100, 1)
        rows.append(row)
        print(f"  rho={rho:.1f}  group-split {row['cov_grp']:5.1f}%  "
              f"row-split {row['cov_row']:5.1f}%   widths "
              f"{row['wid_grp']:.2f} / {row['wid_row']:.2f}")

    payload = {
        "nominal": 90,
        "groups": N_GROUPS,
        "perGroup": M_PER_GROUP,
        "reps": N_REPS,
        "rows": rows,
        "note": ("seeded simulation; both constructions score the same unseen groups with the "
                 "same predictions, so the only difference between them is where the "
                 "calibration rows came from"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "background_clustered.json"
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

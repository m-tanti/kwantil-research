"""E8 (recalibration learning curve, conformal) and E10 (decision cost), from stored E1 responses.

Held-out test slice: 500 Banking77 items, seed 20260923. Calibration draws come from the
remaining Banking77 items plus all CLINC150 items (the 'pooled Choice slice' of claims.yaml),
with a single scalar temperature fitted on log p-hat by NLL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (SEED, apply_temperature, aps_sets, boot, decision_cost, ece_top,  # noqa: E402
                      fit_isotonic_binary, fit_platt_binary, fit_temperature, load_choice_full)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "interim"
FLAGGED = ["card_swallowed", "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone",
           "cash_withdrawal_not_recognised", "card_payment_not_recognised",
           "direct_debit_payment_not_recognised", "cancel_transfer", "request_refund",
           "transaction_charged_twice"]


def main(pool_clinc: bool = True) -> dict:
    Pb, yb, lab_b, ids_b = load_choice_full("banking77")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(yb))
    test, rest = perm[:500], perm[500:]
    P_test, y_test = Pb[test], yb[test]
    pool_P = [Pb[rest]]; pool_y = [yb[rest]]
    if pool_clinc:
        Pc, yc, _, _ = load_choice_full("clinc150")
        pool_P.append(Pc); pool_y.append(yc)
    # logits per row; pooled rows have different K, so keep as a list of (logit_row, y)
    eps = 1e-6
    pool = [(np.log(np.clip(P, eps, 1)), y) for P, y in zip(pool_P, pool_y)]
    conf_raw = P_test.max(1); corr_raw = (P_test.argmax(1) == y_test).astype(float)
    ece_raw = ece_top(conf_raw, corr_raw)
    res = {"n_test": 500, "ece_raw": ece_raw, "acc": float(corr_raw.mean()), "curve": {}}

    def fit_T(idx_list):
        # idx_list: list of (block, indices). Concatenate NLL over blocks with a shared T.
        from scipy.optimize import minimize_scalar
        import math

        def nll(logT):
            T = math.exp(logT); tot = 0.0; n = 0
            for b, idx in idx_list:
                z = pool[b][0][idx] / T
                z = z - z.max(1, keepdims=True)
                lp = z - np.log(np.exp(z).sum(1, keepdims=True))
                tot += -lp[np.arange(len(idx)), pool[b][1][idx]].sum(); n += len(idx)
            return tot / n
        return math.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)

    sizes = [50, 100, 250, 500]
    total = sum(len(y) for _, y in pool)
    all_idx = [(b, i) for b, (_, y) in enumerate(pool) for i in range(len(y))]
    for n in sizes:
        ratios, Ts, eces = [], [], []
        for d in range(25):
            r2 = np.random.default_rng(SEED + 1000 * n + d)
            pick = r2.choice(total, n, replace=False)
            groups = {}
            for k in pick:
                b, i = all_idx[k]; groups.setdefault(b, []).append(i)
            T = fit_T([(b, np.array(v)) for b, v in groups.items()])
            Pt = apply_temperature(P_test, T)
            e = ece_top(Pt.max(1), (Pt.argmax(1) == y_test).astype(float))
            ratios.append(e / ece_raw); Ts.append(T); eces.append(e)
        res["curve"][n] = {"median_ratio": float(np.median(ratios)), "q25": float(np.percentile(ratios, 25)),
                           "q75": float(np.percentile(ratios, 75)), "median_T": float(np.median(Ts)),
                           "median_ece": float(np.median(eces))}
        print(f"n={n}: median ECE ratio {np.median(ratios):.3f}  median T {np.median(Ts):.2f}  median ECE after {np.median(eces):.3f}")
    # the pinned JC-05 model: median temperature at n=100
    T100 = res["curve"][100]["median_T"]
    P_scaled = apply_temperature(P_test, T100)
    res["T100"] = T100
    # reliability before/after on the test slice
    from analysis import reliability
    res["rel_raw"] = reliability(conf_raw, corr_raw)
    res["rel_scaled"] = reliability(P_scaled.max(1), (P_scaled.argmax(1) == y_test).astype(float))

    # conformal (APS) at alpha 0.1 on Banking77 only: calibration = rest, test = test
    cov_rows = []
    for n in sizes:
        covs, sizes_ = [], []
        for d in range(25):
            r2 = np.random.default_rng(SEED + 77 * n + d)
            cal = r2.choice(rest, n, replace=False)
            S = aps_sets(Pb[cal], yb[cal], P_test, alpha=0.1, seed=SEED + d)
            covs.append(S[np.arange(500), y_test].mean()); sizes_.append(S.sum(1).mean())
        cov_rows.append({"n": n, "coverage": float(np.median(covs)), "cov_q25": float(np.percentile(covs, 25)),
                         "cov_q75": float(np.percentile(covs, 75)), "set_size": float(np.median(sizes_))})
        print(f"conformal n={n}: coverage {np.median(covs):.3f} [{np.percentile(covs,25):.3f},{np.percentile(covs,75):.3f}] mean set size {np.median(sizes_):.2f}")
    res["conformal"] = cov_rows
    # naive threshold set for comparison: include labels until cumulative p >= 0.9
    order = np.argsort(-P_test, 1); srt = np.take_along_axis(P_test, order, 1); cum = np.cumsum(srt, 1)
    k = (cum < 0.9).sum(1) + 1
    inc = np.arange(P_test.shape[1])[None, :] < k[:, None]
    S = np.zeros_like(P_test, dtype=bool); np.put_along_axis(S, order, inc, 1)
    res["naive90"] = {"coverage": float(S[np.arange(500), y_test].mean()), "set_size": float(S.sum(1).mean())}
    print("naive 'sets to 0.9 mass': coverage", res["naive90"]["coverage"], "size", res["naive90"]["set_size"])

    # E10: flagged-set binary task on the test slice
    fl = np.array([lab_b.index(x) for x in FLAGGED])
    p_flag_raw = P_test[:, fl].sum(1); p_flag_sc = P_scaled[:, fl].sum(1)
    y_flag = np.isin(y_test, fl).astype(float)
    res["e10"] = {"n_flag_pos": int(y_flag.sum())}
    for name, (cfp, cfn) in {"fn10": (1, 10), "fn3": (1, 3), "fn1": (1, 1)}.items():
        c_raw = decision_cost(p_flag_raw, y_flag, cfp, cfn); c_sc = decision_cost(p_flag_sc, y_flag, cfp, cfn)

        def ratio(pr, ps, yy):
            a = decision_cost(pr, yy, cfp, cfn); b = decision_cost(ps, yy, cfp, cfn)
            return a / b if b > 0 else np.nan
        est, lo, hi = boot(ratio, p_flag_raw, p_flag_sc, y_flag, draws=2000)
        # oracle: best threshold in hindsight on raw p
        thr_grid = np.linspace(0.01, 0.99, 99)
        costs = [1000 * (cfp * np.sum((p_flag_raw >= t) & (y_flag == 0)) + cfn * np.sum((p_flag_raw < t) & (y_flag == 1))) / 500 for t in thr_grid]
        res["e10"][name] = {"cost_raw": c_raw, "cost_scaled": c_sc, "ratio": est, "lo": lo, "hi": hi,
                            "bayes_thr": cfp / (cfp + cfn), "oracle_cost": float(min(costs)),
                            "oracle_thr": float(thr_grid[int(np.argmin(costs))])}
        print(f"E10 {name}: cost raw {c_raw:.1f} scaled {c_sc:.1f} ratio {est:.3f} [{lo:.3f},{hi:.3f}]  oracle {min(costs):.1f} at thr {thr_grid[int(np.argmin(costs))]:.2f}")
    # cost curve over thresholds for the figure (raw and scaled), fn10
    grid = np.linspace(0.02, 0.98, 49)
    def curve(p):
        return [1000 * (1 * np.sum((p >= t) & (y_flag == 0)) + 10 * np.sum((p < t) & (y_flag == 1))) / 500 for t in grid]
    res["e10"]["curve"] = {"thr": grid.tolist(), "raw": curve(p_flag_raw), "scaled": curve(p_flag_sc)}
    # reliability of the flag probability itself
    res["e10"]["items"] = [{"r": float(np.round(a, 4)), "s": float(np.round(b, 4)), "y": int(c)}
                          for a, b, c in zip(p_flag_raw, p_flag_sc, y_flag)]
    res["e10"]["rel_raw"] = reliability(p_flag_raw, y_flag, bins=8)
    res["e10"]["rel_scaled"] = reliability(p_flag_sc, y_flag, bins=8)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "e8_e10.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    main(pool_clinc=(sys.argv[1:] != ["nopool"]))

"""Write the three figure JSONs and the numbers table the article quotes.

Outputs to artifacts/figures/ and, if the BLOG checkout is present, copies to the article's
figures/data folder. Every number in the article comes from summary.json.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis import (SEED, auroc, boot, brier_binary, ece_top, load_choice, load_choice_full,  # noqa: E402
                      load_noul, load_score, logscore_binary, reliability, rps)
from jev_client import load_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
BLOG = Path("C:/work/BLOG/content/articles/JevCalibration/figures/data")
ITEMS = ROOT / "artifacts" / "items"


def r3(x):
    return float(np.round(x, 3))


def fig01() -> dict:
    """Reliability by question type, plus the probe scatter."""
    out = {"nominal_bins": 10, "interval": "95% percentile bootstrap over items, 2,000 draws", "sets": {}}
    for name in ("banking77", "clinc150"):
        d = load_choice(name)
        e, lo, hi = boot(ece_top, d["conf"], d["correct"])
        out["sets"][name] = {
            "type": "choice", "K": d["K"], "n": d["n"], "acc": r3(d["correct"].mean()),
            "mean_conf": r3(d["conf"].mean()), "ece": r3(e), "lo": r3(lo), "hi": r3(hi),
            "share_at_1": r3((d["conf"] >= 0.995).mean()), "acc_at_1": r3(d["correct"][d["conf"] >= 0.995].mean()),
            "bins": [{k: r3(v) if isinstance(v, float) else v for k, v in b.items()} for b in reliability(d["conf"], d["correct"])],
            "contaminated": True}
    b = load_noul("boolq")
    p, y = b["p"], b["y"]
    e, lo, hi = boot(ece_top, p, y)
    out["sets"]["boolq"] = {
        "type": "noul", "n": b["n"], "acc": r3(((p >= 0.5) == y).mean()), "base_rate": r3(y.mean()),
        "mean_p": r3(p.mean()), "ece": r3(e), "lo": r3(lo), "hi": r3(hi),
        "brier": r3(brier_binary(p, y)), "log": r3(logscore_binary(p, y)),
        "bins": [{k: r3(v) if isinstance(v, float) else v for k, v in bb.items()} for bb in reliability(p, y)],
        "contaminated": True, "note": "bins are p(yes) against observed frequency of yes"}
    a = load_score("amazon")
    P, yy = a["P"], a["y"]
    top = P.argmax(1)
    e, lo, hi = boot(ece_top, P.max(1), (top == yy).astype(float))
    rr, rlo, rhi = boot(rps, P, yy)
    out["sets"]["amazon"] = {
        "type": "score", "K": 5, "n": a["n"], "acc": r3((top == yy).mean()), "within1": r3((np.abs(top - yy) <= 1).mean()),
        "mean_conf": r3(P.max(1).mean()), "ece": r3(e), "lo": r3(lo), "hi": r3(hi),
        "rps": r3(rr), "rps_lo": r3(rlo), "rps_hi": r3(rhi), "rps_uniform": r3(rps(np.full_like(P, 0.2), yy)),
        "rps_onehot": r3(rps(np.eye(5)[top], yy)),
        "bins": [{k: r3(v) if isinstance(v, float) else v for k, v in bb.items()} for bb in reliability(P.max(1), (top == yy).astype(float))],
        "levels": [{"level": k + 1, "mean_p": r3(P[:, k].mean()), "freq": r3((yy == k).mean()),
                    "ece_k": r3(ece_top(P[:, k], (yy == k).astype(float)))} for k in range(5)],
        "contaminated": True}
    # probes
    items = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS / "probes.jsonl")}
    pts = []
    for r in load_log("e4_probes"):
        it = items[r["meta"]["id"]]
        pts.append({"stated": it["p"], "p_hat": r["response"]["answers"]["q"]["noul"], "mech": it["mech"]})
    dev = np.array([abs(x["p_hat"] - x["stated"]) for x in pts])
    m, lo, hi = boot(lambda z: z.mean(), dev)
    out["probes"] = {"n": len(pts), "mad": r3(m), "lo": r3(lo), "hi": r3(hi), "points": pts,
                     "by_stated": [{"stated": s, "median": r3(np.median([x["p_hat"] for x in pts if abs(x["stated"] - s) < 1e-9])),
                                    "n": int(sum(1 for x in pts if abs(x["stated"] - s) < 1e-9))}
                                   for s in sorted(set(round(x["stated"], 2) for x in pts))]}
    # synthetic block: Noul factorial and the known-distribution Choice and Score items
    syn = {json.loads(l)["id"]: json.loads(l) for l in open(ITEMS / "synthetic.jsonl")}
    nf, ck, sk = [], [], []
    for r in load_log("e4b_synthetic"):
        it = syn[r["meta"]["id"]]; a = r["response"]["answers"]["q"]
        if it["block"] == "noul_factorial":
            nf.append({"stated": it["p"], "p_hat": a["noul"], "form": it["form"], "dist": it["dist"]})
        elif it["block"] == "choice_known":
            q = np.array([a["probabilities"].get(c, 0.0) for c in it["labels"]]); t = np.array(it["truth"])
            ck.append({"K": it["K"], "dist": it["dist"], "stated_max": r3(t.max()), "p_hat_max": r3(q.max()),
                       "tv": r3(np.abs(q - t).sum() / 2)})
        else:
            q = np.array([a["probabilities"][str(i)] for i in range(5)]); t = np.array(it["truth"])
            sk.append({"stated_max": r3(t.max()), "p_hat_max": r3(q.max()), "tv": r3(np.abs(q - t).sum() / 2)})
    cells = []
    for form in ("digits", "words", "percent"):
        for dist in ("none", "short", "long"):
            sub = [abs(x["p_hat"] - x["stated"]) for x in nf if x["form"] == form and x["dist"] == dist]
            cells.append({"form": form, "dist": dist, "mad": r3(np.mean(sub)), "n": len(sub)})
    out["synthetic"] = {"noul_points": nf, "noul_mad": r3(np.mean([abs(x["p_hat"] - x["stated"]) for x in nf])), "noul_cells": cells,
                        "choice_points": ck, "choice_by_K": [{"K": K, "mean_excess": r3(np.mean([x["p_hat_max"] - x["stated_max"] for x in ck if x["K"] == K])),
                                                              "mean_tv": r3(np.mean([x["tv"] for x in ck if x["K"] == K])), "n": sum(1 for x in ck if x["K"] == K)} for K in (2, 3, 5, 10)],
                        "score_points": sk, "score_mean_tv": r3(np.mean([x["tv"] for x in sk])),
                        "score_mean_excess": r3(np.mean([x["p_hat_max"] - x["stated_max"] for x in sk]))}
    # sealed cohort: only the descriptive state, no outcomes yet
    sealed = [json.loads(l) for l in open(ROOT / "artifacts" / "sealed" / "cohort-1.jsonl", encoding="utf-8")]
    ps = np.array([s["p_yes"] for s in sealed])
    cp = np.array([s["community_p"] if s["community_p"] is not None else np.nan for s in sealed])
    ok = ~np.isnan(cp)
    out["sealed"] = {"n": len(sealed), "sha256": (ROOT / "artifacts" / "sealed" / "cohort-1.sha256").read_text().split()[0],
                     "mean_p": r3(ps.mean()), "share_above_0_9": r3((ps > 0.9).mean()), "share_below_0_1": r3((ps < 0.1).mean()),
                     "corr_with_market": r3(np.corrcoef(ps[ok], cp[ok])[0, 1]), "mean_abs_gap_to_market": r3(np.abs(ps[ok] - cp[ok]).mean()),
                     "market_mean": r3(cp[ok].mean()), "hist_jev": np.histogram(ps, bins=10, range=(0, 1))[0].tolist(),
                     "hist_market": np.histogram(cp[ok], bins=10, range=(0, 1))[0].tolist(),
                     "status": "pending resolution; scored after 2026-12-31"}
    return out


def fig02() -> dict:
    """Invariance: distribution of |delta p(top)| per perturbation, flips, and cardinality effect."""
    recs = load_log("e5_invariance")
    truth = {json.loads(l)["id"]: json.loads(l)["truth"] for l in open(ITEMS / "e5_sample.jsonl")}
    base, per = {}, {}
    for r in recs:
        m = r["meta"]; a = list(r["response"]["answers"].values())[0]
        if m["pert"] == "repeat" and m["rep"] == 0:
            base[m["id"]] = a
        per.setdefault(m["pert"], {}).setdefault(m["id"], []).append((a, m))

    def p_of(a, label, m):
        pr = a["probabilities"]
        if m.get("pert") == "letters":
            inv = {v: k for k, v in m["map"].items()}
            return pr.get(inv[label], 0.0)
        return pr.get(label, 0.0)

    def choice_of(a, m):
        return m["map"][a["choice"]] if m.get("pert") == "letters" else a["choice"]

    LABELS = {"repeat": "identical call, repeated", "order": "options reordered", "letters": "option keys as letters",
              "surface": "lower-case, no punctuation", "distractor": "irrelevant sentence added",
              "card20": "20 options instead of 77", "card5": "5 options instead of 77"}
    rows = []
    for pert in ("repeat", "order", "letters", "surface", "distractor", "card20", "card5"):
        d, flips, pt, acc = [], 0, [], [0, 0]
        for iid, lst in per[pert].items():
            b = base[iid]; top = b["choice"]
            for a, m in lst:
                if pert == "repeat" and m["rep"] == 0:
                    continue
                d.append(abs(p_of(a, top, m) - b["probabilities"][top]))
                flips += choice_of(a, m) != top
                pt.append((b["probabilities"].get(truth[iid], 0.0), p_of(a, truth[iid], m)))
                acc[0] += top == truth[iid]; acc[1] += choice_of(a, m) == truth[iid]
        d = np.array(d); pt = np.array(pt)
        qs = np.percentile(d, [10, 25, 50, 75, 90, 95, 99])
        flip_rate, flo, fhi = boot(lambda z: z.mean(), np.array([1.0] * flips + [0.0] * (len(d) - flips)))
        rows.append({"pert": pert, "label": LABELS[pert], "n": int(len(d)), "median": r3(qs[2]), "mean": r3(d.mean()),
                     "q10": r3(qs[0]), "q25": r3(qs[1]), "q75": r3(qs[3]), "q90": r3(qs[4]), "q95": r3(qs[5]), "q99": r3(qs[6]),
                     "flip_rate": r3(flip_rate), "flip_lo": r3(flo), "flip_hi": r3(fhi),
                     "p_true_base": r3(pt[:, 0].mean()), "p_true_pert": r3(pt[:, 1].mean()),
                     "acc_base": r3(acc[0] / len(d)), "acc_pert": r3(acc[1] / len(d)),
                     "hist": np.histogram(np.clip(d, 0, 1), bins=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0001])[0].tolist()})
    med_letters = [abs(p_of(a, base[i]["choice"], m) - base[i]["probabilities"][base[i]["choice"]]) for i, l in per["letters"].items() for a, m in l]
    est, lo, hi = boot(np.median, np.array(med_letters))
    return {"rows": rows, "hist_edges": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "jc03": {"median": r3(est), "lo": r3(lo), "hi": r3(hi)},
            "n_items": len(base), "source": "300 Banking77 test items, seed 20260923"}


def fig03() -> dict:
    return json.load(open(ROOT / "artifacts" / "interim" / "e8_e10.json"))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    f1, f2, f3 = fig01(), fig02(), fig03()
    for name, obj in (("reliability", f1), ("invariance", f2), ("decision", f3)):
        with open(FIG / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=0, default=float)
    # summary table for the article
    d = load_choice("banking77")
    c = d["confidence"]; K = d["K"]
    diff_est, dlo, dhi = boot(lambda cc, mm, yy: auroc(cc, yy) - auroc(mm, yy), c, d["conf"], d["correct"], draws=500)
    summary = {
        "sets": {k: {kk: vv for kk, vv in v.items() if kk not in ("bins", "levels")} for k, v in f1["sets"].items()},
        "probes": {k: v for k, v in f1["probes"].items() if k != "points"},
        "sealed": {k: v for k, v in f1["sealed"].items() if not k.startswith("hist")},
        "confidence_field": {"banking77_r2_on_max": None, "formula_mean_abs_dev": r3(np.abs(c - (K * d["conf"] - 1) / (K - 1)).mean()),
                             "auroc_conf": r3(auroc(c, d["correct"])), "auroc_max": r3(auroc(d["conf"], d["correct"])),
                             "auroc_diff": r3(diff_est), "auroc_diff_lo": r3(dlo), "auroc_diff_hi": r3(dhi)},
        "invariance": {r["pert"]: {k: r[k] for k in ("median", "mean", "q90", "flip_rate", "flip_lo", "flip_hi", "p_true_base", "p_true_pert", "acc_base", "acc_pert")} for r in f2["rows"]},
        "jc03": f2["jc03"],
        "recal": {"curve": f3["curve"], "T100": f3["T100"], "ece_raw_test": f3["ece_raw"], "conformal": f3["conformal"], "naive90": f3["naive90"]},
        "e10": {k: v for k, v in f3["e10"].items() if k not in ("curve", "rel_raw", "rel_scaled")},
    }
    X = np.c_[np.ones(d["n"]), d["conf"]]
    bb = np.linalg.lstsq(X, c, rcond=None)[0]; pr = X @ bb
    summary["confidence_field"]["banking77_r2_on_max"] = r3(1 - ((c - pr) ** 2).sum() / ((c - c.mean()) ** 2).sum())
    with open(FIG / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1, default=float)
    if BLOG.parent.exists():
        BLOG.mkdir(parents=True, exist_ok=True)
        for name in ("reliability", "invariance", "decision"):
            shutil.copy(FIG / f"{name}.json", BLOG / f"{name}.json")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()

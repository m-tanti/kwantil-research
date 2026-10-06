"""Calibration statistics for the audit. Everything here is computed from the call logs.

Conventions, pinned in claims.yaml:
  - ECE_top: top-label ECE with 10 equal-mass bins on max p-hat
  - intervals: 95% percentile bootstrap over items, 2,000 draws, seed 20260923
  - RPS: ranked probability score, divided by K-1 so it lies in [0, 1]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import load_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "artifacts" / "items"
SEED = 20260923
B = 2000


def items(name: str) -> dict[str, dict]:
    return {json.loads(line)["id"]: json.loads(line) for line in open(ITEMS / f"{name}.jsonl", encoding="utf-8")}


# ---------- loading ----------

def load_choice(name: str, log: str | None = None) -> dict:
    """Returns dict with arrays: conf (max p), correct, p_true, entropy, margin, confidence field, K."""
    it = items(name)
    labels = json.load(open(ITEMS / f"{name}.labels.json"))["labels"]
    K = len(labels)
    conf, correct, ptrue, ent, marg, cfield, ids, lens = [], [], [], [], [], [], [], []
    for r in load_log(log or f"e1_{name}"):
        a = r["response"]["answers"]["q"]
        probs = a["probabilities"]
        p = np.array([probs.get(l, 0.0) for l in labels], dtype=float)
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        t = it[r["meta"]["id"]]["truth"]
        srt = np.sort(p)[::-1]
        conf.append(srt[0]); marg.append(srt[0] - srt[1])
        ent.append(float(-(p[p > 0] * np.log(p[p > 0])).sum()))
        correct.append(a["choice"] == t)
        ptrue.append(float(probs.get(t, 0.0)))
        cfield.append(a.get("confidence", np.nan))
        ids.append(r["meta"]["id"]); lens.append(len(r["request"]["state"]))
    return {"conf": np.array(conf), "correct": np.array(correct, dtype=float), "p_true": np.array(ptrue),
            "entropy": np.array(ent), "margin": np.array(marg), "confidence": np.array(cfield, dtype=float),
            "ids": ids, "len": np.array(lens), "K": K, "n": len(conf), "name": name}


def load_choice_full(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Full probability matrix (n x K), integer truth index, labels. For recalibration."""
    it = items(name)
    labels = json.load(open(ITEMS / f"{name}.labels.json"))["labels"]
    idx = {l: i for i, l in enumerate(labels)}
    P, y, ids = [], [], []
    for r in load_log(f"e1_{name}"):
        probs = r["response"]["answers"]["q"]["probabilities"]
        p = np.array([probs.get(l, 0.0) for l in labels], dtype=float)
        if p.sum() <= 0:
            continue
        P.append(p / p.sum()); y.append(idx[it[r["meta"]["id"]]["truth"]]); ids.append(r["meta"]["id"])
    return np.array(P), np.array(y), labels, ids


def load_noul(name: str, log: str | None = None) -> dict:
    it = items(name)
    p, y, ids = [], [], []
    for r in load_log(log or f"e1_{name}"):
        p.append(r["response"]["answers"]["q"]["noul"]); y.append(float(it[r["meta"]["id"]]["truth"]))
        ids.append(r["meta"]["id"])
    return {"p": np.array(p), "y": np.array(y), "ids": ids, "n": len(p), "name": name}


def load_score(name: str = "amazon") -> dict:
    it = items(name)
    P, y, score, cfield = [], [], [], []
    for r in load_log(f"e1_{name}"):
        a = r["response"]["answers"]["q"]
        p = np.array([a["probabilities"][str(i)] for i in range(5)], dtype=float)
        P.append(p / p.sum()); y.append(int(it[r["meta"]["id"]]["truth"]) - 1)
        score.append(a["score"]); cfield.append(a["confidence"])
    return {"P": np.array(P), "y": np.array(y), "score": np.array(score), "confidence": np.array(cfield),
            "n": len(y), "name": name}


# ---------- metrics ----------

def ece_top(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE, equal-mass bins on conf."""
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    c, a = conf[order], correct[order]
    edges = np.linspace(0, n, bins + 1).astype(int)
    e = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            e += (hi - lo) / n * abs(c[lo:hi].mean() - a[lo:hi].mean())
    return float(e)


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    c, a = conf[order], correct[order]
    edges = np.linspace(0, n, bins + 1).astype(int)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            out.append({"conf": float(c[lo:hi].mean()), "acc": float(a[lo:hi].mean()), "n": int(hi - lo),
                        "lo": float(c[lo]), "hi": float(c[hi - 1])})
    return out


def brier_binary(p, y):
    return float(np.mean((p - y) ** 2))


def logscore_binary(p, y, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def rps(P: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score over ordered levels, normalised by K-1."""
    K = P.shape[1]
    cp = np.cumsum(P, axis=1)
    cy = (np.arange(K)[None, :] >= y[:, None]).astype(float)
    return float(np.mean(((cp - cy) ** 2).sum(axis=1) / (K - 1)))


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """AUROC for predicting y=1 from score, via rank statistic."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot(fn, *arrays, seed: int = SEED, draws: int = B) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval over items."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = fn(*arrays)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[a[idx] for a in arrays]))
    return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------- recalibration ----------

def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature minimising NLL. logits = log p-hat."""
    from scipy.optimize import minimize_scalar

    def nll(logT):
        T = math.exp(logT)
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return math.exp(res.x)


def apply_temperature(P: np.ndarray, T: float, eps: float = 1e-6) -> np.ndarray:
    z = np.log(np.clip(P, eps, 1)) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_platt_binary(p: np.ndarray, y: np.ndarray, eps: float = 1e-6):
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    m = LogisticRegression(C=1e6).fit(x, y)
    return lambda q: m.predict_proba(np.log(np.clip(q, eps, 1 - eps) / (1 - np.clip(q, eps, 1 - eps))).reshape(-1, 1))[:, 1]


def fit_isotonic_binary(p: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return lambda q: m.predict(q)


# ---------- conformal ----------

def aps_sets(P_cal: np.ndarray, y_cal: np.ndarray, P_test: np.ndarray, alpha: float = 0.1, seed: int = SEED):
    """Split conformal with APS scores (randomised). Returns boolean set matrix for the test rows."""
    rng = np.random.default_rng(seed)

    def scores(P, y, u):
        order = np.argsort(-P, axis=1)
        srt = np.take_along_axis(P, order, axis=1)
        cum = np.cumsum(srt, axis=1)
        rank = np.argmax(order == y[:, None], axis=1)
        s = cum[np.arange(len(y)), rank] - u * srt[np.arange(len(y)), rank]
        return s

    u = rng.random(len(y_cal))
    s = scores(P_cal, y_cal, u)
    n = len(s)
    q = np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    order = np.argsort(-P_test, axis=1)
    srt = np.take_along_axis(P_test, order, axis=1)
    cum = np.cumsum(srt, axis=1)
    u2 = rng.random((len(P_test), 1))
    inc = (cum - u2 * srt) <= q
    # always include the top label
    inc[:, 0] = True
    sets = np.zeros_like(P_test, dtype=bool)
    np.put_along_axis(sets, order, inc, axis=1)
    return sets


# ---------- decision cost ----------

def decision_cost(p_flag: np.ndarray, y_flag: np.ndarray, c_fp: float, c_fn: float) -> float:
    """Cost per 1,000 decisions when acting on p_flag with the Bayes threshold for (c_fp, c_fn)."""
    thr = c_fp / (c_fp + c_fn)
    act = p_flag >= thr
    fp = np.sum(act & (y_flag == 0)); fn = np.sum(~act & (y_flag == 1))
    return float(1000 * (c_fp * fp + c_fn * fn) / len(y_flag))

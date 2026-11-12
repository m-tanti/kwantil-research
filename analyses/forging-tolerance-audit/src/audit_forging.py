"""Specimen Process Calibration Audit on the AFRC radial forging data set.

Four parts, in the order a client receives them:

  structure   the day-2 data-structure check: can these records support the
              analysis at all, and what is wrong with them before any scoring
  limits      each stated tolerance against the measured dimension
  ranges      a 90% range around a prediction of part length, built 3 ways and
              scored on days the model never saw
  decision    flag-or-ship policies in counts per 100 parts, which the page
              converts to money at rates the reader supplies

Severities are derived by rule (see the three severity_* functions) and the
run ends with a lineage block: script hash and a digest over the artifacts.

Everything fitted is fitted on some days and scored on a day held out whole.
The unit that repeats is the production day: parts forged on one day share a
setup, a tool state and a warm-up, so 81 parts are not 81 independent trials.

Usage:
    python src/features.py
    python src/audit_forging.py [--publish-to <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from features import FEATURES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ fixed rules
# Fixed before the ranges and the policies were scored, though not in
# ignorance: the share of parts inside each tolerance was already known.
LEVEL = 0.90
RIDGE = 1.0                            # penalty on standardised features, not tuned
SEED = 20261111
ROW_SPLIT_REPEATS = 25
LIMIT_FLOOR = 0.01                     # share of parts outside a limit that makes it Material
CPK_MIN = 1.33                         # conventional minimum capability index
UNTESTED_SD = 6.0                      # nearest limit further than this many SDs: limit not tested
LENGTH_DIMS = ["Top1", "Top2", "Top3", "Top4"]
TARGET = "top_max"                     # the longest of the 4 measured lengths
THRESHOLD_GRID_MM = np.round(np.arange(233.0, 238.01, 0.25), 2)


@dataclass
class Finding:
    id: str
    kind: str
    title: str
    severity: str
    severity_reason: str
    measurement: str
    evidence: str
    consequence: str
    action: str
    limit: str


def wilson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - level) / 2)
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- severity rules
def severity_limit(share_outside: float, outside_lo: float, cpk: float,
                   nearest_sd: float) -> tuple[str, str]:
    """A stated tolerance against the measured dimension.

    Material     the lower Wilson bound of the share of parts outside the
                 tolerance exceeds LIMIT_FLOOR.
    Significant  no established share outside, but the capability index is
                 below CPK_MIN: the process sits closer to a limit than its own
                 scatter allows for.
    Observation  every part inside and capability at or above CPK_MIN. Where
                 the nearest limit is more than UNTESTED_SD standard deviations
                 away the reason says so: such a limit was never tested.
    """
    if outside_lo > LIMIT_FLOOR:
        return "Material", (
            f"{share_outside:.0%} of parts lie outside the stated tolerance (lower bound "
            f"{outside_lo:.1%}, above the {LIMIT_FLOOR:.0%} floor)")
    if cpk < CPK_MIN:
        return "Significant", (
            f"no share outside the tolerance is established, but the capability index is "
            f"{cpk:.2f}, below the conventional {CPK_MIN}")
    if nearest_sd > UNTESTED_SD:
        return "Observation", (
            f"every part is inside; the nearest limit is {nearest_sd:.0f} standard deviations "
            f"from the mean, so these records never tested it")
    return "Observation", f"every part is inside and the capability index is {cpk:.2f}"


def severity_range(stated: float, lo: float, hi: float, regime_cov: float | None,
                   regime_hi: float | None) -> tuple[str, str]:
    """A stated range against outcomes on held-out days.

    Observation  the stated level lies inside the interval of delivered
                 coverage, overall.
    Significant  it lies outside overall, but not established as failing in
                 the regime where the decision is made (the longest parts).
    Material     it lies outside overall, AND the upper bound of coverage in
                 that regime is also below the stated level.
    """
    if lo <= stated <= hi:
        return "Observation", (
            f"the stated {stated:.0%} lies inside the interval of delivered coverage "
            f"[{lo:.0%}, {hi:.0%}]")
    if regime_hi is not None and regime_hi < stated:
        return "Material", (
            f"the stated {stated:.0%} lies outside the interval of delivered coverage "
            f"[{lo:.0%}, {hi:.0%}], and on the longest parts coverage is {regime_cov:.0%} with "
            f"an upper bound of {regime_hi:.0%}")
    return "Significant", (
        f"the stated {stated:.0%} lies outside the interval of delivered coverage "
        f"[{lo:.0%}, {hi:.0%}]; the shortfall on the longest parts is not established")


def severity_structure(blocks_analysis: bool, affects_a_limit: bool) -> tuple[str, str]:
    """A data-structure finding.

    Material     the records cannot support part of the analysis as supplied.
    Significant  the records support it, but a stated limit or a channel is
                 ambiguous or uninformative and the client has to resolve it.
    Observation  recorded for completeness; nothing depends on it.
    """
    if blocks_analysis:
        return "Material", "part of the analysis cannot be run on the records as supplied"
    if affects_a_limit:
        return "Significant", ("the analysis runs, but a stated limit or a channel is ambiguous "
                               "and only the owner of the specification can resolve it")
    return "Observation", "recorded for completeness; no number in this report depends on it"


# ----------------------------------------------------------------------- model
def ridge_fit(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-9, sd, 1.0)            # a constant channel standardises to zero
    Z = (X - mu) / sd
    ym = y.mean()
    beta = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ym + ((Xn - mu) / sd) @ beta


def conformal_q(abs_res: np.ndarray, level: float) -> float:
    n = len(abs_res)
    k = min(n, math.ceil((n + 1) * level))
    return float(np.sort(abs_res)[k - 1])


def predict_all(d: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Leave one DAY out. For each held-out day: the prediction, and the half
    width of a 90% range built three ways from the training days only."""
    y = d[TARGET].to_numpy(float)
    X = d[features].to_numpy(float) if features else np.zeros((len(d), 1))
    days = d["date"].to_numpy()
    rng = np.random.default_rng(SEED)
    z = norm.ppf(1 - (1 - LEVEL) / 2)
    out = np.full((len(d), 4), np.nan)
    for day in np.unique(days):
        te, tr = days == day, days != day
        f = ridge_fit(X[tr], y[tr])
        out[te, 0] = f(X[te])
        # (1) residuals on the data the model was fitted to, Gaussian width
        res_in = y[tr] - f(X[tr])
        out[te, 1] = z * res_in.std(ddof=1)
        # (2) residuals on whole held-out days inside the training set
        res_day = np.empty(tr.sum())
        tr_days = days[tr]
        for inner in np.unique(tr_days):
            m = tr_days == inner
            g = ridge_fit(X[tr][~m], y[tr][~m])
            res_day[m] = y[tr][m] - g(X[tr][m])
        out[te, 2] = conformal_q(np.abs(res_day), LEVEL)
        # (3) residuals on random held-out ROWS: parts of one day on both sides
        qs = []
        idx = np.arange(tr.sum())
        for _ in range(ROW_SPLIT_REPEATS):
            perm = rng.permutation(idx)
            res_row = np.empty(tr.sum())
            for fold in np.array_split(perm, 5):
                m = np.zeros(tr.sum(), bool)
                m[fold] = True
                g = ridge_fit(X[tr][~m], y[tr][~m])
                res_row[m] = y[tr][m] - g(X[tr][m])
            qs.append(conformal_q(np.abs(res_row), LEVEL))
        out[te, 3] = float(np.mean(qs))
    return pd.DataFrame(out, columns=["pred", "hw_insample", "hw_day", "hw_row"], index=d.index)


def write_lf(path: Path, text: str) -> None:
    path.write_bytes(text.replace(chr(13) + chr(10), chr(10)).encode("utf-8"))


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "specimen"))
    ap.add_argument("--publish-to", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(ROOT / "artifacts" / "interim" / "parts.csv")
    tol = json.loads((ROOT / "artifacts" / "interim" / "tolerances.json").read_text(encoding="utf-8"))
    dims = list(tol)
    n = len(d)
    upper = tol["Top4"]["upper"]
    d[TARGET] = d[LENGTH_DIMS].max(axis=1)
    over = d[TARGET] > upper
    days = sorted(d["date"].unique())

    # ---------------------------------------------------------------- structure
    raw = ROOT / "artifacts" / "raw"
    summary_tol = {}
    try:
        s = pd.read_excel(raw / "ForgedPartDataStructureSummaryv3.xlsx",
                          sheet_name="CMM Data Structure", header=None)
        for _, r in s.iloc[2:].iterrows():
            summary_tol[str(r[0]).strip()] = str(r[4])
    except Exception as exc:                      # the check reports it rather than dying
        summary_tol = {"error": str(exc)}
    top4_doc = summary_tol.get("Top4", "")
    constant = [f for f in FEATURES if d[f].std(ddof=1) < 1e-2 * max(1.0, abs(d[f].mean())) * 1e-2]
    row_counts = d["n_rows"]
    structure = {
        "n_parts": n, "n_traces": int(d["n_rows"].notna().sum()),
        "n_days": len(days), "parts_per_day": {k: int(v) for k, v in d["date"].value_counts().sort_index().items()},
        "missing_values": int(d[dims + FEATURES].isna().sum().sum()),
        "trace_rows_median": int(row_counts.median()), "trace_rows_min": int(row_counts.min()),
        "trace_rows_max": int(row_counts.max()),
        "long_traces": [int(p) for p in d.loc[row_counts > 1.2 * row_counts.median(), "part"]],
        "description_says_days": 1,
        "top4_lower_in_measurement_file": tol["Top4"]["lower"] - tol["Top4"]["nominal"],
        "top4_tolerance_in_summary_file": top4_doc,
        "constant_features": constant,
        "supports_analysis": True,
    }
    write_lf(out / "structure.json", json.dumps(structure, indent=1))

    # ------------------------------------------------------------------- limits
    limits = []
    for name in dims:
        x = d[name].to_numpy(float)
        lo, hi = tol[name]["lower"], tol[name]["upper"]
        n_out = int(((x < lo) | (x > hi)).sum())
        w_lo, w_hi = wilson(n_out, n)
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        cpk = min(hi - mean, mean - lo) / (3 * sd)
        nearest = min(hi - mean, mean - lo) / sd
        sev, why = severity_limit(n_out / n, w_lo, cpk, nearest)
        limits.append({
            "dimension": name, "nominal": tol[name]["nominal"], "lower": lo, "upper": hi,
            "mean": mean, "sd": sd, "min": float(x.min()), "max": float(x.max()),
            "n": n, "n_outside": n_out, "share_outside": n_out / n,
            "outside_lo": w_lo, "outside_hi": w_hi, "cpk": cpk, "nearest_limit_sd": nearest,
            "outside_by_day": {day: int((((d[name] < lo) | (d[name] > hi)) & (d["date"] == day)).sum())
                               for day in days},
            "severity": sev, "severity_reason": why,
        })
    write_lf(out / "limits.json", json.dumps(limits, indent=1))
    n_all_inside = int(np.all([(d[k] >= tol[k]["lower"]) & (d[k] <= tol[k]["upper"]) for k in dims],
                              axis=0).sum())

    # ------------------------------------------------------------------- ranges
    p = predict_all(d, FEATURES)
    p0 = predict_all(d, [])                          # no features: the day-held-out mean
    y = d[TARGET].to_numpy(float)
    long_parts = y >= np.quantile(y, 0.75)
    CONS = [("hw_insample", "From errors on its own training data"),
            ("hw_day", "From errors on held-out days"),
            ("hw_row", "From errors on held-out rows")]
    ranges = []
    for col, label in CONS:
        inside = np.abs(y - p["pred"].to_numpy()) <= p[col].to_numpy()
        lo, hi = wilson(int(inside.sum()), n)
        r_lo, r_hi = wilson(int(inside[long_parts].sum()), int(long_parts.sum()))
        sev, why = severity_range(LEVEL, lo, hi, float(inside[long_parts].mean()), r_hi)
        ranges.append({
            "construction": col, "label": label, "stated": LEVEL,
            "coverage": float(inside.mean()), "lo": lo, "hi": hi,
            "coverage_long": float(inside[long_parts].mean()), "long_lo": r_lo, "long_hi": r_hi,
            "n": n, "n_long": int(long_parts.sum()),
            "median_width_mm": float(2 * np.median(p[col])),
            "coverage_by_day": {day: float(inside[(d["date"] == day).to_numpy()].mean()) for day in days},
            "severity": sev, "severity_reason": why,
        })
    mae = float(np.mean(np.abs(y - p["pred"])))
    mae0 = float(np.mean(np.abs(y - p0["pred"])))
    model = {"target": "longest of the 4 measured lengths, mm", "n_features": len(FEATURES),
             "features": FEATURES, "ridge_penalty": RIDGE,
             "mae_mm": mae, "mae_no_features_mm": mae0, "sd_target_mm": float(y.std(ddof=1)),
             "pred_min": float(p["pred"].min()), "pred_max": float(p["pred"].max()),
             "target_min": float(y.min()), "target_max": float(y.max()),
             "upper_limit_mm": upper, "n_over_limit": int(over.sum())}
    write_lf(out / "ranges.json", json.dumps({"model": model, "constructions": ranges}, indent=1))
    scatter = [{"part": int(a), "date": b, "actual": round(float(c), 3), "pred": round(float(e), 3),
                "hw_day": round(float(f), 3)}
               for a, b, c, e, f in zip(d["part"], d["date"], y, p["pred"], p["hw_day"])]
    write_lf(out / "predictions.json", json.dumps(scatter, indent=1))

    # ----------------------------------------------------------------- decision
    def tally(flag: np.ndarray) -> dict:
        return {"flagged_per_100": float(100 * flag.mean()),
                "flagged_in_limit_per_100": float(100 * (flag & ~over.to_numpy()).mean()),
                "escaped_per_100": float(100 * (~flag & over.to_numpy()).mean()),
                "caught_per_100": float(100 * (flag & over.to_numpy()).mean())}
    pred = p["pred"].to_numpy()
    policies = [
        {"policy": "ship_all", "label": "Ship every part unchecked", **tally(np.zeros(n, bool))},
        {"policy": "check_all", "label": "Check every part", **tally(np.ones(n, bool))},
        {"policy": "range_top", "label": "Check when the top of the range passes the limit",
         **tally(pred + p["hw_day"].to_numpy() > upper)},
        {"policy": "prediction", "label": "Check when the prediction itself passes the limit",
         **tally(pred > upper)},
    ]
    curve = [{"threshold_mm": float(t), **tally(pred > t)} for t in THRESHOLD_GRID_MM]
    decision = {"upper_limit_mm": upper, "n": n, "n_over_limit": int(over.sum()),
                "over_limit_per_100": float(100 * over.mean()), "policies": policies, "curve": curve,
                "illustrative_rates": {"cost_check_eur": 15.0, "cost_escape_eur": 400.0,
                                       "note": "ILLUSTRATIVE defaults for the page's control. Not "
                                               "evidence, not from the data set."}}
    write_lf(out / "decision.json", json.dumps(decision, indent=1))

    # ----------------------------------------------------------------- register
    rank = {"Material": 0, "Significant": 1, "Observation": 2}
    F: list[Finding] = []
    L = {r["dimension"]: r for r in limits}

    id55 = L["ID @55"]
    F.append(Finding("", "limit",
        f"No part meets the stated tolerance on the inner diameter at 55 mm: all {n} are outside",
        id55["severity"], id55["severity_reason"],
        (f"Stated {id55['nominal']:.0f} mm, limits {id55['lower']:.1f} to {id55['upper']:.1f}. Measured "
         f"mean {id55['mean']:.3f} mm, standard deviation {id55['sd']:.3f} mm, range {id55['min']:.3f} to "
         f"{id55['max']:.3f}. {id55['n_outside']} of {n} parts outside. The same diameter at 100 mm "
         f"averages {L['ID @100']['mean']:.3f} mm and is inside on every part."),
        "limits.json",
        ("Read as stated, every part in the record is a reject. The scatter is 8 thousandths of a "
         "millimetre, so this is not a process out of control: it is a process holding a value the "
         "specification does not describe."),
        ("Resolve the specification before anything else. Either the nominal at this position is "
         "not 30 mm, or the tolerance was copied from the 100 mm position. Only the owner of the "
         "drawing can say which."),
        "The audit cannot tell a wrong specification from a wrong process. It can tell that one of "
        "them is wrong, and that the offset is constant across 5 days."))

    over_dims = [r for r in limits if r["dimension"] in LENGTH_DIMS and r["n_outside"] > 0]
    worst = max(over_dims, key=lambda r: r["n_outside"])
    F.append(Finding("", "limit",
        f"Part length runs at the upper limit: {int(over.sum())} of {n} parts exceed it on at least "
        f"one of the 4 measurements",
        min((r["severity"] for r in over_dims), key=lambda s: rank[s]),
        worst["severity_reason"].replace("of parts", f"of parts on {worst['dimension']}"),
        ("Stated 222 mm, plus 15 and minus 5, so the upper limit is 237 mm. " +
         "; ".join(f"{r['dimension']}: mean {r['mean']:.1f} mm, {r['n_outside']} of {n} over "
                   f"[{pct(r['outside_lo'], 1)}, {pct(r['outside_hi'], 1)}], capability index {r['cpk']:.2f}"
                   for r in limits if r["dimension"] in LENGTH_DIMS) +
         f". Intervals are 95% Wilson on {n} parts. By day, parts over the limit on any measurement: " +
         ", ".join(f"{day} {int((over & (d['date'] == day)).sum())} of {int((d['date'] == day).sum())}"
                   for day in days) + "."),
        "limits.json, predictions.json",
        (f"{pct(over.mean(), 1)} of parts are over the stated length limit, and the mean sits "
         f"{(upper - L['Top4']['mean']) / L['Top4']['sd']:.1f} standard deviations below it on the "
         f"worst measurement. The decision table prices that at your own rates."),
        ("Either re-centre the process about 2 mm shorter, or confirm that the upper limit is the "
         "one the customer holds you to. If the length is trimmed later, the limit that matters is "
         "the trim allowance, and this finding should be re-read against it."),
        f"{n} parts over {len(days)} days from a testbed, not a production line. The Wilson "
        "intervals treat parts as independent, which parts forged on one day are not; the per-day "
        "counts are given so the reader can see how uneven they are."))

    untested = [r for r in limits if r["severity"] == "Observation" and r["nearest_limit_sd"] > UNTESTED_SD]
    close = [r for r in limits if r["severity"] == "Significant"]
    if close:
        F.append(Finding("", "limit",
            f"{len(close)} length measurements are inside the limit on every part and too close to it",
            "Significant", close[0]["severity_reason"],
            "; ".join(f"{r['dimension']}: maximum {r['max']:.1f} mm against {r['upper']:.0f}, "
                      f"capability index {r['cpk']:.2f}" for r in close) + ".",
            "limits.json",
            "No reject yet on these measurements. At this capability a run several times longer "
            "would be expected to produce some.",
            "Treat with the finding above: it is the same length, measured at another angle.",
            "A capability index assumes a stable, roughly normal process, and 81 parts cannot "
            "establish either."))
    F.append(Finding("", "limit",
        f"{len(untested)} of {len(dims)} stated tolerances were never tested by these records",
        "Observation",
        f"every part is inside, and the nearest limit is more than {UNTESTED_SD:.0f} standard "
        f"deviations from the mean on each",
        "; ".join(f"{r['dimension']} ({r['nearest_limit_sd']:.0f} SD)" for r in untested) + ".",
        "limits.json",
        "These limits are met, and by so wide a margin that the record says nothing about what "
        "would happen near them. A coverage figure over all 18 dimensions would be dominated by "
        "limits that cannot fail.",
        "None. Do not average them into a headline.",
        "Wide margins on a testbed part do not show the tolerances are right for a customer."))

    R = {r["construction"]: r for r in ranges}
    for col, title in [
        ("hw_insample", "A 90% range built from the model's own training errors"),
        ("hw_day", "A 90% range built from errors on held-out days"),
        ("hw_row", "A 90% range built from errors on held-out rows"),
    ]:
        r = R[col]
        F.append(Finding("", "range",
            f"{title} covered {pct(r['coverage'])} of parts on days it never saw",
            r["severity"], r["severity_reason"],
            (f"Predicting the longest of the 4 lengths from {len(FEATURES)} process features, each day "
             f"held out whole. Covered {pct(r['coverage'])} [{pct(r['lo'])}, {pct(r['hi'])}] of {n} parts "
             f"against a stated {pct(LEVEL)}; on the {r['n_long']} longest parts {pct(r['coverage_long'])} "
             f"[{pct(r['long_lo'])}, {pct(r['long_hi'])}]. Median width {r['median_width_mm']:.2f} mm. "
             f"By day: " + ", ".join(f"{k} {pct(v)}" for k, v in r["coverage_by_day"].items()) + "."),
            "ranges.json, predictions.json",
            ("A range that covers less than it states will pass parts it should have flagged."
             if r["severity"] != "Observation" else
             "On this evidence the range does what it states, at the width shown."),
            ("Do not quote this range at 90%." if r["severity"] != "Observation" else
             "Usable as stated. Watch the per-day figures: a single day far below the rest is how "
             "a regime change first shows."),
            f"{n} parts, {len(days)} days. With 5 days, one unusual day moves every figure here, and "
            "the intervals treat parts as independent."))

    F.append(Finding("", "range",
        f"The process features improve the prediction of part length by {mae0 - mae:.2f} mm on an error of {mae0:.2f} mm",
        "Observation",
        "recorded as a limit on what the ranges above can be used for, not as a failure of a stated claim",
        (f"Mean absolute error {mae:.2f} mm with {len(FEATURES)} features, {mae0:.2f} mm predicting the "
         f"mean of the training days, against a standard deviation of {model['sd_target_mm']:.2f} mm. "
         f"Predictions span {model['pred_min']:.1f} to {model['pred_max']:.1f} mm; measured lengths span "
         f"{model['target_min']:.1f} to {model['target_max']:.1f} mm."),
        "ranges.json",
        ("A flag-or-ship rule built on this prediction separates long parts only weakly. The decision "
         "curve shows the price: catching every over-limit part means checking about 9 parts in 10."),
        ("Before paying for in-process prediction of length, find the variable that moves it. "
         "Billet length or volume before forging is the obvious candidate and is not in these records."),
        "A fixed, short feature list on 81 parts. A richer model might do better and would need "
        "far more parts to show it honestly."))

    st_sev, st_why = severity_structure(False, True)
    F.append(Finding("", "structure",
        "The two files that state the length tolerance disagree, and one sensor channel is constant",
        st_sev, st_why,
        (f"The measurement file gives the lower tolerance on Top4 as "
         f"{structure['top4_lower_in_measurement_file']:+.0f} mm; the structure summary gives "
         f"'{top4_doc}'. The data set's description says the parts were forged over one day; the "
         f"measurement dates span {len(days)} days ({days[0]} to {days[-1]}). The pyrometer reading at "
         f"the start of forging is the same on all {n} parts ({d['temp_at_forge_start'].mean():.1f}), "
         f"which is the bottom of its range, so billet temperature at the hammer is not recorded. "
         f"{len(structure['long_traces'])} traces are more than 20% longer than the median "
         f"(parts {', '.join(map(str, structure['long_traces']))})."),
        "structure.json",
        "None of these stops the analysis. Each changes how a result should be read, and the first "
        "is a question about which document is the specification.",
        "Confirm the lower length tolerance, the number of production days, and whether a billet "
        "temperature at transfer is logged anywhere else.",
        "A structure check finds what is inconsistent inside the records. It cannot find what the "
        "records leave out without being told."))

    order = {"structure": 0, "limit": 1, "range": 2}
    F.sort(key=lambda f: (rank[f.severity], order[f.kind]))
    for i, f in enumerate(F, 1):
        f.id = f"F-{i:02d}"
    write_lf(out / "findings.json", json.dumps([asdict(f) for f in F], indent=1))
    write_lf(out / "findings.csv", pd.DataFrame([asdict(f) for f in F]).to_csv(index=False))
    print("\nFindings")
    for f in F:
        print(f"  {f.id}  {f.severity:<12} {f.title}")

    DIGEST_RULE = ("sha256 over every file in artifacts/specimen except meta.json, in filename "
                   "order, feeding each file's name as UTF-8 followed by its bytes. meta.json is "
                   "excluded because it carries the digest and cannot hash itself.")
    digest = hashlib.sha256()
    for f in sorted(out.glob("*")):
        if f.name != "meta.json":
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
    meta = dict(
        subject="Radial forging of a hollow part, AFRC testbed, GFM SKK10/R radial forge",
        n_parts=n, n_days=len(days), first_day=days[0], last_day=days[-1],
        n_dimensions=len(dims), n_all_inside=n_all_inside,
        level=LEVEL, ridge=RIDGE, seed=SEED, row_split_repeats=ROW_SPLIT_REPEATS,
        limit_floor=LIMIT_FLOOR, cpk_min=CPK_MIN, untested_sd=UNTESTED_SD,
        source={
            "title": "Sensor data set radial forging at AFRC testbed v2",
            "authors": "Tachtatzis, C.; Gourlay, G.; Andonovic, I.; Panni, O.",
            "publisher": "University of Strathclyde, Advanced Forming Research Centre, via Zenodo",
            "doi": "10.5281/zenodo.3405265", "published": "2019-09-11",
            "licence": "Creative Commons Attribution 4.0 International",
            "archive_md5": "549e0e2f365c51748373f265dff7b268",
        },
        artifact_digest_sha256=digest.hexdigest(), digest_rule=DIGEST_RULE,
        script="src/audit_forging.py",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        features_script_sha256=hashlib.sha256((Path(__file__).parent / "features.py").read_bytes()).hexdigest(),
    )
    write_lf(out / "meta.json", json.dumps(meta, indent=1))
    if args.publish_to:
        dest = Path(args.publish_to)
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(out.glob("*.json")):
            (dest / f.name).write_bytes(f.read_bytes())
        print(f"published to {dest}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

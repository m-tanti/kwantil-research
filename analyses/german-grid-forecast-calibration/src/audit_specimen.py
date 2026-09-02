"""Specimen Vendor Band Audit: the deliverable, computed.

This is the audit a client buys, run on a public series so that every number in
the report regenerates from public data. It is deliberately NOT the article: the
article argues a case, this scores one series and returns a findings register.

Two disciplines the register is held to, both fixed here rather than chosen after
seeing results:

1. COVERAGE INTERVALS ARE BLOCK-BOOTSTRAPPED BY DAY, never binomial. Quarter-hour
   forecast errors are autocorrelated within a day and across neighbouring slots,
   so treating ~38,000 intervals as independent trials understates the interval by
   the design effect. Coverage findings are downgraded to Observation when the
   nominal level sits inside the bootstrap interval, which is the honest behaviour
   and the whole point of stating severity by rule.

2. "EXPENSIVE" IS DEFINED BEFORE SCORING. An interval is expensive if its adverse
   imbalance spread lands in the top EXPENSIVE_Q of all scored intervals. The
   threshold is a constant in this file, computed once over every scored interval
   and applied unchanged to every method and level. The finding that misses
   concentrate where they cost money is the sharpest commercial claim in the
   audit, and it is worthless if the definition can be chosen after the fact.

Emits artifacts/specimen/*.json and findings.csv. Zone-parameterised: the German
re-run against reBAP is --zone DE_LU with a price loader, not a rewrite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import glob
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- pre-registered
WIN, MINP, GAMMA = 90, 60, 0.02        # identical to the article's constructions
LEVELS = (0.80, 0.90, 0.95)
# A band wider than this fraction of mean load is not a decision an operator can
# act on, whatever its coverage. Fixed here so the width finding is scored by the
# same discipline as the coverage one.
WIDTH_USELESS_FRAC = 0.50
PORTFOLIO_MW = 100.0
EXPENSIVE_Q = 0.80                     # top quintile of adverse spread
N_BOOT = 2000
BOOT_SEED = 20260901
CONCENTRATION_TRIGGER = 1.25
HEADLINE_LEVEL = 0.90                  # the level the register leads with           # miss-rate ratio that makes a finding Material


@dataclass
class Finding:
    id: str
    title: str
    severity: str
    severity_reason: str
    measurement: str
    evidence: str
    consequence: str
    action: str
    limit: str


# ---------------------------------------------------------------- severity rule
# Pre-registered. A finding's severity is derived, never asserted:
#
#   Material     nominal coverage lies OUTSIDE the day-block bootstrap interval,
#                AND misses concentrate in expensive intervals by at least
#                CONCENTRATION_TRIGGER, AND that concentration interval excludes 1.
#   Significant  nominal outside the coverage interval, but the concentration
#                test fails either limb (not distinguishable from none, or real
#                but below the materiality floor).
#   Observation  nominal lies INSIDE the coverage interval. The band holds at
#                this level on this evidence; sample size is allowed to win.
#
# The floor exists so that a concentration of 1.02 with a tight interval cannot
# be called Material. It is a commercial judgement and it is stated so a client
# can argue with it.
def severity(nominal_inside_ci: bool, conc: float, conc_lo: float) -> tuple[str, str]:
    if nominal_inside_ci:
        return "Observation", (
            "nominal coverage lies inside the day-block bootstrap interval; "
            "this evidence does not show the band failing at this level")
    if conc >= CONCENTRATION_TRIGGER and conc_lo > 1.0:
        return "Material", (
            f"nominal outside the coverage interval, and misses run {conc:.2f}x more "
            f"frequent in the most expensive intervals (interval lower bound {conc_lo:.2f} > 1)")
    if conc_lo <= 1.0:
        return "Significant", (
            f"nominal outside the coverage interval; concentration {conc:.2f}x is not "
            f"distinguishable from none (interval lower bound {conc_lo:.2f})")
    return "Significant", (
        f"nominal outside the coverage interval; concentration {conc:.2f}x is real but "
        f"below the {CONCENTRATION_TRIGGER:.2f}x materiality floor")

# ------------------------------------------------------------ per-finding text
# What to do about a finding depends on which construction failed and how. A
# register that repeats one action across every row tells a reader the rows were
# templated, and in this one the passing row was being told to recalibrate.
ACTION = {
    "gaussian": ("Re-state the band at its measured coverage, or replace the parametric step. "
                 "The failure is the Gaussian assumption itself: a symmetric interval at a fixed "
                 "multiple of sigma (1.645 at 90%) "
                 "cannot track an error distribution whose shape moves, and widening it uniformly "
                 "buys coverage in the calm hours it already had."),
    "conformal": ("Do not read the conformal guarantee as insurance here. It is conditional on "
                  "exchangeability, and a trailing window under distribution shift violates that "
                  "condition, which is why this construction covers no better than the parametric "
                  "one it was meant to replace. Either recalibrate adaptively or widen on a "
                  "measured schedule."),
    "aci": ("Nothing to remediate on coverage. Three things to watch instead. Its misses are "
            "the most price-concentrated of any construction audited, which the severity rule "
            "does not grade because concentration only enters once coverage has failed: the band "
            "holds its promise and still misses disproportionately when a miss is expensive, and "
            "the exposure column is where that shows. Beyond that: the band pays for "
            "its coverage in width, so read this row against the width line before reserving "
            "against it; and it is reactive, so a regime break costs a stretch of "
            "under-coverage before it recovers. If either matters, the question is how much "
            "width the desk can carry, not whether to recalibrate."),
}
LIMIT = {
    "gaussian": ("Measured on this series over this period. The interval is block-bootstrapped by "
                 "day and so carries within-day dependence; it does not carry uncertainty in the "
                 "price series used to value the misses."),
    "conformal": ("As above. Note also that this construction is the static split-conformal band, "
                  "not the adaptive one: the finding says nothing about conformal methods that "
                  "update, which are audited separately as F-03."),
    "aci": ("An Observation is not a pass in general. It says the stated level sits inside the "
            "bootstrap interval on this series and period, at a sample size this audit reports "
            "rather than hides. It does not establish that the band holds through a regime change, "
            "which this period contains only in part."),
}


def write_lf(path: Path, text: str) -> None:
    """Write artifacts with LF endings, as bytes, on every platform.

    The digest hashes what the script wrote; git hands a reader what it checked
    out. Those were not the same file. Path.write_text emits CRLF on Windows
    while pandas' to_json emits LF, so one artifact in seven was the odd one out
    and git's end-of-line normalisation rewrote it between hashing and commit.
    Every recomputation from a clean checkout then failed on a file nobody had
    touched. Writing bytes with LF, plus a .gitattributes that tells git to keep
    its hands off, makes the hashed bytes and the shipped bytes the same object.
    """
    path.write_bytes(text.replace(chr(13) + chr(10), chr(10)).encode("utf-8"))


def load_series(raw: Path, name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / name / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {raw / name}; see DATA.md for the pull step")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df[~df.index.duplicated(keep="first")].sort_index()


def build_panel(raw: Path, tz: str) -> pd.DataFrame:
    fc = load_series(raw, "load_forecast").iloc[:, 0]
    act = load_series(raw, "load_actual").iloc[:, 0]
    imb = load_series(raw, "imbalance_prices")
    da = load_series(raw, "day_ahead_prices").iloc[:, 0]

    grid = pd.date_range(act.index.min(), act.index.max(), freq="15min", tz="UTC")
    p = pd.DataFrame(index=grid)
    p["fc"], p["act"] = fc.reindex(grid), act.reindex(grid)
    p["long"] = imb["Long"].reindex(grid) if "Long" in imb else np.nan
    p["short"] = imb["Short"].reindex(grid) if "Short" in imb else np.nan
    p["da"] = da.reindex(grid).ffill(limit=3)
    p = p.dropna(subset=["fc", "act", "da"])

    local = p.index.tz_convert(tz)
    p["date"] = local.date
    p["slot"] = local.hour * 4 + local.minute // 15
    p["month"] = [f"{d.year:04d}-{d.month:02d}" for d in local]
    p["err"] = p["act"] - p["fc"]
    return p


def to_matrix(p: pd.DataFrame, col: str) -> tuple[np.ndarray, pd.Index]:
    """(date x slot) matrix. Everything downstream is numpy, so the scoring loop
    never touches a pandas index; the original version did a .loc per interval."""
    w = p.pivot_table(index="date", columns="slot", values=col, aggfunc="first")
    return w.to_numpy(dtype=float), w.index


def score(p: pd.DataFrame) -> pd.DataFrame:
    """One row per scored interval per method per level."""
    E, dates = to_matrix(p, "err")
    n_dates, n_slots = E.shape

    p = p.copy()
    adverse = np.where(
        p["err"].to_numpy() > 0,
        np.maximum(p["short"].to_numpy() - p["da"].to_numpy(), 0.0),
        np.maximum(p["da"].to_numpy() - p["long"].to_numpy(), 0.0),
    )
    p["adverse"] = adverse
    A, _ = to_matrix(p, "adverse")
    M = p.pivot_table(index="date", columns="slot", values="month", aggfunc="first").to_numpy()

    rows = []
    for level in LEVELS:
        alpha = 1 - level
        z = norm.ppf(1 - alpha / 2)
        a_t = np.full(n_slots, alpha)          # ACI state, per slot, sequential in t
        for t in range(MINP, n_dates):
            w0 = max(0, t - WIN)
            for s in range(n_slots):
                y = E[t, s]
                if np.isnan(y):
                    continue
                window = E[w0:t, s]
                window = window[~np.isnan(window)]
                if len(window) < MINP:
                    continue
                mu, sd = window.mean(), window.std(ddof=1)
                bands = {
                    "gaussian": (mu - z * sd, mu + z * sd),
                    "conformal": (np.quantile(window, alpha / 2),
                                  np.quantile(window, 1 - alpha / 2)),
                }
                aa = float(np.clip(a_t[s], 0.001, 0.45))
                bands["aci"] = (np.quantile(window, aa / 2), np.quantile(window, 1 - aa / 2))
                # Oracle: this slot's quantiles over the WHOLE period, which no
                # forecaster could have known. It is not a competitor and never
                # enters the register; it exists so the cost table has a floor.
                # A perfectly calibrated 90% band still misses 10% of the time
                # and those misses still cost money, and without this line a
                # reader cannot tell how much of the exposure was avoidable.
                # Fitted on the SCORED rows only. Fitting on all days and
                # scoring the post-warm-up subset left the oracle about a point
                # below nominal (0.887 against 0.899 in-sample) identically in
                # both zones, which is a property of the subset rather than of
                # either series and read as a broken reference.
                full = E[MINP:, s][~np.isnan(E[MINP:, s])]
                bands["oracle"] = (np.quantile(full, alpha / 2), np.quantile(full, 1 - alpha / 2))
                a_t[s] = a_t[s] + GAMMA * (alpha - (0.0 if bands["aci"][0] <= y <= bands["aci"][1] else 1.0))

                for method, (lo, hi) in bands.items():
                    inside = lo <= y <= hi
                    excess = 0.0 if inside else (y - hi if y > hi else lo - y)
                    rows.append((str(dates[t]), int(s), M[t, s], level, method,
                                 bool(inside), float(hi - lo), float(excess),
                                 float(A[t, s]) if not np.isnan(A[t, s]) else np.nan))
    return pd.DataFrame(rows, columns=["date", "slot", "month", "level", "method",
                                       "inside", "width", "excess", "adverse"])


def day_counts(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Collapse a method-level group to per-day counts.

    The bootstrap resamples days, and every statistic it needs is a ratio of
    sums over days, so the counts are all it ever has to touch. Resampling the
    rows themselves (concatenating day frames per replicate) is the same
    estimator and about three orders of magnitude slower.
    """
    d = g.groupby("date", sort=True)
    exp = g[g["expensive"]].groupby("date", sort=True)
    chp = g[~g["expensive"]].groupby("date", sort=True)
    days = d.size().index
    reidx = lambda x: x.reindex(days).fillna(0).to_numpy(dtype=float)
    return {
        "n": reidx(d.size()),
        "inside": reidx(d["inside"].sum()),
        "n_exp": reidx(exp.size()),
        "miss_exp": reidx(exp["inside"].size() - exp["inside"].sum()),
        "n_chp": reidx(chp.size()),
        "miss_chp": reidx(chp["inside"].size() - chp["inside"].sum()),
    }


def stream(*key) -> np.random.Generator:
    """A generator keyed by what is being estimated, not by call order.

    A single shared Generator makes every interval depend on how many draws
    preceded it, so adding one statistic moves the intervals of every statistic
    after it. The numbers then look unstable when nothing about the estimate has
    changed. Seeding from the key makes each interval reproducible on its own.
    """
    digest = hashlib.sha256(("|".join(str(k) for k in key)).encode()).digest()
    return np.random.default_rng([BOOT_SEED, int.from_bytes(digest[:8], "big")])


def boot_days(c: dict[str, np.ndarray], stat, rng: np.random.Generator) -> tuple[float, float]:
    """Resample whole DAYS with replacement. A day is the block because the
    dependence that breaks the independent-trials assumption lives inside it."""
    n_days = len(c["n"])
    idx = rng.integers(0, n_days, size=(N_BOOT, n_days))
    vals = stat({k: v[idx].sum(axis=1) for k, v in c.items()})
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="NL")
    ap.add_argument("--raw", default=None, help="raw dir (default artifacts/raw_<zone lower>)")
    ap.add_argument("--tz", default="Europe/Amsterdam")
    # The site renders its own copy of these artifacts, because a static deploy
    # cannot read across repositories. Two copies kept in step by hand is why the
    # published lineage digest drifted from the repository's: writing both in one
    # run removes the step that gets forgotten.
    ap.add_argument("--publish-to", default=None,
                    help="also write the artifacts here (the site's specimen data dir)")
    args = ap.parse_args()

    raw = Path(args.raw) if args.raw else ROOT / "artifacts" / f"raw_{args.zone.lower()}"
    out = ROOT / "artifacts" / f"specimen_{args.zone.lower()}"
    out.mkdir(parents=True, exist_ok=True)

    p = build_panel(raw, args.tz)
    print(f"panel: {len(p):,} quarter-hours, {p['date'].nunique()} days, "
          f"{p['month'].nunique()} months, zone {args.zone}")
    print(f"imbalance price coverage: long {p['long'].notna().mean():.1%}, "
          f"short {p['short'].notna().mean():.1%}")

    fc, ac = p["fc"].to_numpy(), p["act"].to_numpy()
    slope_i, slope = np.polynomial.polynomial.polyfit(fc, ac, 1)
    integrity = dict(
        correlation=float(np.corrcoef(fc, ac)[0, 1]),
        ols_slope=float(slope), ols_intercept=float(slope_i),
        mean_bias_mw=float((ac - fc).mean()),
        mean_bias_frac_load=float((ac - fc).mean() / ac.mean()),
        thresholds=dict(min_correlation=0.90, slope_within=0.15, max_abs_bias_frac=0.05))
    integrity["passes"] = bool(
        integrity["correlation"] >= 0.90
        and abs(integrity["ols_slope"] - 1.0) <= 0.15
        and abs(integrity["mean_bias_frac_load"]) <= 0.05)
    print(f"series integrity: correlation {integrity['correlation']:.3f}, "
          f"slope {integrity['ols_slope']:.3f}, "
          f"mean bias {integrity['mean_bias_frac_load']:+.2%} of load "
          f"-> {'PASSES' if integrity['passes'] else 'FAILS'}")

    s = score(p)
    s = s.dropna(subset=["adverse"])
    print(f"scored: {len(s):,} method-level-interval rows "
          f"({s['date'].nunique()} days after the {MINP}-day warm-up)")

    # --- the pre-registered threshold, computed once over every scored interval
    universe = s.drop_duplicates(subset=["date", "slot"])
    expensive_threshold = float(np.quantile(universe["adverse"], EXPENSIVE_Q))
    s["expensive"] = s["adverse"] >= expensive_threshold
    print(f"pre-registered 'expensive' = adverse spread >= EUR {expensive_threshold:.2f}/MWh "
          f"(top {(1-EXPENSIVE_Q)*100:.0f}% of {len(universe):,} scored intervals)")

    mean_load = float(p["act"].mean())
    coverage = []
    for (level, method), g in s.groupby(["level", "method"]):
        cov = g["inside"].mean()
        c = day_counts(g)
        lo, hi = boot_days(c, lambda b: b["inside"] / np.maximum(b["n"], 1),
                           stream(method, level, "coverage"))
        exp_miss = 1 - g[g["expensive"]]["inside"].mean()
        cheap_miss = 1 - g[~g["expensive"]]["inside"].mean()
        ratio = exp_miss / cheap_miss if cheap_miss > 0 else np.nan
        r_lo, r_hi = boot_days(
            c, lambda b: (b["miss_exp"] / np.maximum(b["n_exp"], 1))
                         / np.maximum(b["miss_chp"] / np.maximum(b["n_chp"], 1), 1e-9),
            stream(method, level, "concentration"))
        coverage.append(dict(level=level, method=method, n=int(len(g)),
                             coverage=float(cov), lo=lo, hi=hi,
                             nominal_inside_ci=bool(lo <= level <= hi),
                             miss_expensive=float(exp_miss), miss_cheap=float(cheap_miss),
                             concentration=float(ratio),
                             concentration_lo=float(r_lo), concentration_hi=float(r_hi),
                             mean_width=float(g["width"].mean()),
                             p95_width=float(np.quantile(g["width"], 0.95)),
                             max_width=float(g["width"].max()),
                             share_unusable=float((g["width"] > WIDTH_USELESS_FRAC * mean_load).mean()),
                             # Where a band spends its width. A band can lower its
                             # exposure simply by being wider everywhere, so the
                             # question that separates skill from padding is
                             # whether the extra width lands in the intervals that
                             # cost money.
                             width_expensive=float(g[g["expensive"]]["width"].mean()),
                             width_cheap=float(g[~g["expensive"]]["width"].mean()),
                             # How big a miss is when it happens. Coverage counts
                             # misses; exposure is paid on their size, and the two
                             # come apart for any band that changes width over
                             # time rather than across slots.
                             mean_excess_on_miss=float(g[~g["inside"]]["excess"].mean()),
                             p95_excess_on_miss=float(np.quantile(g[~g["inside"]]["excess"], 0.95)),
                             max_width_frac_load=float(g["width"].max() / mean_load)))
        print(f"  {method:<10} {level:.0%}  coverage {cov:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  nominal {'INSIDE' if lo <= level <= hi else 'outside'}  "
              f"concentration {ratio:.2f}x [{r_lo:.2f}, {r_hi:.2f}]")

    for c_ in coverage:
        c_["severity"], c_["severity_reason"] = severity(
            c_["nominal_inside_ci"], c_["concentration"], c_["concentration_lo"])
    write_lf(out / "coverage.json", json.dumps(coverage, indent=1))

    # ---- cost: the excess beyond the band edge, priced at that interval's adverse
    # spread. Favourable spreads were already clipped to zero upstream, so this is
    # the risk side only and is stated as such in the assumptions register.
    scale = PORTFOLIO_MW / p["act"].mean()
    s["cost"] = s["excess"] * scale * s["adverse"] * 0.25          # MW -> MWh
    cost = (s.groupby(["level", "method"])["cost"].sum().reset_index()
              .rename(columns={"cost": "eur"}))
    cost["eur"] = cost["eur"].round(0)
    missed = s[~s["inside"]]
    spread_note = dict(
        mean_adverse_all=float(universe["adverse"].mean()),
        mean_adverse_on_miss=float(
            missed[(missed["method"] == "gaussian") & (missed["level"] == 0.90)]["adverse"].mean()),
        scale_factor=float(scale), portfolio_mw=PORTFOLIO_MW)
    write_lf(out / "cost.json",
             json.dumps({"by_method_level": cost.to_dict(orient="records"), **spread_note}, indent=1))
    head = cost[(cost.level == 0.90)].set_index("method")["eur"]
    print()
    print(f"cost @90%: gaussian EUR {head['gaussian']:,.0f}  conformal EUR "
          f"{head['conformal']:,.0f}  adaptive EUR {head['aci']:,.0f} per {PORTFOLIO_MW:.0f} MW / 13 mo")
    print(f"mean adverse spread: all EUR {spread_note['mean_adverse_all']:.1f}/MWh, "
          f"on gaussian-90 misses EUR {spread_note['mean_adverse_on_miss']:.1f}/MWh")

    # ---- findings register
    # One finding per construction, headlined at HEADLINE_LEVEL, with the other
    # levels carried inside it as supporting rows. A construction's severity is
    # the most severe it reaches at ANY audited level: a band that fails
    # materially at 95% has failed, and reporting only the headline level would
    # let the register choose its own evidence. The CSV keeps all nine rows.
    RANK = {"Observation": 0, "Significant": 1, "Material": 2}
    LABEL = {"oracle": "Oracle band (lookahead, not achievable)",
             "gaussian": "Parametric Gaussian band",
             "conformal": "Rolling split-conformal band",
             "aci": "Adaptive conformal band"}
    eur_of = lambda lv, m: float(
        cost[(cost.level == lv) & (cost.method == m)]["eur"].iloc[0])

    findings, n = [], 0
    for method in ("gaussian", "conformal", "aci"):   # oracle excluded: it is a floor, not a competitor
        rows = sorted([c_ for c_ in coverage if c_["method"] == method],
                      key=lambda r: r["level"])
        head = next(r for r in rows if r["level"] == HEADLINE_LEVEL)
        top = max(RANK[r["severity"]] for r in rows)
        worst = (head if RANK[head["severity"]] == top
                 else next(r for r in rows if RANK[r["severity"]] == top))
        n += 1
        support = "; ".join(
            f"{r['level']:.0%}: covered {r['coverage']:.1%} "
            f"[{r['lo']:.1%}, {r['hi']:.1%}], concentration {r['concentration']:.2f}x "
            f"[{r['concentration_lo']:.2f}, {r['concentration_hi']:.2f}], {r['severity']}"
            for r in rows)
        sev_note = (worst["severity_reason"] if worst["level"] == HEADLINE_LEVEL else
                    f"{worst['severity_reason']} (reached at the {worst['level']:.0%} level)")
        findings.append(Finding(
            id=f"F-{n:02d}",
            title=(f"{LABEL[method]} at nominal {HEADLINE_LEVEL:.0%} covered "
                   f"{head['coverage']:.1%}"),
            severity=worst["severity"], severity_reason=sev_note,
            measurement=(f"Headline {HEADLINE_LEVEL:.0%}: coverage {head['coverage']:.3f}, "
                         f"day-block bootstrap 95% interval [{head['lo']:.3f}, {head['hi']:.3f}] "
                         f"over {head['n']:,} scored intervals. Every audited level: {support}"),
            evidence="coverage.json, findings.csv; Fig. Coverage ladder, Fig. Concentration",
            consequence=(f"EUR {eur_of(HEADLINE_LEVEL, method):,.0f} of un-reserved adverse "
                         f"exposure per {PORTFOLIO_MW:.0f} MW mean load over the audited period "
                         f"at the {HEADLINE_LEVEL:.0%} level"),
            action=ACTION[method],
            limit=LIMIT[method]))

    write_lf(out / "findings.json", json.dumps([asdict(f) for f in findings], indent=1))

    # The CSV is the full matrix: one row per construction per level, so nothing
    # the register summarises is hidden from someone who wants to check it.
    buf = io.StringIO()
    if True:
        w = csv.writer(buf, lineterminator=chr(10))
        w.writerow(["finding_id", "method", "level", "n_scored", "coverage",
                    "coverage_lo", "coverage_hi", "nominal_inside_ci",
                    "concentration", "concentration_lo", "concentration_hi",
                    "severity", "severity_reason", "eur_exposure",
                    "mean_width_mw", "p95_width_mw", "max_width_mw"])
        for f, method in zip(findings, ("gaussian", "conformal", "aci")):
            for r in sorted([c_ for c_ in coverage if c_["method"] == method],
                            key=lambda x: x["level"]):
                w.writerow([f.id, method, r["level"], r["n"], round(r["coverage"], 4),
                            round(r["lo"], 4), round(r["hi"], 4), r["nominal_inside_ci"],
                            round(r["concentration"], 4), round(r["concentration_lo"], 4),
                            round(r["concentration_hi"], 4), r["severity"],
                            r["severity_reason"], eur_of(r["level"], method),
                            round(r["mean_width"], 1), round(r["p95_width"], 1),
                            round(r["max_width"], 1)])
    write_lf(out / "findings.csv", buf.getvalue())

    for f in findings:
        print(f"  {f.id}  {f.severity:<12} {f.title}")

    # --- bias by slot (the bias clock, this zone)
    bias = (p.groupby("slot")["err"].mean() * -1).round(1)   # sign: + = over-forecast
    write_lf(out / "bias_slot.json",
             json.dumps({"slot": [int(i) for i in bias.index],
                         "over_forecast_mw": [float(v) for v in bias]}, indent=1))

    # --- monthly coverage, for the regime cut
    monthly = (s.groupby(["month", "level", "method"])["inside"].mean().reset_index()
                 .rename(columns={"inside": "coverage"}))
    write_lf(out / "monthly.json", monthly.to_json(orient="records", indent=1))

    # Lineage by content, not by commit. The commit that adds a run cannot be
    # known while the run is happening, so a hash written here always names the
    # parent and sends a reader to a tree without the script in it. A digest over
    # the emitted artifacts is checkable instead: regenerate, hash, compare.
    DIGEST_RULE = ("sha256 over every file in artifacts/specimen except meta.json, "
                   "in filename order, feeding each file's name as UTF-8 followed by "
                   "its bytes. meta.json is excluded because it carries the digest and "
                   "cannot hash itself.")
    digest = hashlib.sha256()
    for f in sorted(out.glob("*")):
        if f.name != "meta.json":
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
    # A digest says whether the outputs match. It does not say what code produced
    # them, which is what a reader actually needs to reproduce the run, and which
    # the commit line used to carry before it was removed for naming the wrong
    # tree. The script hashes itself: checkable, and knowable at run time.
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scored_intervals = int(max(c_["n"] for c_ in coverage))

    meta = dict(zone=args.zone, mean_load_mw=mean_load, integrity=integrity,
                width_useless_frac=WIDTH_USELESS_FRAC, tz=args.tz, window=WIN, min_points=MINP, gamma=GAMMA,
                levels=list(LEVELS), portfolio_mw=PORTFOLIO_MW,
                expensive_quantile=EXPENSIVE_Q, expensive_threshold_eur_mwh=expensive_threshold,
                n_boot=N_BOOT, boot_seed=BOOT_SEED,
                concentration_trigger=CONCENTRATION_TRIGGER,
                panel_rows=int(len(p)), scored_days=int(s["date"].nunique()),
                scored_intervals=scored_intervals,
                artifact_digest_sha256=digest.hexdigest(),
                digest_rule=DIGEST_RULE,
                script='src/audit_specimen.py',
                script_sha256=script_sha,
                months=sorted(p["month"].unique().tolist()))
    write_lf(out / "meta.json", json.dumps(meta, indent=1))

    if args.publish_to:
        dest = Path(args.publish_to)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in sorted(out.glob("*.json")):
            (dest / f.name).write_bytes(f.read_bytes())
            copied += 1
        print(f"published {copied} artifacts to {dest}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Specimen Model Health Check: the deliverable, run on the public backtest.

Reads the row-level backtest output and the null-sensitivity results, and
writes a findings register whose severities are derived by rule, the evidence
tables behind it, and a lineage block (script hash, artifact digest).

Three kinds of finding, each with its own rule, all fixed in this file:

  model     does the VaR breach as often as it states, and how often did the
            miss put it in a worse supervisory backtesting zone
  test      does the expected-shortfall verdict depend on the reference
            distribution, and does the test catch a model built to be broken
  process   was the reference changed in the same revision as the model

The supervisory rule used for the count and the multiplication factor is the
internal-models rule of Regulation (EU) No 575/2013 as it applies on the date
of this run: Article 366 for the overshooting count and the addend, Article
364(1)(a) for the value-at-risk component of the own funds requirement, Article
365(1) for the 99% level and the 10-day holding period, all in the version in
force on 8 July 2024, which Article 520a keeps applicable until 1 January 2027.
SOURCES below records the versions read and is written to meta.json. Only the
VaR component is computed. The stressed-VaR component, which carries the same
addend, is not, so every own-funds figure here understates the effect of the
addend on a real requirement. The rule counts the higher of overshootings on
hypothetical and actual changes in value; this book has fixed weights and one
P&L series, so the two are the same series here.

The book size is an illustration, not evidence: the three-asset book has no
euro value, so one is assumed and labelled wherever it is used.

Usage:
    python scripts/audit_health_check.py
    python scripts/audit_health_check.py --publish-to <dir>
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
from scipy.stats import binom

from var_backtest_uq.backtest.coverage_tests import (
    christoffersen_independence,
    kupiec_pof,
    wilson_score_interval,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ fixed rules
# Fixed before this script scored anything, though not in ignorance: the
# aggregate breach counts of the five models were already published when these
# were written. What was not known is any zone share, interval or euro figure.
HEADLINE_ALPHA = 0.01                  # the 99% level the supervisory rule counts at
ALPHAS = (0.01, 0.025)
WINDOW = 250                           # business days in the overshooting count
AVG_DAYS = 60                          # averaging window of the VaR component
HOLDING_DAYS = 10
BASE_FACTOR = 3.0
N_BOOT = 2000
BOOT_SEED = 20261111
ZONE_SHARE_TRIGGER = 0.25              # share of windows outside green that makes a miss Material
INTEGRITY_FLOOR = 0.01                 # share of days with an impossible VaR and ES pair
INDEPENDENCE_P = 0.05                  # Christoffersen independence, Monte Carlo p-value
RULE_REVISION = (
    "The model rule was revised once, after the first run of this script and before anything "
    "was published. The first version tested breach frequency only. Under it Historical "
    "simulation scored Observation at both levels, because its breaches cluster and the "
    "clustering widens the month-block interval until it contains the stated rate "
    "(1.75% [0.98%, 2.57%] at the 99% level). A VaR also states that breaches do not cluster, "
    "so an independence limb was added. That is a change made after seeing a result, and it "
    "is recorded here for that reason."
)
NULL_CHANGE_TRIGGER = 0.50             # share of ES verdicts that change with the reference
BOOK_EUR = 100_000_000.0               # ILLUSTRATIVE book value; the public book has none
MC_SEED = 4242

# Article 366(2), Table 1: number of overshootings -> addend.
ADDEND = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}

# Article 325bf(6), Table 3: the add-on that replaces Table 1 from 1 January 2027,
# on a base of 1.5. Used only to state the relative uplift under the successor rule.
ADDON_2027 = {5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}
BASE_FACTOR_2027 = 1.5

SOURCES = {
    "rule_applied": (
        "Regulation (EU) No 575/2013, Article 364(1)(a), Article 365(1) and Article 366(2) and "
        "(3) with Table 1, consolidated version of 9 January 2024 (CELEX 02013R0575-20240109). "
        "That is the text in force on 8 July 2024."),
    "why_that_version": (
        "Regulation (EU) 2024/1623 deleted Part Three, Title IV, Chapter 5. Article 520a, as "
        "replaced by Commission Delegated Regulation (EU) 2025/1496 of 12 June 2025, reads: "
        "'Until 1 January 2027, institutions shall continue to apply Part Three, Title IV, and "
        "the market risk requirements of Articles 430, 430b, 445 and 455 of this Regulation in "
        "the version in force on 8 July 2024.'"),
    "successor_rule": (
        "From 1 January 2027: Article 325bf(6) with Table 3, consolidated version of 26 June "
        "2026 (CELEX 02013R0575-20260626). Same count (overshootings of a one-day 99% VaR over "
        "the most recent 250 business days), a base of 1,5 and an add-on of 0,20 to 0,50."),
    "basel": (
        "Zone names (green, yellow, red) and the probability a correct model lands in each: "
        "Basle Committee on Banking Supervision, Supervisory framework for the use of "
        "'backtesting' in conjunction with the internal models approach to market risk capital "
        "requirements, January 1996, Table 2. The Regulation's table carries addends only."),
    "retrieved": "2026-11-11, from the Publications Office of the European Union and bis.org",
    "scope": (
        "The multiplication factor applies to institutions granted permission to use internal "
        "models under Article 363. It does not apply to a firm on the standardised approach."),
}

MODELS = ["parametric", "historical", "distributional", "conformal_pid"]
MODEL_LABEL = {
    "parametric": "Parametric EWMA",
    "historical": "Historical simulation",
    "distributional": "GARCH with a t-copula",
    "conformal_pid": "Conformal-PID",
    "conformal_pid_es": "Conformal-PID, joint VaR and ES",
}
ALL_MODELS = MODELS + ["conformal_pid_es"]
PARAMETRIC_NULLS = ["gaussian", "t_df15", "t_moment", "t_df8", "t_df5"]
NULL_LABEL = {
    "gaussian": "Gaussian",
    "t_df15": "Student-t, df 15",
    "t_moment": "Moment-matched t (as used)",
    "t_df8": "Student-t, df 8",
    "t_df5": "Student-t, df 5",
    "empirical": "Vol-adjusted empirical",
}
NULL_AS_USED = "t_moment"
# What each statistic is meant to detect. Z1 averages over the breach days and
# cannot see their number, so the frequency control is not held against it.
CONTROLS = {
    "z1": {"broken_gauss_matched": "depth"},
    "z2": {"broken_gauss_matched": "depth", "broken_scaled_0.8": "frequency"},
}
CONTROL_LABEL = {
    "broken_gauss_matched": "right breach count, thin tail",
    "broken_scaled_0.8": "VaR and ES scaled by 0.8",
}


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


# ------------------------------------------------------------- supervisory rule
def addend_for(n_overshootings: int) -> float:
    """Article 366(2), Table 1."""
    if n_overshootings < 5:
        return 0.0
    return ADDEND.get(int(n_overshootings), 1.0)


def addon_2027_for(n_overshootings: int) -> float:
    """Article 325bf(6), Table 3, applying from 1 January 2027."""
    if n_overshootings < 5:
        return 0.0
    return ADDON_2027.get(int(n_overshootings), 0.50)


def zone_for(n_overshootings: int) -> str:
    """Green, yellow and red are the Basel Committee's names for the three
    ranges of the same count. The Regulation's table carries the addends only."""
    if n_overshootings < 5:
        return "green"
    return "yellow" if n_overshootings < 10 else "red"


def rolling_count(breaches: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Overshootings in each COMPLETE trailing window. Entry i is the count over
    days i .. i+window-1, so the result has len(breaches) - window + 1 entries.
    Partial windows are not scored: a count over 40 days is not the rule's count."""
    b = np.asarray(breaches, dtype=int)
    if b.size < window:
        return np.empty(0, dtype=int)
    c = np.concatenate([[0], np.cumsum(b)])
    return (c[window:] - c[:-window]).astype(int)


def p_outside_green_if_correct(alpha: float = HEADLINE_ALPHA, window: int = WINDOW) -> float:
    """Share of windows a correctly stated model spends outside green, if
    overshootings were independent: P(Binomial(250, 0.01) >= 5)."""
    return float(1.0 - binom.cdf(4, window, alpha))


def var_component(var_abs_1d: np.ndarray, counts: np.ndarray, book: float = BOOK_EUR) -> dict:
    """VaR component of the own funds requirement, Article 364(1)(a): the higher
    of the previous day's VaR and the 60-business-day average of the daily VaR
    multiplied by the multiplication factor. The one-day VaR is scaled to ten
    days by the square root of time, which Article 365(1) permits.

    Returned per scored day, aligned with `counts` (one entry per complete
    250-day window), with and without the addend, in euros on `book`.
    """
    v10 = np.asarray(var_abs_1d, float) * math.sqrt(HOLDING_DAYS)
    avg60 = pd.Series(v10).rolling(AVG_DAYS, min_periods=AVG_DAYS).mean().to_numpy()
    prev = np.concatenate([[np.nan], v10[:-1]])
    # counts[i] is the window ending on day i + WINDOW - 1
    end = np.arange(counts.size) + WINDOW - 1
    add = np.array([addend_for(int(c)) for c in counts])
    with_add = np.maximum(prev[end], (BASE_FACTOR + add) * avg60[end])
    without = np.maximum(prev[end], BASE_FACTOR * avg60[end])
    return {
        "addend": add,
        "with_addend_eur": with_add * book,
        "base_eur": without * book,
        "extra_eur": (with_add - without) * book,
    }


# ---------------------------------------------------------------- severity rules
def severity_model(rate_inside_ci: bool, independence_p: float, alpha: float,
                   share_outside: float | None, share_lo: float | None,
                   expected_share: float, direction: str = "more") -> tuple[str, str]:
    """Model findings. A VaR states two things: how often it is breached, and
    that breaches do not arrive in clusters. The rule tests both.

    Observation  the stated breach rate lies INSIDE the month-block bootstrap
                 interval of the observed rate, AND independence of breaches is
                 not rejected at INDEPENDENCE_P. Sample size is allowed to win.
    Significant  either limb fails, but the zone limb fails or does not apply
                 (the supervisory count is defined at the 99% level only).
    Material     either limb fails, AND the model spent at least
                 ZONE_SHARE_TRIGGER of complete 250-day windows outside the
                 green zone, AND the lower bound of that share exceeds the
                 share a correctly stated model would spend there.

    Revised once, after the first run and before anything was published: the
    first version had no independence limb. See RULE_REVISION.
    """
    clustered = independence_p < INDEPENDENCE_P
    if rate_inside_ci and not clustered:
        return "Observation", (
            "the stated breach rate lies inside the month-block bootstrap interval of the "
            "observed rate, and independence of breaches is not rejected "
            f"(p = {independence_p:.3f}); this evidence does not show the VaR failing at this level")
    fails = []
    if not rate_inside_ci:
        fails.append(f"the stated breach rate lies outside the interval ({direction} breaches "
                     f"than stated)")
    if clustered:
        fails.append(f"breaches arrive in clusters (independence rejected, p = {independence_p:.3f})")
    why = " and ".join(fails)
    if not math.isclose(alpha, HEADLINE_ALPHA) or share_outside is None:
        return "Significant", (
            f"{why}; the supervisory count is defined at the 99% level only, so no zone "
            f"consequence is scored at this level")
    if share_outside >= ZONE_SHARE_TRIGGER and share_lo > expected_share:
        return "Material", (
            f"{why}, and the model sat outside the green zone in {share_outside:.0%} of 250-day "
            f"windows (interval lower bound {share_lo:.1%}, against {expected_share:.1%} for a "
            f"correctly stated model)")
    if share_lo <= expected_share:
        return "Significant", (
            f"{why}; time outside the green zone, {share_outside:.0%} of windows, is not "
            f"distinguishable from the {expected_share:.1%} a correctly stated model would spend "
            f"there (interval lower bound {share_lo:.1%})")
    return "Significant", (
        f"{why}; time outside the green zone, {share_outside:.0%} of windows, is real but below "
        f"the {ZONE_SHARE_TRIGGER:.0%} materiality floor")


def severity_test(share_changed: float, controls_missed: list[str]) -> tuple[str, str]:
    """Test findings.

    Material     the test passes a model built to be broken in a way the
                 statistic is meant to detect, OR more than NULL_CHANGE_TRIGGER
                 of its verdicts change zone across the parametric references.
    Significant  at least one verdict changes zone, and every control is caught.
    Observation  no verdict changes zone, and every control is caught.
    """
    reasons = []
    if controls_missed:
        reasons.append("it passes a model built to be broken (" + "; ".join(controls_missed) + ")")
    if share_changed > NULL_CHANGE_TRIGGER:
        reasons.append(f"{share_changed:.0%} of its verdicts change zone when only the "
                       f"reference distribution changes")
    if reasons:
        return "Material", " and ".join(reasons)
    if share_changed > 0:
        return "Significant", (
            f"{share_changed:.0%} of its verdicts change zone when only the reference "
            f"distribution changes; every known-broken model it is meant to detect is caught")
    return "Observation", (
        "no verdict changes zone across the parametric references, and every known-broken "
        "model it is meant to detect is caught")


def severity_integrity(worst_share: float) -> tuple[str, str]:
    """Integrity findings. An expected shortfall is the average loss beyond the
    VaR, so a model that states an ES shallower than its own VaR has stated an
    impossible pair.

    Material     the pair is impossible on more than INTEGRITY_FLOOR of days at
                 any audited level. Every ES verdict for that model is void.
    Observation  it happens, on no more than INTEGRITY_FLOOR of days.
    """
    if worst_share > INTEGRITY_FLOOR:
        return "Material", (
            f"the model states an expected shortfall shallower than its own VaR on "
            f"{worst_share:.0%} of days at its worst audited level, above the "
            f"{INTEGRITY_FLOOR:.0%} floor; no ES verdict for it can be graded")
    return "Observation", (
        f"an expected shortfall shallower than the VaR occurs on {worst_share:.1%} of days, "
        f"at or below the {INTEGRITY_FLOOR:.0%} floor")


def severity_process(changed_together: bool, model_effect: float, reference_effect: float,
                     model_alone_passes: bool) -> tuple[str, str]:
    """Process finding.

    Material     the reference and the model changed in one revision, AND the
                 reference moved the threshold further than the model moved the
                 statistic, AND the model change alone does not reach a pass.
    Significant  they changed in one revision, but the model change carries the
                 result on its own.
    Observation  the two were changed separately.
    """
    if not changed_together:
        return "Observation", "the reference and the model were changed in separate revisions"
    ratio = abs(reference_effect) / abs(model_effect) if model_effect else float("inf")
    if abs(reference_effect) > abs(model_effect) and not model_alone_passes:
        return "Material", (
            f"the reference and the model changed in one revision; the reference moved the "
            f"threshold {ratio:.0f} times as far as the model moved the statistic, and "
            f"against the original reference the changed model still does not pass")
    return "Significant", (
        "the reference and the model changed in one revision, but the model change reaches "
        "a pass against the original reference on its own")


# ------------------------------------------------------------------- bootstrap
def stream(*key) -> np.random.Generator:
    """A generator keyed by what is being estimated, not by call order."""
    digest = hashlib.sha256(("|".join(str(k) for k in key)).encode()).digest()
    return np.random.default_rng([BOOT_SEED, int.from_bytes(digest[:8], "big")])


def month_blocks(dates: pd.Series) -> list[np.ndarray]:
    """Row positions of each calendar month, in order. The month is the block
    because breaches cluster: a bad week contributes several, and treating 2,172
    days as independent trials gives an interval tighter than the evidence."""
    key = pd.DatetimeIndex(dates).to_period("M")
    pos = np.arange(len(dates))
    return [pos[key == m] for m in key.unique()]


def boot_months(blocks: list[np.ndarray], stat, rng: np.random.Generator) -> tuple[float, float]:
    """Resample whole months with replacement, N_BOOT times; 2.5 and 97.5
    percentiles of `stat(row_positions)`."""
    n = len(blocks)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, n, size=n)
        vals[i] = stat(np.concatenate([blocks[j] for j in pick]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------- integrity
def integrity(g: pd.DataFrame) -> dict:
    """Checked before anything is scored. A backtest of a series that is not a
    forecast and its own outcome grades nothing."""
    var = g["var"].to_numpy(float)
    es = g["es"].to_numpy(float)
    pnl = g["realized_pnl"].to_numpy(float)
    flag = g["breached"].astype(bool).to_numpy()
    scale_ratio = float(np.mean(np.abs(pnl)) / np.mean(np.abs(var)))
    shallow = es > var + 1e-12
    checks = {
        "n_days": int(len(g)),
        "es_shallower_days": int(shallow.sum()),
        "es_shallower_share": float(shallow.mean()),
        "es_shallower_median_gap": (float(np.median((es[shallow] - var[shallow])
                                                    / np.abs(var[shallow])))
                                    if shallow.any() else 0.0),
        "es_shallower_on_breach_days": int((shallow & (pnl < var)).sum()),
        "breach_flag_matches_series": bool(np.array_equal(flag, pnl < var)),
        "es_at_least_as_deep_as_var": bool(not shallow.any()),
        "var_is_a_loss_level": bool(np.all(var < 0)),
        "no_repeated_dates": bool(g["date"].is_unique),
        "scale_ratio": scale_ratio,
        "scale_ratio_bounds": [0.10, 0.70],
    }
    checks["scale_plausible"] = bool(0.10 <= scale_ratio <= 0.70)
    checks["passes"] = all(v for k, v in checks.items()
                           if isinstance(v, bool))
    return checks


# ------------------------------------------------------------------------ text
ACTION_MODEL = {
    "Material": ("Do not quote this VaR at its stated level. Either re-state it at the level it "
                 "delivered, or replace the step that sets its width: the breach count is too "
                 "high for too long to be a bad year. Read the row below it in the table before "
                 "choosing a replacement, because the model that holds its level is also the "
                 "one that quotes the larger VaR."),
    "Significant": ("Re-state the VaR at the level it delivered, or recalibrate. The miss is "
                    "established; its supervisory consequence is not, on this evidence."),
    "Observation": ("Nothing to remediate on breach frequency. Keep the count under review: "
                    "an Observation is a statement about this window, and the interval beside "
                    "it says how much a window of this length can show."),
}
ACTION_TOO_FEW = ("The VaR at this level is wider than its stated level requires: it was breached "
                  "less often than it says. That is the cheaper direction to be wrong in, and it "
                  "is still a misstatement. Re-state the level, or accept the width as a "
                  "deliberate margin and record it as one.")
LIMIT_MODEL = ("One three-asset book over one period. The interval resamples calendar months, so "
               "it carries clustering within a month and not dependence across months. The zone "
               "shares come from overlapping 250-day windows, of which this period holds fewer "
               "than 9 that do not overlap.")
LIMIT_MONEY = (" The euro figures assume an illustrative book of EUR 100 million, cover the VaR "
               "component only, and scale a one-day VaR to ten days by the square root of time.")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def write_lf(path: Path, text: str) -> None:
    """LF bytes on every platform, so the hashed bytes are the shipped bytes."""
    path.write_bytes(text.replace(chr(13) + chr(10), chr(10)).encode("utf-8"))


# ------------------------------------------------------------------------ main
def score_model(g: pd.DataFrame, method: str, alpha: float) -> dict:
    g = g.sort_values("date").reset_index(drop=True)
    b = (g["realized_pnl"].to_numpy(float) < g["var"].to_numpy(float)).astype(int)
    var_abs = np.abs(g["var"].to_numpy(float))
    n, x = int(b.size), int(b.sum())
    blocks = month_blocks(g["date"])
    lo, hi = boot_months(blocks, lambda idx: b[idx].mean(), stream(method, alpha, "rate"))
    w_lo, w_hi = wilson_score_interval(x, n, level=0.95)
    kup = kupiec_pof(b, alpha=alpha, n_simulations=5000, seed=MC_SEED)
    ind = christoffersen_independence(b, n_simulations=5000, seed=MC_SEED)
    row = {
        "method": method, "label": MODEL_LABEL[method], "alpha": alpha, "level": 1 - alpha,
        "n_days": n, "n_breaches": x, "expected_breaches": n * alpha,
        "rate": x / n, "rate_lo": lo, "rate_hi": hi,
        "wilson_lo": w_lo, "wilson_hi": w_hi,
        "stated_inside_ci": bool(lo <= alpha <= hi),
        "kupiec_p": float(kup["p_value"]), "independence_p": float(ind["p_value"]),
        "mean_var_abs": float(var_abs.mean()),
        "n_months": len(blocks),
    }
    if math.isclose(alpha, HEADLINE_ALPHA):
        counts = rolling_count(b)
        cap = var_component(var_abs, counts)
        outside = counts >= 5

        def share_stat(idx):
            c = rolling_count(b[idx])
            return float((c >= 5).mean()) if c.size else np.nan

        def extra_stat(idx):
            c = rolling_count(b[idx])
            return float(var_component(var_abs[idx], c)["extra_eur"].mean()) if c.size else np.nan

        s_lo, s_hi = boot_months(blocks, share_stat, stream(method, alpha, "share"))
        e_lo, e_hi = boot_months(blocks, extra_stat, stream(method, alpha, "extra"))
        row.update({
            "n_windows": int(counts.size),
            "n_windows_non_overlapping": int(counts.size // WINDOW + 1),
            "share_green": float((counts < 5).mean()),
            "share_yellow": float(((counts >= 5) & (counts < 10)).mean()),
            "share_red": float((counts >= 10).mean()),
            "share_outside_green": float(outside.mean()),
            "share_outside_lo": s_lo, "share_outside_hi": s_hi,
            "max_count": int(counts.max()), "last_count": int(counts[-1]),
            "last_zone": zone_for(int(counts[-1])), "last_addend": addend_for(int(counts[-1])),
            "mean_addend": float(cap["addend"].mean()),
            "mean_uplift": float(cap["addend"].mean() / BASE_FACTOR),
            "mean_uplift_2027": float(
                np.mean([addon_2027_for(int(c)) for c in counts]) / BASE_FACTOR_2027),
            "mean_base_eur": float(cap["base_eur"].mean()),
            "mean_with_addend_eur": float(cap["with_addend_eur"].mean()),
            "mean_extra_eur": float(cap["extra_eur"].mean()),
            "extra_lo_eur": e_lo, "extra_hi_eur": e_hi,
            "peak_extra_eur": float(cap["extra_eur"].max()),
        })
        row["_counts"] = counts
        row["_dates"] = g["date"].to_numpy()[WINDOW - 1:]
        row["_cap"] = cap
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", default=str(ROOT / "experiments" / "factorial" / "t" / "details.csv"))
    ap.add_argument("--sweep", default=str(ROOT / "experiments" / "factorial" / "t" / "as_null_sensitivity.csv"))
    ap.add_argument("--sweep-control-arm",
                    default=str(ROOT / "experiments" / "factorial" / "gaussian" / "as_null_sensitivity.csv"))
    ap.add_argument("--control", default=str(ROOT / "experiments" / "control_v0.1" / "details.csv"),
                    help="Row-level output of release 0.1.x, kept as the case where the "
                         "integrity check fires. Optional.")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "health_check"))
    ap.add_argument("--publish-to", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.details, parse_dates=["date"])
    sweep = pd.read_csv(args.sweep)
    sweep_g = pd.read_csv(args.sweep_control_arm)
    expected_share = p_outside_green_if_correct()

    # --- integrity, before anything is scored
    integ = {m: {f"{1 - a:.3f}": integrity(d[(d["method"] == m) & np.isclose(d["alpha"], a)])
                 for a in ALPHAS} for m in ALL_MODELS}
    integ_pass = {m: all(v["passes"] for v in by_level.values()) for m, by_level in integ.items()}

    # The control: the same check on the output of release 0.1.x, in which the
    # joint model stated an ES shallower than its own VaR. That defect was found
    # by this check and fixed in the model; the old run is kept because a check
    # nobody has seen fail is a check nobody should trust.
    control = None
    if Path(args.control).exists():
        c = pd.read_csv(args.control, parse_dates=["date"])
        lv = {f"{1 - a:.3f}": integrity(c[(c["method"] == "conformal_pid_es") & np.isclose(c["alpha"], a)])
              for a in ALPHAS}
        worst = max(lv.values(), key=lambda v: v["es_shallower_share"])
        control = {
            "label": "Conformal-PID, joint VaR and ES, as released in 0.1.x",
            "levels": lv, "passes": all(v["passes"] for v in lv.values()),
            "severity_if_reported": severity_integrity(worst["es_shallower_share"])[0],
            "source": "experiments/control_v0.1/details.csv (the published run of 2026-08-24, "
                      "rebuilt by checking out release 0.1.x; not shipped)",
        }

    # --- model scoring
    rows = [score_model(d[(d["method"] == m) & np.isclose(d["alpha"], a)], m, a)
            for m in ALL_MODELS for a in ALPHAS]
    for r in rows:
        r["direction"] = "more" if r["rate"] > r["alpha"] else "fewer"
        r["severity"], r["severity_reason"] = severity_model(
            r["stated_inside_ci"], r["independence_p"], r["alpha"],
            r.get("share_outside_green"), r.get("share_outside_lo"), expected_share,
            r["direction"])

    # rolling counts, weekly, for the figure
    rolling = []
    for r in rows:
        if "_counts" in r:
            dates = pd.DatetimeIndex(r["_dates"])
            rolling.append({
                "method": r["method"],
                "dates": [t.strftime("%Y-%m-%d") for t in dates[::5]],
                "count": [int(c) for c in r["_counts"][::5]],
                "extra_eur": [round(float(v)) for v in r["_cap"]["extra_eur"][::5]],
            })
    write_lf(out / "rolling.json", json.dumps(rolling, indent=1))

    # breaches by calendar year at the headline level
    yearly = []
    for m in MODELS:
        g = d[(d["method"] == m) & np.isclose(d["alpha"], HEADLINE_ALPHA)]
        b = (g["realized_pnl"] < g["var"]).astype(int)
        by = b.groupby(g["date"].dt.year).agg(["sum", "size"])
        for y, (s, n) in by.iterrows():
            yearly.append({"method": m, "year": int(y), "breaches": int(s), "n_days": int(n),
                           "expected": round(float(n * HEADLINE_ALPHA), 2)})
    write_lf(out / "yearly.json", json.dumps(yearly, indent=1))

    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    write_lf(out / "coverage.json", json.dumps(public, indent=1))

    # --- ES tests against the references
    real = sweep[sweep["method"].isin(ALL_MODELS)]
    tests = []
    for stat in ("z1", "z2"):
        zone_col = f"{stat}_zone"
        piv = real.pivot_table(index=["method", "alpha"], columns="null", values=zone_col,
                               aggfunc="first")
        changed = int((piv[PARAMETRIC_NULLS].nunique(axis=1) > 1).sum())
        n_cells = int(len(piv))
        green_by_null = {n_: int((piv[n_] == "green").sum())
                         for n_ in PARAMETRIC_NULLS + ["empirical"]}
        # A cell passes only in the green zone, so a cell can change zone (yellow to red)
        # without changing between pass and fail. Count the cells whose pass or fail changes.
        is_green = piv[PARAMETRIC_NULLS] == "green"
        pass_changed = int((is_green.any(axis=1) & ~is_green.all(axis=1)).sum())
        controls = []
        missed = []
        for ctrl, kind in CONTROLS[stat].items():
            for a in ALPHAS:
                c = sweep[(sweep["method"] == ctrl) & np.isclose(sweep["alpha"], a)
                          & (sweep["null"] == NULL_AS_USED)].iloc[0]
                caught = c[zone_col] != "green"
                controls.append({"control": ctrl, "kind": kind, "alpha": a,
                                 "zone_as_used": c[zone_col],
                                 "p_value": float(c[f"{stat}_p_value"]),
                                 "kupiec_p": float(c["kupiec_pvalue"]),
                                 "n_breaches": int(c["n_breaches"]), "caught": bool(caught)})
                if not caught:
                    missed.append(f"{CONTROL_LABEL[ctrl]}, green at the {pct(1 - a)} level, "
                                  f"p = {c[f'{stat}_p_value']:.3f}")
        sev, why = severity_test(changed / n_cells, missed)
        tests.append({"statistic": stat, "n_cells": n_cells, "cells_changed": changed,
                      "cells_pass_changed": pass_changed,
                      "share_changed": changed / n_cells, "green_by_null": green_by_null,
                      "controls": controls, "severity": sev, "severity_reason": why})
    write_lf(out / "tests.json", json.dumps(tests, indent=1))

    sweep_rows = []
    for _, s in sweep.iterrows():
        sweep_rows.append({
            "method": s["method"], "alpha": float(s["alpha"]), "null": s["null"],
            "n_breaches": int(s["n_breaches"]),
            "z1": round(float(s["z1"]), 4), "z1_zone": s["z1_zone"],
            "z1_green_threshold": round(float(s["z1_green_threshold"]), 4),
            "z1_p": round(float(s["z1_p_value"]), 4),
            "z2": round(float(s["z2"]), 4), "z2_zone": s["z2_zone"],
            "z2_green_threshold": round(float(s["z2_green_threshold"]), 4),
            "z2_p": round(float(s["z2_p_value"]), 4),
        })
    write_lf(out / "null_sweep.json", json.dumps(sweep_rows, indent=1))

    # --- process: the factorial
    def cell(frame, null):
        return frame[(frame["method"] == "conformal_pid") & np.isclose(frame["alpha"], HEADLINE_ALPHA)
                     & (frame["null"] == null)].iloc[0]
    t_g, t_t = cell(sweep, "gaussian"), cell(sweep, "t_moment")
    g_g = cell(sweep_g, "gaussian")
    model_effect = float(t_g["z1"] - g_g["z1"])
    reference_effect = float(t_t["z1_green_threshold"] - t_g["z1_green_threshold"])
    model_alone_zone = str(t_g["z1_zone"])
    p_sev, p_why = severity_process(True, model_effect, reference_effect,
                                    model_alone_zone == "green")
    process = {
        "model": "conformal_pid", "alpha": HEADLINE_ALPHA,
        "z1_before": float(g_g["z1"]), "z1_after": float(t_g["z1"]),
        "model_effect": model_effect,
        "threshold_original_reference": float(t_g["z1_green_threshold"]),
        "threshold_new_reference": float(t_t["z1_green_threshold"]),
        "reference_effect": reference_effect,
        "ratio": abs(reference_effect) / abs(model_effect),
        "zone_model_change_alone": model_alone_zone,
        "zone_both_changes": str(t_t["z1_zone"]),
        "severity": p_sev, "severity_reason": p_why,
    }
    write_lf(out / "process.json", json.dumps(process, indent=1))

    # --- the register
    findings: list[Finding] = []
    by_key = {(r["method"], r["alpha"]): r for r in rows}
    rank = {"Material": 0, "Significant": 1, "Observation": 2}
    ref = by_key[("conformal_pid", HEADLINE_ALPHA)]
    for m in MODELS:
        h = by_key[(m, HEADLINE_ALPHA)]
        o = by_key[(m, 0.025)]
        worst = min((h, o), key=lambda r: rank[r["severity"]])  # ties go to the 99% level
        levels = "; ".join(
            f"{pct(r['level'])} level: {r['n_breaches']} breaches against {r['expected_breaches']:.1f} "
            f"stated, rate {pct(r['rate'], 2)} [{pct(r['rate_lo'], 2)}, {pct(r['rate_hi'], 2)}], "
            f"{r['severity']}" for r in (h, o))
        measurement = (
            f"{levels}. Intervals are 95% month-block bootstrap over {h['n_months']} calendar months, "
            f"{h['n_days']:,} trading days. At the 99% level the trailing 250-day count reached "
            f"{h['max_count']} and stood at {h['last_count']} on the last day; the model sat outside "
            f"the green zone in {pct(h['share_outside_green'], 0)} of {h['n_windows']:,} complete windows "
            f"[{pct(h['share_outside_lo'], 0)}, {pct(h['share_outside_hi'], 0)}], against "
            f"{pct(expected_share)} for a correctly stated model.")
        if h["mean_extra_eur"] > 0:
            consequence = (
                f"Illustration on an assumed {eur(BOOK_EUR)} book: the addend raised the VaR component "
                f"of the own funds requirement by {eur(h['mean_extra_eur'])} on average over the period "
                f"[{eur(h['extra_lo_eur'])}, {eur(h['extra_hi_eur'])}], and by {eur(h['peak_extra_eur'])} "
                f"at its peak. With the addend the component averaged {eur(h['mean_with_addend_eur'])}")
            consequence += (
                f" (a multiplication factor {pct(h['mean_uplift'], 2)} higher on average; under the "
                f"add-on table that applies from 1 January 2027 the same counts give "
                f"{pct(h['mean_uplift_2027'], 2)})")
            gap = ref["mean_with_addend_eur"] - h["mean_with_addend_eur"]
            if m == "conformal_pid":
                consequence += "."
            elif gap > 0:
                consequence += (
                    f", which is still {eur(gap)} below the {eur(ref['mean_with_addend_eur'])} of "
                    f"{MODEL_LABEL['conformal_pid']}, the model that held its level. The addend did "
                    f"not close the gap the understated VaR opened.")
            else:
                consequence += (f", against {eur(ref['mean_with_addend_eur'])} for "
                                f"{MODEL_LABEL['conformal_pid']}.")
        else:
            consequence = (
                f"Illustration on an assumed {eur(BOOK_EUR)} book: no addend applied in any complete "
                f"window. The VaR component averaged {eur(h['mean_with_addend_eur'])}.")
        too_few = (worst["severity"] != "Observation" and worst["direction"] == "fewer"
                   and not worst["stated_inside_ci"])
        findings.append(Finding(
            id="", kind="model",
            title=(f"{MODEL_LABEL[m]} VaR at the stated {pct(worst['level'])} level was breached on "
                   f"{pct(worst['rate'], 2)} of days, against {pct(worst['alpha'])} stated"),
            severity=worst["severity"], severity_reason=worst["severity_reason"],
            measurement=measurement,
            evidence="coverage.json, rolling.json, yearly.json, findings.csv",
            consequence=consequence,
            action=ACTION_TOO_FEW if too_few else ACTION_MODEL[worst["severity"]],
            limit=LIMIT_MODEL + LIMIT_MONEY))

    stat_name = {"z1": "conditional ES statistic (Z1)", "z2": "unconditional ES statistic (Z2)"}
    for t in tests:
        g = t["green_by_null"]
        controls_txt = "; ".join(
            f"{CONTROL_LABEL[c['control']]} at {pct(1 - c['alpha'])}: {c['zone_as_used']}"
            for c in t["controls"])
        if t["statistic"] == "z1":
            title = (f"The conditional ES test passes {g['gaussian']} of {t['n_cells']} model-and-level "
                     f"cells under one reference and {g['t_df5']} under another")
            action = ("Do not report this zone as evidence that the expected shortfall is adequate. "
                      "Report the reference distribution beside every verdict, and read the statistic "
                      "with an exceedance-count test, which it is designed to be paired with.")
            limit = ("One implementation on one dataset. The references here are moment-matched "
                     "approximations, not each model's exact predictive distribution; with an exact "
                     "reference the threshold moves less. With 17 to 79 breach days per cell the "
                     "statistic is an average over a few dozen numbers.")
        else:
            passes = sorted({g[n_] for n_ in PARAMETRIC_NULLS})
            held = (f"passes the same {passes[0]} of {t['n_cells']} cells under every parametric "
                    f"reference" if len(passes) == 1 else
                    f"passes {passes[0]} to {passes[-1]} of {t['n_cells']} cells across the "
                    f"parametric references")
            title = (f"The unconditional ES test {held} and passes a thin-tailed model with the "
                     f"right breach count")
            action = ("Keep this statistic as the working ES check, paired with an exceedance-count "
                      "test, and do not read a pass as evidence about tail depth. Add a known-broken "
                      "model of each kind to the validation pack so the blind side stays visible.")
            limit = ("The pass and fail verdicts are stable here because the models it rejects "
                     "breach too often by a wide margin. That is a property of these models on this "
                     "data, not of the statistic in general. Stable pass and fail is not a stable "
                     "zone: a rejected cell can still move between yellow and red.")
        findings.append(Finding(
            id="", kind="test", title=title,
            severity=t["severity"], severity_reason=t["severity_reason"],
            measurement=(
                f"Models, data, VaR and ES held fixed; only the reference distribution changes. "
                f"Cells passing by reference: " +
                ", ".join(f"{NULL_LABEL[k]} {v}" for k, v in g.items()) +
                f", of {t['n_cells']} (5 models at 2 levels). {t['cells_changed']} of {t['n_cells']} "
                f"verdicts change zone across the 5 parametric references, and "
                f"{t['cells_pass_changed']} of {t['n_cells']} change between pass and fail. "
                f"Known-broken models under the reference as used: {controls_txt}. Each reference is {N_SIM_NOTE}."),
            evidence="tests.json, null_sweep.json",
            consequence=("A verdict that moves with a modelling choice nobody recorded cannot be "
                         "defended as a finding about the model. No euro figure is attached: an ES "
                         "zone has no supervisory multiplier in the rule this report applies."
                         if t["statistic"] == "z1" else
                         "A pass from this test says the breach count and depth together are not "
                         "alarming. It does not say the tail is as deep as the model claims."),
            action=action, limit=limit))

    findings.append(Finding(
        id="", kind="process",
        title="The reference distribution was changed in the same revision as the model",
        severity=p_sev, severity_reason=p_why,
        measurement=(
            f"{MODEL_LABEL['conformal_pid']} at the 99% level, 2 backtest runs and 2 references. "
            f"The model change moved Z1 by {model_effect:+.4f} (from {process['z1_before']:+.4f} to "
            f"{process['z1_after']:+.4f}). The reference change moved the pass threshold by "
            f"{reference_effect:+.3f} (from {process['threshold_original_reference']:+.3f} to "
            f"{process['threshold_new_reference']:+.3f}). Against the original reference the changed "
            f"model is {model_alone_zone}; with both changes it is {process['zone_both_changes']}."),
        evidence="process.json",
        consequence=("The recorded reason for the improvement was wrong. A validation pack that "
                     "credits a model change for a result the test change produced will not "
                     "survive a second reader who separates the two."),
        action=("Record the reference as a model with its own change log. When a reference and a "
                "model change together, run the 2-by-2 that separates them before attributing "
                "the result; the reference is applied after the fact, so it costs 2 runs."),
        limit=("One model at one level. The 2 effects are point values from seeded runs with "
               "2,000 reference draws each; no interval is attached to their ratio.")))

    for m in ALL_MODELS:
        if integ_pass[m]:
            continue
        lv = integ[m]
        worst_level, worst = max(lv.items(), key=lambda kv: kv[1]["es_shallower_share"])
        if worst["es_shallower_days"] == 0:
            continue
        i_sev, i_why = severity_integrity(worst["es_shallower_share"])
        per_level = "; ".join(
            f"{pct(float(k))} level: {v['es_shallower_days']:,} of {v['n_days']:,} days "
            f"({pct(v['es_shallower_share'], 0)}), median gap {pct(v['es_shallower_median_gap'], 0)} "
            f"of the VaR, {v['es_shallower_on_breach_days']} of them breach days"
            for k, v in lv.items())
        findings.append(Finding(
            id="", kind="integrity",
            title=(f"{MODEL_LABEL[m]} states an expected shortfall shallower than its own VaR on "
                   f"{pct(worst['es_shallower_share'], 0)} of days at the {pct(float(worst_level))} level"),
            severity=i_sev, severity_reason=i_why,
            measurement=(f"Days on which the stated ES is a smaller loss than the stated VaR. "
                         f"{per_level}. The other 4 models never do this."),
            evidence="meta.json (integrity), findings.csv",
            consequence=("The ES statistics divide realised losses by the stated ES. Where the ES "
                         "is shallower than the VaR that ratio has no meaning, so this model's ES "
                         "zones, and its 2 cells in the reference sweep, are counts of a quantity "
                         "the model did not coherently state. Its VaR series is unaffected and is "
                         "the same as Conformal-PID's."),
            action=("Constrain the ES correction so the stated ES can never be shallower than "
                    "the stated VaR, re-run, and withdraw this model's ES verdicts until then."),
            limit=("The check establishes that the pair is impossible, not why. The cause is in "
                   "the model's ES correction step, which this report did not open.")))

    findings.sort(key=lambda f: (rank[f.severity],
                                 {"integrity": 0, "model": 1, "test": 2, "process": 3}[f.kind]))
    for i, f in enumerate(findings, 1):
        f.id = f"F-{i:02d}"
    write_lf(out / "findings.json", json.dumps([asdict(f) for f in findings], indent=1))

    csv_rows = []
    for r in public:
        csv_rows.append({
            "kind": "model", "subject": r["label"], "level": r["level"],
            "severity": r["severity"], "n_days": r["n_days"], "n_breaches": r["n_breaches"],
            "rate": round(r["rate"], 6), "rate_lo": round(r["rate_lo"], 6),
            "rate_hi": round(r["rate_hi"], 6),
            "share_outside_green": r.get("share_outside_green"),
            "share_outside_lo": r.get("share_outside_lo"),
            "share_outside_hi": r.get("share_outside_hi"),
            "mean_extra_eur_illustrative_book": r.get("mean_extra_eur"),
            "severity_reason": r["severity_reason"],
        })
    for t in tests:
        csv_rows.append({"kind": "test", "subject": stat_name[t["statistic"]],
                         "severity": t["severity"], "severity_reason": t["severity_reason"]})
    csv_rows.append({"kind": "process", "subject": "attribution of the ES improvement",
                     "severity": p_sev, "severity_reason": p_why})
    write_lf(out / "findings.csv", pd.DataFrame(csv_rows).to_csv(index=False))

    print("\nFindings")
    for f in findings:
        print(f"  {f.id}  {f.severity:<12} {f.title}")

    # --- lineage
    DIGEST_RULE = ("sha256 over every file in artifacts/health_check except meta.json, in "
                   "filename order, feeding each file's name as UTF-8 followed by its bytes. "
                   "meta.json is excluded because it carries the digest and cannot hash itself.")
    digest = hashlib.sha256()
    for f in sorted(out.glob("*")):
        if f.name != "meta.json":
            digest.update(f.name.encode())
            digest.update(f.read_bytes())
    h = by_key[("parametric", HEADLINE_ALPHA)]
    meta = dict(
        subject="Five VaR and expected-shortfall models on an equal-weight SPY, TLT, GLD book",
        first_day=str(d["date"].min().date()), last_day=str(d["date"].max().date()),
        n_days=int(h["n_days"]), n_months=int(h["n_months"]),
        n_windows=int(h["n_windows"]),
        n_windows_non_overlapping=int(h["n_windows_non_overlapping"]),
        headline_level=1 - HEADLINE_ALPHA, levels=[1 - a for a in ALPHAS],
        window=WINDOW, avg_days=AVG_DAYS, holding_days=HOLDING_DAYS, base_factor=BASE_FACTOR,
        addend_table={"fewer than 5": 0.0, **{str(k): v for k, v in ADDEND.items()},
                      "10 or more": 1.0},
        addon_table_2027={"fewer than 5": 0.0, **{str(k): v for k, v in ADDON_2027.items()},
                          "more than 9": 0.50},
        base_factor_2027=BASE_FACTOR_2027,
        expected_share_outside_green=expected_share,
        zone_share_trigger=ZONE_SHARE_TRIGGER, null_change_trigger=NULL_CHANGE_TRIGGER,
        independence_p=INDEPENDENCE_P, rule_revision=RULE_REVISION,
        book_eur_illustrative=BOOK_EUR,
        n_boot=N_BOOT, boot_seed=BOOT_SEED, mc_seed=MC_SEED,
        integrity=integ, integrity_passes=integ_pass, integrity_floor=INTEGRITY_FLOOR,
        integrity_control=control,
        sources=SOURCES,
        artifact_digest_sha256=digest.hexdigest(), digest_rule=DIGEST_RULE,
        script="scripts/audit_health_check.py",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={
            "details": "experiments/factorial/t/details.csv (rebuilt by run_backtest.py, not shipped)",
            "sweep": "experiments/factorial/t/as_null_sensitivity.csv",
            "sweep_sha256": hashlib.sha256(Path(args.sweep).read_bytes()).hexdigest(),
            "sweep_control_arm": "experiments/factorial/gaussian/as_null_sensitivity.csv",
            "sweep_control_arm_sha256": hashlib.sha256(
                Path(args.sweep_control_arm).read_bytes()).hexdigest(),
        },
    )
    write_lf(out / "meta.json", json.dumps(meta, indent=1))

    if args.publish_to:
        dest = Path(args.publish_to)
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(out.glob("*.json")):
            (dest / f.name).write_bytes(f.read_bytes())
        print(f"published to {dest}")
    print(f"\nwrote {out}")


N_SIM_NOTE = "2,000 simulated paths, seeded, the same paths for both statistics"

if __name__ == "__main__":
    main()

"""Export conditional coverage by market regime, with honest intervals.

The pooled breach rate is the number every backtest reports and the least useful one.
A model can sit at nominal across eight years and breach at ten times nominal in the
twenty days anybody would actually ask about. This artefact is that breakdown.

INTERVALS. Breaches inside a stress window are the opposite of independent: they arrive
in runs, driven by the same underlying dislocation. A binomial interval computed as
though each day were an independent trial would be far too narrow, and would let a
figure like "10% breach rate in the tariff window" read as precise when it rests on two
events in twenty days.

So the interval here is a MOVING BLOCK BOOTSTRAP. Contiguous blocks of trading days are
resampled with replacement, preserving the local dependence structure that makes runs of
breaches possible. Block length is 5 days, or a third of the window where the window is
short. The result is wide, and the width is the message.

Writes regime_coverage.json into the dashboard payload directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from var_backtest_uq.data.stress_windows import STRESS_WINDOWS, tag_dates  # noqa: E402


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N_BOOT = 4000
BLOCK = 5
RNG = np.random.default_rng(20260824)


def block_bootstrap_rate(breaches: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Moving-block bootstrap CI for a breach rate on a dependent series.

    Zero-breach windows are handled separately by the caller: a bootstrap over an
    all-zero series returns [0, 0], which reads as certainty and is the opposite of
    what no events means. The rule of three is used there instead.
    """
    n = len(breaches)
    if n < 6:
        return (float("nan"), float("nan"))
    block = max(2, min(BLOCK, n // 3))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = RNG.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([breaches[s:s + block] for s in starts])[:n]
        out[i] = sample.mean()
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> None:
    details = pd.read_csv(ROOT / "output" / "details.csv", parse_dates=["date"])
    details["regime"] = tag_dates(pd.DatetimeIndex(details["date"])).values
    details.loc[details["regime"] == "", "regime"] = "Calm"

    order = ["Calm"] + [w[0] for w in STRESS_WINDOWS]
    rows = []

    for (method, alpha, regime), g in details.groupby(["method", "alpha", "regime"]):
        g = g.sort_values("date")
        b = g["breached"].to_numpy().astype(float)
        n_days, n_br = len(b), int(b.sum())

        if n_br == 0:
            # Rule of three: with 0 events in n trials the upper 95% bound is ~3/n.
            # A window of 18 days cannot rule out a 16% breach rate, and saying so is
            # the whole point of reporting an interval.
            lo, hi = 0.0, min(1.0, 3.0 / n_days) if n_days else float("nan")
        else:
            lo, hi = block_bootstrap_rate(b)

        informative = bool(np.isfinite(lo) and np.isfinite(hi) and (hi - lo) < 0.5)
        excludes = bool(
            np.isfinite(lo) and np.isfinite(hi) and not (lo <= alpha <= hi)
        )
        rows.append({
            "method": method,
            "alpha": float(alpha),
            "regime": regime,
            "n_days": int(len(b)),
            "n_breaches": int(b.sum()),
            "rate": float(b.mean()),
            "target": float(alpha),
            "ci_lo": lo,
            "ci_hi": hi,
            # ratio to target: the number a risk manager reads first
            "ratio": float(b.mean() / alpha) if alpha > 0 else float("nan"),
            "consistent_with_target": not excludes,
            "excludes_target": excludes,
            # a window can be too short to say anything at all; that is a finding too
            "informative": informative,
            "verdict": (
                "too few days to say" if not informative
                else "below target" if excludes and b.mean() < alpha
                else "ABOVE TARGET" if excludes
                else "consistent with target"
            ),
        })

    rows.sort(key=lambda r: (r["alpha"], order.index(r["regime"]) if r["regime"] in order else 99,
                             r["method"]))

    out_dirs = [ROOT / "output" / "dashboard",
                ROOT.parent / "dashboard" / "static" / "data"]
    payload = json.dumps(rows, separators=(",", ":"))
    for d in out_dirs:
        if d.exists():
            (d / "regime_coverage.json").write_text(payload, encoding="utf-8")
            print(f"wrote {d / 'regime_coverage.json'}")

    # ---- console summary at the regulatory alpha ----
    print(f"\n{'=' * 96}")
    print("CONDITIONAL COVERAGE BY REGIME - alpha = 0.01, with moving-block bootstrap intervals")
    print("=" * 96)
    a = [r for r in rows if r["alpha"] == 0.01]
    methods = sorted({r["method"] for r in a})
    for reg in order:
        rr = [r for r in a if r["regime"] == reg]
        if not rr:
            continue
        print(f"\n  {reg}  ({rr[0]['n_days']} days)")
        for r in sorted(rr, key=lambda x: x["rate"]):
            print(f"    {r['method']:<18}{r['n_breaches']:>4} breaches "
                  f"{100 * r['rate']:>6.2f}%   "
                  f"[{100 * r['ci_lo']:>5.2f}, {100 * r['ci_hi']:>5.2f}]"
                  f"   {r['ratio']:>5.1f}x   {r['verdict']}")


if __name__ == "__main__":
    main()

"""Acerbi-Szekely Z1 and Z2 sensitivity to the null specification.

Z1 is A&S's conditional statistic (mean shortfall ratio over breach days) and
Z2 their unconditional one (the same sum over T*alpha). Both are scored on the
same simulated null paths, so any difference in how far their thresholds move
is a property of the statistic, not of the draws. Releases up to 0.1.x called
the conditional statistic "Z2"; the `z1_*` columns here reproduce that column
bit-for-bit.

The AS zone on the dashboard is computed against a null generated from each
method's own predicted marginal. That is the construction A&S intend for a
specification test, but this implementation does not simulate from the model's
*exact* predictive: it draws from a moment-matched Student-t whose df is a
single scalar from the median predicted excess kurtosis. Manual section 8 says
the sign of the resulting bias is unmeasured. This script measures it.

Six nulls per (method, alpha):

  gaussian      N(mu_t, sigma_t).                       Thin reference.
  t_moment      Student-t, df = 4 + 6/kappa_median.     The shipped v1.1 null.
  t_df5/8/15    Student-t at fixed df.                  Brackets the df choice.
  empirical     mu_t + sigma_t * bootstrap(z_hat).      Distribution-free shape.

The empirical null resamples standardised residuals z_hat = (x_t - mu_t)/sigma_t
and re-applies the per-step scale. The library's
`empirical_h0_sampler_from_residuals` resamples raw values iid, which would
destroy the volatility structure and is not usable here.

Two known-broken models are added, built from a base method (default
conformal_pid) so they share its data and scale path:

  broken_scaled_0.8     VaR, ES and the predictive (mu, sigma) all x 0.8. Too
                        many breaches, tail shape unchanged. A frequency error.
  broken_gauss_matched  Gaussian with the base method's VaR (same breaches) and
                        the Gaussian ES implied by that VaR. Right breach rate,
                        thin tail. A depth error.

Caveat carried into the writeup: the empirical pool is built from the same
realised series the test evaluates, so it is distribution-free about shape but
not independent of the data. It is a different circularity, not an absence of
one.

Usage:
    python scripts/as_null_sensitivity.py --details experiments/factorial/t/details.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from scipy.stats import norm

from var_backtest_uq.backtest.acerbi_szekely import (
    acerbi_szekely_test_1,
    acerbi_szekely_test_2,
    gaussian_h0_sampler,
    student_t_h0_sampler,
)
from var_backtest_uq.backtest.coverage_tests import kupiec_pof
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N_SIM = 2_000
SEED = 4242
FIXED_DFS = (5.0, 8.0, 15.0)


def empirical_vol_adjusted_sampler(mu, sigma, realized):
    """Bootstrap standardised residuals, re-apply the per-step scale."""
    mu = np.asarray(mu, float).ravel()
    sigma = np.asarray(sigma, float).ravel()
    realized = np.asarray(realized, float).ravel()
    safe = np.where(sigma > 0, sigma, np.nan)
    pool = ((realized - mu) / safe)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        raise ValueError("empty standardised-residual pool")

    def _sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return mu + sigma * rng.choice(pool, size=n, replace=True)

    return _sampler


def build_nulls(mu, sigma, realized, kurt_excess):
    """Return {name: (sampler, df_used)} for one (method, alpha) slice."""
    nulls = {"gaussian": (gaussian_h0_sampler(mu, sigma), None)}

    k_med = float(np.median(kurt_excess)) if kurt_excess is not None else 0.0
    if k_med > 0.1:
        df_shipped = 4.0 + 6.0 / k_med
        nulls["t_moment"] = (student_t_h0_sampler(mu, sigma, df=df_shipped), df_shipped)
    else:
        # Matches stats.py: fall back to Gaussian when kurtosis is negligible.
        nulls["t_moment"] = (gaussian_h0_sampler(mu, sigma), float("nan"))

    for df in FIXED_DFS:
        nulls[f"t_df{int(df)}"] = (student_t_h0_sampler(mu, sigma, df=df), df)

    nulls["empirical"] = (empirical_vol_adjusted_sampler(mu, sigma, realized), None)
    return nulls, k_med


def add_broken_models(d: pd.DataFrame, base: str) -> pd.DataFrame:
    """Append the two known-broken variants of `base` as extra methods."""
    b = d[d["method"] == base]
    if b.empty:
        raise SystemExit(f"base method {base!r} not in details")

    scaled = b.copy()
    for c in ("var", "es", "pnl_mu", "pnl_sigma"):
        scaled[c] = 0.8 * b[c]
    scaled["method"] = "broken_scaled_0.8"

    gm = b.copy()
    alpha = b["alpha"].to_numpy(float)
    z = norm.ppf(alpha)
    mu = b["pnl_mu"].to_numpy(float)
    # Gaussian whose alpha-quantile equals the base VaR: var = mu + s * z.
    s = (b["var"].to_numpy(float) - mu) / z
    gm["pnl_sigma"] = s
    gm["es"] = mu - s * norm.pdf(z) / alpha
    gm["pnl_kurt_excess"] = 0.0
    gm["method"] = "broken_gauss_matched"

    return pd.concat([d, scaled, gm], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--label", default=None, help="Arm label for the output rows.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--broken-from", default="conformal_pid",
                    help="Base method for the two known-broken models; '' to skip.")
    args = ap.parse_args()

    path = Path(args.details)
    label = args.label or path.parent.name
    d = pd.read_csv(path, parse_dates=["date"])

    need = {"pnl_mu", "pnl_sigma", "var", "es", "realized_pnl"}
    missing = need - set(d.columns)
    if missing:
        raise SystemExit(
            f"{path} lacks {sorted(missing)}. Re-run the backtest with the widened "
            "details export (cli.details_long)."
        )

    if args.broken_from:
        d = add_broken_models(d, args.broken_from)

    rows = []
    for (method, alpha), g in d.groupby(["method", "alpha"], sort=True):
        g = g.sort_values("date")
        mu = g["pnl_mu"].to_numpy(float)
        sigma = g["pnl_sigma"].to_numpy(float)
        realized = g["realized_pnl"].to_numpy(float)
        var = g["var"].to_numpy(float)
        es = g["es"].to_numpy(float)
        if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
            print(f"  skip {method} a={alpha}: non-finite predicted moments")
            continue
        kurt = (
            g["pnl_kurt_excess"].to_numpy(float)
            if "pnl_kurt_excess" in g.columns else None
        )
        nulls, k_med = build_nulls(mu, sigma, realized, kurt)
        breaches = (realized < var).astype(int)
        kup = kupiec_pof(breaches, alpha=float(alpha), n_simulations=5_000, seed=SEED)

        for name, (sampler, df_used) in nulls.items():
            # Same seed for both: the two statistics are scored on identical
            # simulated paths.
            r1 = acerbi_szekely_test_1(
                realized, var, es,
                sample_under_h0=sampler, n_simulations=N_SIM, seed=SEED,
            )
            r2 = acerbi_szekely_test_2(
                realized, var, es, alpha=float(alpha),
                sample_under_h0=sampler, n_simulations=N_SIM, seed=SEED,
            )
            row = {
                "arm": label,
                "method": method,
                "alpha": float(alpha),
                "null": name,
                "df_used": df_used,
                "kurt_median": k_med,
                "n_breaches": r1["n_breaches"],
                "kupiec_pvalue": kup["p_value"],
            }
            for tag, r in (("z1", r1), ("z2", r2)):
                row.update({
                    tag: r["statistic"],
                    f"{tag}_zone": r["zone"],
                    f"{tag}_p_value": r["p_value"],
                    f"{tag}_green_threshold": r["green_threshold"],
                    f"{tag}_yellow_threshold": r["yellow_threshold"],
                })
            rows.append(row)
        print(f"  done {method} a={alpha} (kappa_med={k_med:.3f})")

    out = pd.DataFrame(rows)
    dest = Path(args.out) if args.out else path.parent / "as_null_sensitivity.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {len(out)} rows -> {dest}")

    for tag in ("z1", "z2"):
        piv = out.pivot_table(
            index=["method", "alpha"], columns="null", values=f"{tag}_zone", aggfunc="first",
        )
        print(f"\n{tag.upper()} zones by null specification:\n")
        print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

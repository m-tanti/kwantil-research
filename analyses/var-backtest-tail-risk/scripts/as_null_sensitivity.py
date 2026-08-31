"""Acerbi-Szekely Z2 sensitivity to the null specification.

The AS zone on the dashboard is computed against a null generated from each
method's own predicted marginal. That is the construction A&S intend for a
specification test, but this implementation does not simulate from the model's
*exact* predictive: it draws from a moment-matched Student-t whose df is a
single scalar from the median predicted excess kurtosis. Manual section 8 says
the sign of the resulting bias is unmeasured. This script measures it.

Five nulls per (method, alpha):

  gaussian      N(mu_t, sigma_t).                       Thin reference.
  t_moment      Student-t, df = 4 + 6/kappa_median.     The shipped v1.1 null.
  t_df5/8/15    Student-t at fixed df.                  Brackets the df choice.
  empirical     mu_t + sigma_t * bootstrap(z_hat).      Distribution-free shape.

The empirical null resamples standardised residuals z_hat = (x_t - mu_t)/sigma_t
and re-applies the per-step scale. The library's
`empirical_h0_sampler_from_residuals` resamples raw values iid, which would
destroy the volatility structure and is not usable here.

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

from var_backtest_uq.backtest.acerbi_szekely import (
    acerbi_szekely_test_2,
    gaussian_h0_sampler,
    student_t_h0_sampler,
)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--label", default=None, help="Arm label for the output rows.")
    ap.add_argument("--out", default=None)
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

        for name, (sampler, df_used) in nulls.items():
            r = acerbi_szekely_test_2(
                realized, var, es,
                sample_under_h0=sampler, n_simulations=N_SIM, seed=SEED,
            )
            rows.append({
                "arm": label,
                "method": method,
                "alpha": float(alpha),
                "null": name,
                "df_used": df_used,
                "kurt_median": k_med,
                "z2": r["statistic"],
                "zone": r["zone"],
                "p_value": r["p_value"],
                "green_threshold": r["green_threshold"],
                "yellow_threshold": r["yellow_threshold"],
                "n_breaches": r["n_breaches"],
            })
        print(f"  done {method} a={alpha} (kappa_med={k_med:.3f})")

    out = pd.DataFrame(rows)
    dest = Path(args.out) if args.out else path.parent / "as_null_sensitivity.csv"
    out.to_csv(dest, index=False)
    print(f"\nWrote {len(out)} rows -> {dest}")

    piv = out.pivot_table(
        index=["method", "alpha"], columns="null", values="zone", aggfunc="first",
    )
    print("\nZones by null specification:\n")
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

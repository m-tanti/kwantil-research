"""Per-(method, alpha) summary stats: Kupiec/Christoffersen + Basel TL + AS Z1/Z2 (A&S 2014 naming)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .coverage_tests import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
)

from .acerbi_szekely import (
    acerbi_szekely_test_1,
    acerbi_szekely_test_2,
    gaussian_h0_sampler,
    student_t_h0_sampler,
)
from .basel_tl import basel_traffic_light, ima_capital_path
from .runner import BacktestResult


def compute_stats(
    result: BacktestResult,
    basel_window: int = 250,
    *,
    coverage_n_simulations: int = 5_000,
    es_n_simulations: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One stats row per (method, alpha)."""
    rows: list[dict] = []
    fc = result.forecasts.set_index(["method", "alpha", "date"]).sort_index()
    br = result.breaches.set_index(["method", "alpha", "date"]).sort_index()
    realized = result.realized

    for (method, alpha), group_fc in fc.groupby(level=["method", "alpha"]):
        group_br = br.xs((method, alpha), level=["method", "alpha"])
        dates = group_fc.index.get_level_values("date")
        var_arr = group_fc["var"].to_numpy()
        es_arr = group_fc["es"].to_numpy()
        breach_arr = group_br["breached"].astype(int).to_numpy()
        realized_arr = realized.reindex(dates).to_numpy()

        pof = kupiec_pof(breach_arr, alpha=alpha,
                         n_simulations=coverage_n_simulations, seed=seed)
        ind = christoffersen_independence(breach_arr,
                                          n_simulations=coverage_n_simulations, seed=seed + 1)
        cc = christoffersen_conditional_coverage(breach_arr, alpha=alpha,
                                                 n_simulations=coverage_n_simulations, seed=seed + 2)
        basel = basel_traffic_light(breach_arr, window=basel_window)

        # AS bootstrap H0. When kurtosis info is available, use a moment-
        # matched Student-t so the H0 carries the same tail thickness as the
        # predictor; otherwise fall back to Gaussian.
        if {"pnl_mu", "pnl_sigma"}.issubset(group_fc.columns):
            mu = group_fc["pnl_mu"].to_numpy()
            sigma = group_fc["pnl_sigma"].to_numpy()
            if "pnl_kurt_excess" in group_fc.columns:
                # Single shared df from the median; per-date df would over-fit MC noise.
                k_med = float(np.median(group_fc["pnl_kurt_excess"].to_numpy()))
                if k_med > 0.1:
                    sampler = student_t_h0_sampler(mu, sigma, df=4.0 + 6.0 / k_med)
                else:
                    sampler = gaussian_h0_sampler(mu, sigma)
            else:
                sampler = gaussian_h0_sampler(mu, sigma)
        else:
            sampler = None

        # Seeds are tied to the statistic, not the name, so outputs from
        # releases that had Z1/Z2 swapped reproduce under the corrected labels.
        as_z1 = acerbi_szekely_test_1(
            realized_arr, var_arr, es_arr,
            sample_under_h0=sampler, n_simulations=es_n_simulations, seed=seed + 4,
        )
        as_z2 = acerbi_szekely_test_2(
            realized_arr, var_arr, es_arr, alpha=alpha,
            sample_under_h0=sampler, n_simulations=es_n_simulations, seed=seed + 3,
        )

        ima_path = ima_capital_path(var_arr, breach_arr, window=basel_window)
        mean_capital = float(ima_path["ima_capital"].mean()) if len(ima_path) else float("nan")
        median_capital = float(ima_path["ima_capital"].median()) if len(ima_path) else float("nan")
        peak_capital = float(ima_path["ima_capital"].max()) if len(ima_path) else float("nan")
        mean_var_abs = float(ima_path["var_abs"].mean()) if len(ima_path) else float("nan")

        rows.append({
            "method": method,
            "alpha": float(alpha),
            "n_obs": pof["n_obs"],
            "n_breaches": pof["n_breaches"],
            "observed_rate": pof["observed_rate"],
            "observed_rate_ci_lo": pof.get("observed_rate_ci_lo", float("nan")),
            "observed_rate_ci_hi": pof.get("observed_rate_ci_hi", float("nan")),
            "target_rate": pof["target_rate"],
            "kupiec_lr": pof["statistic"],
            "kupiec_pvalue": pof["p_value"],
            "kupiec_pvalue_method": pof["p_value_method"],
            "christoffersen_ind_lr": ind["statistic"],
            "christoffersen_ind_pvalue": ind["p_value"],
            "christoffersen_cc_lr": cc["statistic"],
            "christoffersen_cc_pvalue": cc["p_value"],
            "basel_zone": basel["zone"],
            "basel_n_breaches_window": basel["n_breaches"],
            "basel_addon": basel["addon"],
            "basel_total_multiplier": basel["total_multiplier"],
            "basel_window_used": basel["window_used"],
            "as_z1_statistic": as_z1["statistic"],
            "as_z1_zone": as_z1.get("zone"),
            "as_z1_pvalue": as_z1["p_value"],
            "as_z2_statistic": as_z2["statistic"],
            "as_z2_zone": as_z2.get("zone"),
            "as_z2_pvalue": as_z2["p_value"],
            "as_n_breaches": as_z1["n_breaches"],
            "mean_var_abs": mean_var_abs,
            "mean_ima_capital": mean_capital,
            "median_ima_capital": median_capital,
            "peak_ima_capital": peak_capital,
        })

    return pd.DataFrame(rows).sort_values(["method", "alpha"]).reset_index(drop=True)


def ima_capital_series(result: BacktestResult, basel_window: int = 250) -> pd.DataFrame:
    """Per-step IMA capital path per (method, alpha)."""
    fc = result.forecasts.set_index(["method", "alpha", "date"]).sort_index()
    br = result.breaches.set_index(["method", "alpha", "date"]).sort_index()
    out_rows: list[pd.DataFrame] = []
    for (method, alpha), g in fc.groupby(level=["method", "alpha"]):
        gbr = br.xs((method, alpha), level=["method", "alpha"])
        var_arr = g["var"].to_numpy()
        breach_arr = gbr["breached"].astype(int).to_numpy()
        path = ima_capital_path(var_arr, breach_arr, window=basel_window)
        path["date"] = g.index.get_level_values("date").values
        path["method"] = method
        path["alpha"] = float(alpha)
        out_rows.append(path)
    if not out_rows:
        return pd.DataFrame(columns=[
            "date", "method", "alpha", "var_abs", "var_60d_mean",
            "multiplier", "zone", "ima_capital",
        ])
    return pd.concat(out_rows, ignore_index=True)


def pairwise_capital_impact(comparison: pd.DataFrame) -> pd.DataFrame:
    """All-pairs (method_a, method_b) capital change at each alpha (negative = method_a saves)."""
    rows: list[dict] = []
    for alpha, g in comparison.groupby("alpha"):
        for _, ra in g.iterrows():
            for _, rb in g.iterrows():
                if ra["method"] == rb["method"]:
                    continue
                rows.append({
                    "alpha": float(alpha),
                    "adopt_method": ra["method"],
                    "replace_method": rb["method"],
                    "mean_ima_capital_change": float(ra["mean_ima_capital"] - rb["mean_ima_capital"]),
                    "mean_ima_capital_change_pct": float(
                        (ra["mean_ima_capital"] - rb["mean_ima_capital"]) / rb["mean_ima_capital"]
                    ) if rb["mean_ima_capital"] else float("nan"),
                    "multiplier_change": float(ra["basel_total_multiplier"] - rb["basel_total_multiplier"]),
                    "kupiec_p_change": float(ra["kupiec_pvalue"] - rb["kupiec_pvalue"]),
                })
    return pd.DataFrame(rows)

"""Dashboard JSON / Parquet writer. Parquet is canonical; JSON shadows let SvelteKit skip an Arrow.js bundle."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.runner import BacktestResult
from ..backtest.stats import compute_stats, ima_capital_series, pairwise_capital_impact
from ..data.stress_windows import STRESS_WINDOWS, tag_dates


_BASEL_REGULATORY_ALPHA = 0.01
_ROLLING_WINDOW_DAYS = 250


def _json_default(obj):
    # Non-finite floats are pre-converted to None by _sanitize_for_json,
    # so this only sees finite numerics + tz-aware timestamps + ndarrays.
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"unserialisable: {type(obj).__name__}")


def _sanitize_for_json(obj):
    # Replace NaN/Infinity with None up front; browsers' JSON.parse rejects
    # the literals Python's json.dump would otherwise emit.
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _write_table(df: pd.DataFrame, base: Path) -> None:
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    records = _sanitize_for_json(df.to_dict(orient="records"))
    with base.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(records, f, default=_json_default, indent=None)


def coverage_timeseries(result: BacktestResult, window: int = _ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Rolling breach rate per (method, alpha)."""
    br = result.breaches.copy()
    br["breached"] = br["breached"].astype(int)
    out = []
    for (method, alpha), g in br.groupby(["method", "alpha"]):
        g = g.sort_values("date").reset_index(drop=True)
        rolling = g["breached"].rolling(window=window, min_periods=window).mean()
        sub = pd.DataFrame({
            "date": g["date"],
            "method": method,
            "alpha": float(alpha),
            "rolling_breach_rate": rolling,
        })
        out.append(sub)
    df = pd.concat(out, ignore_index=True).dropna(subset=["rolling_breach_rate"])
    if len(df):
        tags = tag_dates(pd.DatetimeIndex(df["date"].unique()))
        df = df.merge(
            pd.DataFrame({"date": tags.index, "stress_window": tags.values}),
            on="date", how="left",
        )
    else:
        df["stress_window"] = ""
    return df


def per_asset_coverage(result: BacktestResult) -> pd.DataFrame:
    """Per-(method, alpha, asset) breach counts and rates."""
    if result.per_asset.empty:
        return pd.DataFrame(
            columns=["method", "alpha", "asset", "n_obs", "n_breaches",
                     "observed_rate", "target_rate"]
        )
    g = (
        result.per_asset
        .groupby(["method", "alpha", "asset"])
        .agg(n_obs=("breached", "size"), n_breaches=("breached", "sum"))
        .reset_index()
    )
    g["observed_rate"] = g["n_breaches"] / g["n_obs"]
    g["target_rate"] = g["alpha"]
    return g


def _days_since_last_breach(
    group: pd.DataFrame,
    as_of: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
) -> int | None:
    """Trading days (not calendar days) since the most recent breach."""
    breached = group[group["breached"]]
    if breached.empty:
        return None
    last_breach = pd.Timestamp(breached["date"].max())
    last_pos = trading_days.searchsorted(last_breach)
    as_of_pos = trading_days.searchsorted(as_of)
    return int(max(0, as_of_pos - last_pos))


def _calibration_stability(coverage_ts: pd.DataFrame) -> dict[tuple[str, float], float]:
    """Std of the rolling breach rate per (method, alpha)."""
    if coverage_ts.empty:
        return {}
    out: dict[tuple[str, float], float] = {}
    for (method, alpha), g in coverage_ts.groupby(["method", "alpha"]):
        out[(method, float(alpha))] = float(g["rolling_breach_rate"].std())
    return out


def build_summary(
    result: BacktestResult,
    comparison: pd.DataFrame,
    coverage_ts: pd.DataFrame | None = None,
) -> dict:
    if result.realized.empty:
        return {"as_of": None, "methods": {}}
    as_of = pd.Timestamp(result.realized.index[-1])
    trading_days = pd.DatetimeIndex(result.realized.index)
    breaches = result.breaches.copy()
    breaches["date"] = pd.to_datetime(breaches["date"])

    stability = _calibration_stability(coverage_ts) if coverage_ts is not None else {}

    methods_payload: dict[str, dict] = {}
    for method in sorted(comparison["method"].unique()):
        per_alpha: dict[str, dict] = {}
        for _, row in comparison[comparison["method"] == method].iterrows():
            alpha = float(row["alpha"])
            sub = breaches[(breaches["method"] == method) & (breaches["alpha"] == alpha)]
            dsb = _days_since_last_breach(sub, as_of, trading_days)
            per_alpha[f"{alpha}"] = {
                "alpha": alpha,
                "n_obs": int(row["n_obs"]),
                "n_breaches": int(row["n_breaches"]),
                "observed_rate": float(row["observed_rate"]),
                "observed_rate_ci_lo": float(row.get("observed_rate_ci_lo", float("nan"))),
                "observed_rate_ci_hi": float(row.get("observed_rate_ci_hi", float("nan"))),
                "target_rate": float(row["target_rate"]),
                "kupiec_pvalue": float(row["kupiec_pvalue"]),
                "kupiec_pvalue_method": str(row.get("kupiec_pvalue_method", "asymptotic")),
                "christoffersen_cc_pvalue": float(row["christoffersen_cc_pvalue"]),
                "basel_zone": str(row["basel_zone"]),
                "basel_n_breaches_window": int(row["basel_n_breaches_window"]),
                "basel_total_multiplier": float(row["basel_total_multiplier"]),
                "as_z1_zone": row.get("as_z1_zone"),
                "as_z1_pvalue": float(row.get("as_z1_pvalue", float("nan"))),
                "as_z2_zone": row.get("as_z2_zone"),
                "as_z2_pvalue": float(row.get("as_z2_pvalue", float("nan"))),
                # Pre-v0.2 dashboard aliases; drop once types.ts is updated.
                "es_zone": row.get("as_z2_zone"),
                "es_mean_shortfall_ratio": float(row.get("as_z2_statistic", float("nan"))),
                "mean_var_abs": float(row.get("mean_var_abs", float("nan"))),
                "mean_ima_capital": float(row.get("mean_ima_capital", float("nan"))),
                "median_ima_capital": float(row.get("median_ima_capital", float("nan"))),
                "peak_ima_capital": float(row.get("peak_ima_capital", float("nan"))),
                "calibration_stability": stability.get((method, alpha), float("nan")),
                "days_since_last_breach": dsb,
            }
        methods_payload[method] = per_alpha

    pairwise = pairwise_capital_impact(comparison)
    pair_at_reg = pairwise[pairwise["alpha"] == _BASEL_REGULATORY_ALPHA]
    pair_payload = pair_at_reg.to_dict(orient="records") if not pair_at_reg.empty else []

    # Headline pair: prefer joint VaR+ES (addresses residual tail-shape too)
    # over bare conformal_pid; both compare against distributional (the base).
    headline_pair = None
    if not pair_at_reg.empty:
        for adopt in ("conformal_pid_es", "conformal_pid"):
            candidate = pair_at_reg[
                (pair_at_reg["adopt_method"] == adopt)
                & (pair_at_reg["replace_method"] == "distributional")
            ]
            if not candidate.empty:
                r = candidate.iloc[0]
                headline_pair = {
                    "adopt_method": adopt,
                    "replace_method": "distributional",
                    "alpha": _BASEL_REGULATORY_ALPHA,
                    "mean_ima_capital_change": float(r["mean_ima_capital_change"]),
                    "mean_ima_capital_change_pct": float(r["mean_ima_capital_change_pct"]),
                    "multiplier_change": float(r["multiplier_change"]),
                    "kupiec_p_change": float(r["kupiec_p_change"]),
                }
                break

    reg = comparison[comparison["alpha"] == _BASEL_REGULATORY_ALPHA].copy()
    best_kupiec_method = None
    if not reg.empty:
        best_kupiec_method = str(reg.sort_values("kupiec_pvalue", ascending=False).iloc[0]["method"])

    return {
        "as_of": as_of.isoformat(),
        "alpha_targets": sorted(comparison["alpha"].unique().tolist()),
        "n_methods": int(comparison["method"].nunique()),
        "n_steps": int(len(result.realized)),
        "methods": methods_payload,
        "regulatory_alpha": _BASEL_REGULATORY_ALPHA,
        "best_kupiec_method": best_kupiec_method,
        "headline_capital_impact": headline_pair,
        "pairwise_capital_impact_alpha_001": pair_payload,
        "stress_windows": [
            {"label": label, "start": start, "end": end}
            for label, start, end in STRESS_WINDOWS
        ],
    }


def write_dashboard_payload(result: BacktestResult, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = compute_stats(result)
    cov_ts = coverage_timeseries(result)
    per_asset = per_asset_coverage(result)
    summary = build_summary(result, comparison, coverage_ts=cov_ts)
    capital_path = ima_capital_series(result)
    pairwise = pairwise_capital_impact(comparison)

    written: dict[str, Path] = {}

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(summary), f, default=_json_default, indent=2)
    written["summary"] = summary_path

    # Pre-v0.2 dashboard aliases; drop once types.ts is updated.
    comparison_with_aliases = comparison.assign(
        es_zone=comparison["as_z2_zone"],
        es_mean_shortfall_ratio=comparison["as_z2_statistic"],
        es_n_exceedances=comparison["as_n_breaches"],
    )
    _write_table(comparison_with_aliases, out_dir / "methods_comparison")
    written["methods_comparison"] = out_dir / "methods_comparison.parquet"

    _write_table(cov_ts, out_dir / "coverage_timeseries")
    written["coverage_timeseries"] = out_dir / "coverage_timeseries.parquet"

    _write_table(per_asset, out_dir / "per_asset")
    written["per_asset"] = out_dir / "per_asset.parquet"

    if not capital_path.empty:
        _write_table(capital_path, out_dir / "ima_capital_path")
        written["ima_capital_path"] = out_dir / "ima_capital_path.parquet"

    if not pairwise.empty:
        _write_table(pairwise, out_dir / "pairwise_capital")
        written["pairwise_capital"] = out_dir / "pairwise_capital.parquet"

    if not result.controller_state.empty:
        _write_table(result.controller_state, out_dir / "controller_trace")
        written["controller_trace"] = out_dir / "controller_trace.parquet"

    return written

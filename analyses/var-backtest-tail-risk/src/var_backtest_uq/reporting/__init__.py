"""Reporting outputs: dashboard JSON / Parquet snapshots and regulatory PDFs.

- `snapshot`, `write_dashboard_payload(result, out_dir)` emits the JSON +
  Parquet files the SvelteKit dashboard reads (see the `dashboard/` sibling
  directory in the var-backtest monorepo).
- `pdf`, `compile_report(result, out_path)` produces a model-validation
  PDF via Typst, with VaR backtesting under CRR Article 366 and a list of
  supervisory references (ECB, PRA SS1/23, OCC 2026-13, CESR/10-788).
"""

from .pdf import compile_report
from .snapshot import (
    build_summary,
    coverage_timeseries,
    per_asset_coverage,
    write_dashboard_payload,
)

__all__ = [
    "write_dashboard_payload",
    "build_summary",
    "coverage_timeseries",
    "per_asset_coverage",
    "compile_report",
]

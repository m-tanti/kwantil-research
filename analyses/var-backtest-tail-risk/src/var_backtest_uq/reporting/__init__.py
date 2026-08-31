"""Reporting outputs: dashboard JSON / Parquet snapshots and regulatory PDFs.

- `snapshot`, `write_dashboard_payload(result, out_dir)` emits the JSON +
  Parquet files the SvelteKit dashboard reads (see the `dashboard/` sibling
  directory in the var-backtest monorepo).
- `pdf`, `compile_report(result, out_path)` produces a regulatory PDF
  mapped section-by-section to OCC 2026-13 lifecycle and FRTB IMA
  backtesting headings via Typst.
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

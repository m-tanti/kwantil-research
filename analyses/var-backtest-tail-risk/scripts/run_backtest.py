"""CLI shim for the var-backtest pipeline.

Parses argv into the keyword arguments that `var_backtest_uq.cli.run_backtest_pipeline`
expects, configures `INFO`-level logging, and prints a tail-summary of the
per-(method, alpha) statistics.

Usage:
    python scripts/run_backtest.py --method all
    python scripts/run_backtest.py --method parametric,historical
    python scripts/run_backtest.py --method distributional --refit-every 20

For library use, prefer the function directly:

    from var_backtest_uq.cli import run_backtest_pipeline
    result = run_backtest_pipeline(method="all", output_dir="/tmp/out")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from var_backtest_uq.cli import (
    METHOD_FACTORIES,
    SUMMARY_PRINT_COLUMNS,
    run_backtest_pipeline,
)
import sys



# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the var-backtest VaR/ES walk-forward pipeline.")
    p.add_argument("--method", default="all",
                   help='"all", a single method, or comma-separated list. '
                        f'Known: {list(METHOD_FACTORIES)}')
    p.add_argument("--tickers", default="SPY,TLT,GLD")
    p.add_argument("--weights", default="0.3333,0.3333,0.3334")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--fit-window", type=int, default=500)
    p.add_argument("--alpha-targets", default="0.025,0.01")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "cache"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "output"))
    p.add_argument("--dashboard-dir", default=str(REPO_ROOT / "output" / "dashboard"))
    p.add_argument("--no-dashboard", action="store_true",
                   help="Skip dashboard payload (parquet/json artefacts).")
    p.add_argument("--pdf", action="store_true",
                   help="Emit OCC 2026-13 / FRTB IMA validation PDF (requires typst).")
    p.add_argument("--pdf-path", default=None,
                   help="Output path for the PDF (default: <output-dir>/validation_report.pdf).")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--refit-every", type=int, default=20,
                   help="GARCH refit cadence (distributional only).")
    p.add_argument("--copula", default="t", choices=["t", "gaussian"],
                   help="Copula for the distributional base. 'gaussian' reproduces the "
                        "pre-v1.1 arm for the de-confounding factorial.")
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="Out-of-sample controller warmup steps before reporting begins.")
    p.add_argument("--log-level", default="INFO",
                   help='Logging level (default INFO). One of DEBUG, INFO, WARNING, ERROR.')
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    alpha_targets = tuple(float(a) for a in args.alpha_targets.split(","))

    result = run_backtest_pipeline(
        method=args.method,
        tickers=tickers,
        weights=weights,
        start=args.start,
        end=args.end,
        fit_window=args.fit_window,
        alpha_targets=alpha_targets,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dashboard_dir=args.dashboard_dir,
        write_dashboard=not args.no_dashboard,
        write_pdf=args.pdf,
        pdf_path=args.pdf_path,
        refresh=args.refresh,
        refit_every=args.refit_every,
        copula=args.copula,
        warmup_steps=args.warmup_steps,
    )

    print()
    print("Summary:")
    print(result.summary_df[list(SUMMARY_PRINT_COLUMNS)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

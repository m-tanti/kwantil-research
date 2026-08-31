"""Backtest orchestration, regulatory tests, and statistical summaries.

- `runner`, `BacktestRunner`: walk-forward orchestrator producing a
  `BacktestResult` from prices + methods + portfolio weights.
- `stats`, `compute_stats(result)`: per-(method, alpha) DataFrame with all
  regulatory backtests; `pairwise_capital_impact`, `ima_capital_series`.
- `coverage_tests`, Kupiec POF, Christoffersen independence + conditional
  coverage tests with Monte-Carlo or asymptotic p-values.
- `acerbi_szekely`, Z1 / Z2 ES backtests with Gaussian or Student-t H0
  bootstrap samplers.
- `basel_tl`, CRR Article 366 traffic-light + IMA capital path.
"""

from .acerbi_szekely import (
    AcerbiSzekelyResult,
    acerbi_szekely_test_1,
    acerbi_szekely_test_2,
    gaussian_h0_sampler,
    student_t_h0_sampler,
)
from .basel_tl import (
    BASE_MULTIPLIER,
    BaselTrafficLightResult,
    basel_traffic_light,
    ima_capital_path,
    rolling_basel_multiplier,
)
from .coverage_tests import (
    ChristoffersenCCResult,
    ChristoffersenIndependenceResult,
    KupiecResult,
    PValueMethod,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
    wilson_score_interval,
)
from .runner import BacktestResult, BacktestRunner
from .stats import compute_stats, ima_capital_series, pairwise_capital_impact

__all__ = [
    # Orchestration
    "BacktestRunner",
    "BacktestResult",
    "compute_stats",
    "pairwise_capital_impact",
    "ima_capital_series",
    # Coverage tests (Kupiec / Christoffersen)
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_conditional_coverage",
    "wilson_score_interval",
    "PValueMethod",
    "KupiecResult",
    "ChristoffersenIndependenceResult",
    "ChristoffersenCCResult",
    # ES tests (Acerbi-Szekely)
    "acerbi_szekely_test_1",
    "acerbi_szekely_test_2",
    "gaussian_h0_sampler",
    "student_t_h0_sampler",
    "AcerbiSzekelyResult",
    # Basel TL
    "basel_traffic_light",
    "rolling_basel_multiplier",
    "ima_capital_path",
    "BASE_MULTIPLIER",
    "BaselTrafficLightResult",
]

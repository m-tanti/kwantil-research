"""Walk-forward VaR/ES backtester with Conformal-PID calibration."""

from .backtest.runner import BacktestResult, BacktestRunner
from .backtest.stats import compute_stats, ima_capital_series, pairwise_capital_impact
from .conformal.calibrator import (
    CalibratedDistribution,
    CalibratedPortfolioForecast,
    ConformalCalibrator,
)
from .conformal.controllers import (
    ConformalPIDController,
    OneSidedQuantileDistribution,
)
from .conformal.joint_calibrator import (
    EsCorrector,
    JointCalibratedDistribution,
    JointCalibratedPortfolioForecast,
    JointVarEsCalibrator,
)
from .forecasters.distributional import DistributionalForecaster
from .forecasters.historical import HistoricalSimulationForecaster
from .forecasters.parametric import ParametricGaussianForecaster
from .reporting.snapshot import write_dashboard_payload

__version__ = "0.1.0"

__all__ = [
    "BacktestRunner",
    "BacktestResult",
    "compute_stats",
    "ima_capital_series",
    "pairwise_capital_impact",
    "ParametricGaussianForecaster",
    "HistoricalSimulationForecaster",
    "DistributionalForecaster",
    "ConformalCalibrator",
    "CalibratedDistribution",
    "CalibratedPortfolioForecast",
    "JointVarEsCalibrator",
    "JointCalibratedDistribution",
    "JointCalibratedPortfolioForecast",
    "EsCorrector",
    "ConformalPIDController",
    "OneSidedQuantileDistribution",
    "write_dashboard_payload",
    "__version__",
]

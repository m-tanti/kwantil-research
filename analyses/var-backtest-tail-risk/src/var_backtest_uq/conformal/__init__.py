"""Conformal calibration: typed protocols, controllers, and calibrators.

- `protocols`, `ConformalMethod` and `PredictiveDistribution` Protocol classes.
- `controllers`, `ConformalPIDController` + the
  `OneSidedQuantileDistribution` it returns. Implementation of Angelopoulos,
  Candès & Tibshirani (2024) with vol-EMA + MDN width-scaling extensions.
- `calibrator`, `ConformalCalibrator`: composes a base `Forecaster` with
  per-alpha + per-asset `ConformalMethod` controllers; returns a
  `CalibratedPortfolioForecast` that exposes calibrated quantiles and ES.
- `joint_calibrator`, `JointVarEsCalibrator`: same VaR pipeline plus a
  per-alpha `EsCorrector` for joint VaR-ES feedback calibration.
"""

from .calibrator import (
    CalibratedDistribution,
    CalibratedPortfolioForecast,
    ConformalCalibrator,
)
from .controllers import ConformalPIDController, OneSidedQuantileDistribution
from .joint_calibrator import (
    EsCorrector,
    JointCalibratedDistribution,
    JointCalibratedPortfolioForecast,
    JointVarEsCalibrator,
)
from .protocols import ConformalMethod, PredictiveDistribution

__all__ = [
    "ConformalMethod",
    "PredictiveDistribution",
    "ConformalPIDController",
    "OneSidedQuantileDistribution",
    "ConformalCalibrator",
    "CalibratedPortfolioForecast",
    "CalibratedDistribution",
    "JointVarEsCalibrator",
    "JointCalibratedPortfolioForecast",
    "JointCalibratedDistribution",
    "EsCorrector",
]

"""Tests for the Basel III TL helpers + rolling/IMA capital path."""

import numpy as np
import pandas as pd

from var_backtest_uq.backtest.basel_tl import (
    BASE_MULTIPLIER,
    basel_traffic_light,
    ima_capital_path,
    rolling_basel_multiplier,
)


def test_static_basel_zone_boundaries():
    breaches = np.zeros(250, dtype=int)
    assert basel_traffic_light(breaches)["zone"] == "green"

    breaches[:4] = 1
    assert basel_traffic_light(breaches)["zone"] == "green"

    breaches[:5] = 1
    res5 = basel_traffic_light(breaches)
    assert res5["zone"] == "yellow"
    assert res5["addon"] == 0.40

    breaches = np.zeros(250, dtype=int); breaches[:10] = 1
    res10 = basel_traffic_light(breaches)
    assert res10["zone"] == "red"
    assert res10["addon"] == 1.00


def test_rolling_basel_per_step_matches_static_at_window_end():
    breaches = np.zeros(300, dtype=int)
    breaches[200:208] = 1  # 8 breaches → yellow at window end
    rolling = rolling_basel_multiplier(breaches, window=250)
    assert len(rolling) == 300
    # Final row should match static call's outcome.
    last = rolling.iloc[-1]
    static = basel_traffic_light(breaches, window=250)
    assert last["n_breaches_window"] == static["n_breaches"]
    assert last["zone"] == static["zone"]


def test_ima_capital_path_uses_max_of_var_and_60d_mean():
    # Synthetic: VaR magnitude shrinks then grows; 60d mean lags.
    n = 80
    var_pred = np.full(n, -0.02)
    var_pred[40:] = -0.04
    breaches = np.zeros(n, dtype=int)
    path = ima_capital_path(var_pred, breaches, var_60d_window=20)
    # First half: var_abs=0.02, var_60d_mean=0.02; ima = 3.0 * 0.02 = 0.06
    assert path["ima_capital"].iloc[0] == 0.06
    # After step 40, var_abs=0.04 > var_60d_mean (warming up)
    assert path["ima_capital"].iloc[60] >= 0.04 * BASE_MULTIPLIER


def test_ima_capital_path_breach_penalty_kicks_in():
    n = 260
    var_pred = np.full(n, -0.02)
    breaches = np.zeros(n, dtype=int)
    # Inject 5 breaches right before step 250 → trailing window has 5 → yellow → +0.40
    breaches[245:250] = 1
    path = ima_capital_path(var_pred, breaches, window=250, var_60d_window=20)
    # Step 250 should be in yellow zone with multiplier 3.40
    assert path["zone"].iloc[250] == "yellow"
    # Capital scaled accordingly
    assert path["ima_capital"].iloc[250] == 3.40 * 0.02

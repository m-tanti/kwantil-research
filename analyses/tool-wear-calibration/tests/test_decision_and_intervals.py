"""The pieces of the tool-wear audit that carry a published number.

The article's decision curve rests on outcomes(): how much tool life a change
rule throws away, and how many cuts it runs past the wear limit. Its coverage
claims rest on the finite-sample conformal quantile. Both are small enough to
check exactly, which is the only reason to test them rather than the pipeline
around them.
"""

import numpy as np
import pandas as pd
import pytest

from ablation import wilson
from calibrate import conformal_q
from decision import outcomes, first_cut_where


def track(vb, start=1):
    """One insert's measured life: cut numbers against flank wear."""
    return pd.DataFrame({"cut_index": range(start, start + len(vb)), "VB": vb})


# ------------------------------------------------------------------ outcomes
def test_changing_early_wastes_the_cuts_between_change_and_limit():
    # Limit reached at cut 8; pulled at cut 5; three cuts of life discarded and
    # nothing run worn.
    t = track([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.62, 0.7])
    assert outcomes(t, change_at=5, limit=0.60) == (3, 0)


def test_changing_late_runs_cuts_past_the_limit():
    t = track([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.62, 0.7])
    # Limit first exceeded at cut 8; pulled at 10; two cuts run worn, no waste.
    assert outcomes(t, change_at=10, limit=0.60) == (0, 2)


def test_changing_exactly_at_the_limit_is_free():
    t = track([0.1, 0.3, 0.5, 0.62, 0.7])
    assert outcomes(t, change_at=4, limit=0.60) == (0, 0)


def test_never_changing_counts_every_cut_past_the_limit():
    t = track([0.1, 0.3, 0.5, 0.62, 0.7, 0.8])
    # First over at cut 4, record ends at cut 6: two cuts run worn.
    assert outcomes(t, change_at=None, limit=0.60) == (0, 2)


def test_never_changing_an_insert_that_never_wore_out_costs_nothing():
    t = track([0.1, 0.2, 0.3])
    assert outcomes(t, change_at=None, limit=0.60) == (0, 0)


def test_pulling_an_insert_that_never_reached_the_limit_is_pure_waste():
    t = track([0.1, 0.2, 0.3, 0.35, 0.4])
    # Pulled at cut 2 with three cuts of usable life left in it.
    assert outcomes(t, change_at=2, limit=0.60) == (3, 0)


def test_outcomes_are_counted_in_physical_cuts_not_row_positions():
    # The docstring's promise, and the bug the article had to fix: an insert
    # whose measured rows are sparse must still be scored on cut numbers.
    sparse = pd.DataFrame({"cut_index": [1, 4, 9, 14], "VB": [0.1, 0.3, 0.55, 0.7]})
    waste, late = outcomes(sparse, change_at=4, limit=0.60)
    assert (waste, late) == (10, 0)   # 14 - 4, not 3 - 1


def test_first_cut_where_returns_a_cut_number_or_none():
    t = track([0.1, 0.5, 0.7], start=5)
    assert first_cut_where(t, t["VB"] >= 0.6) == 7
    assert first_cut_where(t, t["VB"] >= 9.9) is None


# --------------------------------------------------------- conformal quantile
def test_conformal_quantile_is_the_finite_sample_one_not_the_empirical_one():
    # n = 9, alpha = 0.1: ceil((n+1)(1-alpha))/n = ceil(9)/9 = 1.0, so the
    # calibrated radius is the LARGEST score. The plain empirical 90th
    # percentile would be smaller, and would undercover.
    scores = np.arange(1.0, 10.0)
    assert conformal_q(scores, alpha=0.1) == pytest.approx(9.0)
    assert np.quantile(scores, 0.9) < 9.0


def test_conformal_quantile_is_taken_from_above():
    # "higher" interpolation: the returned value is always an observed score,
    # never one interpolated between two of them.
    scores = np.array([1.0, 2.0, 10.0, 11.0])
    assert conformal_q(scores, alpha=0.5) in set(scores)


def test_conformal_quantile_is_infinite_when_there_is_nothing_to_calibrate_on():
    # An empty calibration set cannot promise coverage, and the honest radius is
    # unbounded rather than zero.
    assert conformal_q(np.array([]), alpha=0.1) == np.inf


def test_conformal_quantile_never_shrinks_as_alpha_tightens():
    scores = np.random.default_rng(0).normal(size=200)
    q = [conformal_q(scores, alpha=a) for a in (0.20, 0.10, 0.05)]
    assert q[0] <= q[1] <= q[2]


# --------------------------------------------------------------- wilson bounds
def test_wilson_brackets_the_point_estimate_and_stays_in_range():
    lo, hi = wilson(9, 10)
    assert 0.0 <= lo < 0.9 < hi <= 1.0


def test_wilson_does_not_produce_an_impossible_interval_at_the_boundary():
    # 16 inserts, all covered: a Wald interval would collapse to [1, 1] and
    # claim certainty from a small sample.
    lo, hi = wilson(16, 16)
    assert lo < 1.0 and hi == 1.0
    assert lo < 0.9, "an interval this tight would overstate what 16 groups show"


def test_wilson_is_undefined_rather_than_zero_on_an_empty_sample():
    lo, hi = wilson(0, 0)
    assert np.isnan(lo) and np.isnan(hi)

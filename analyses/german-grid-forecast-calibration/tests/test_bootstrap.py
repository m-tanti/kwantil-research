"""The day-block bootstrap.

The specimen's method note claims that treating quarter-hours as independent
trials understates the interval, and that resampling whole days fixes it. That
is a testable claim about this code, not a general remark, so it is tested.
"""

import numpy as np
import pandas as pd
import pytest

import audit_specimen as A


def frame(inside_by_day, expensive_pattern=None):
    """A method-level group: one row per (day, slot)."""
    rows = []
    for d, insides in enumerate(inside_by_day):
        for s, ok in enumerate(insides):
            rows.append({
                "date": f"2026-01-{d + 1:03d}",
                "slot": s,
                "inside": bool(ok),
                "expensive": bool(expensive_pattern[d][s]) if expensive_pattern else (s % 5 == 0),
            })
    return pd.DataFrame(rows)


def test_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    days = [rng.random(24) < 0.9 for _ in range(200)]
    g = frame(days)
    c = A.day_counts(g)
    lo, hi = A.boot_days(c, lambda b: b["inside"] / np.maximum(b["n"], 1),
                         np.random.default_rng(1))
    assert lo < g["inside"].mean() < hi


def test_day_blocks_are_wider_than_ignoring_the_block_structure():
    """The whole reason the report does not use a binomial interval.

    Perfectly correlated days: every slot in a day shares one outcome. The
    effective sample size is the number of DAYS, not the number of slots, and a
    method that resamples slots would report an interval about sqrt(24) times
    too tight.
    """
    rng = np.random.default_rng(7)
    day_outcome = rng.random(150) < 0.9
    days = [np.repeat(o, 24) for o in day_outcome]
    g = frame(days)

    c = A.day_counts(g)
    lo, hi = A.boot_days(c, lambda b: b["inside"] / np.maximum(b["n"], 1),
                         np.random.default_rng(2))
    block_width = hi - lo

    # The naive alternative: pretend the 3,600 slots are independent trials.
    n = len(g)
    p = g["inside"].mean()
    binom_width = 2 * 1.96 * np.sqrt(p * (1 - p) / n)

    assert block_width > 3 * binom_width, (
        f"day-block width {block_width:.4f} should dwarf the binomial "
        f"{binom_width:.4f} when every slot in a day shares an outcome")


def test_uncorrelated_data_does_not_inflate_the_interval():
    """The correction has to be a correction, not a penalty.

    With slots independent within a day, the block bootstrap should land close
    to the binomial interval rather than far above it.
    """
    rng = np.random.default_rng(11)
    days = [rng.random(24) < 0.9 for _ in range(300)]
    g = frame(days)
    c = A.day_counts(g)
    lo, hi = A.boot_days(c, lambda b: b["inside"] / np.maximum(b["n"], 1),
                         np.random.default_rng(3))
    n = len(g)
    p = g["inside"].mean()
    binom_width = 2 * 1.96 * np.sqrt(p * (1 - p) / n)
    assert 0.6 * binom_width < (hi - lo) < 1.8 * binom_width


def test_day_counts_partitions_every_row():
    rng = np.random.default_rng(5)
    g = frame([rng.random(12) < 0.8 for _ in range(40)])
    c = A.day_counts(g)
    assert c["n"].sum() == len(g)
    # Expensive and cheap together account for every scored row, so a miss
    # cannot be counted twice or dropped when the ratio is formed.
    assert c["n_exp"].sum() + c["n_chp"].sum() == len(g)
    assert c["miss_exp"].sum() + c["miss_chp"].sum() == (~g["inside"]).sum()


def test_concentration_of_one_when_misses_are_spread_evenly():
    # Misses placed without regard to price: the ratio should sit on 1 and the
    # interval should cover it, which is the Significant branch of the rule.
    rng = np.random.default_rng(13)
    days = [rng.random(20) < 0.85 for _ in range(250)]
    g = frame(days)
    c = A.day_counts(g)
    lo, hi = A.boot_days(
        c, lambda b: (b["miss_exp"] / np.maximum(b["n_exp"], 1))
                     / np.maximum(b["miss_chp"] / np.maximum(b["n_chp"], 1), 1e-9),
        np.random.default_rng(4))
    assert lo < 1.0 < hi

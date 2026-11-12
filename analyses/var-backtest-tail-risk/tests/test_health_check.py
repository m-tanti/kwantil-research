"""The specimen Model Health Check: the rules, the supervisory arithmetic, and
the lineage a reader is told they can check.

The rule tests need nothing but the script. The lineage tests need a generated
run and skip without one, because the row-level backtest output is rebuilt
rather than shipped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_health_check.py"
ART = ROOT / "artifacts" / "health_check"

spec = importlib.util.spec_from_file_location("audit_health_check", SCRIPT)
hc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hc      # dataclasses resolve annotations through sys.modules
spec.loader.exec_module(hc)

needs_run = pytest.mark.skipif(not (ART / "meta.json").exists(), reason="specimen not generated")


# ------------------------------------------------------------ supervisory rule
def test_addend_table_is_article_366_table_1():
    expected = {0: 0.0, 4: 0.0, 5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85, 10: 1.0, 25: 1.0}
    for n, a in expected.items():
        assert hc.addend_for(n) == a


def test_addon_table_from_2027_is_article_325bf_table_3():
    expected = {4: 0.0, 5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42, 10: 0.50, 30: 0.50}
    for n, a in expected.items():
        assert hc.addon_2027_for(n) == a


def test_zone_boundaries():
    assert [hc.zone_for(n) for n in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_addend_agrees_with_the_library_table():
    from var_backtest_uq.backtest.basel_tl import _addon_for
    for n in range(0, 15):
        assert hc.addend_for(n) == _addon_for(n)


def test_rolling_count_scores_complete_windows_only():
    b = np.zeros(300, dtype=int)
    b[[10, 20, 260]] = 1
    c = hc.rolling_count(b, window=250)
    assert c.size == 51                      # 300 - 250 + 1
    assert c[0] == 2                         # days 0..249
    assert c[-1] == 1                        # days 50..299
    assert hc.rolling_count(np.zeros(100, dtype=int)).size == 0


def test_a_correct_model_is_outside_green_about_one_window_in_nine():
    # 1 - 89.22%, the cumulative probability at four exceptions in the 1996 table.
    assert hc.p_outside_green_if_correct() == pytest.approx(0.1078, abs=5e-4)


def test_var_component_is_the_higher_of_yesterday_and_the_scaled_average():
    n = 320
    var = np.full(n, 0.01)
    b = np.zeros(n, dtype=int)
    counts = hc.rolling_count(b)
    cap = hc.var_component(var, counts, book=1.0)
    v10 = 0.01 * math.sqrt(10)
    assert cap["base_eur"] == pytest.approx(3.0 * v10)
    assert cap["extra_eur"] == pytest.approx(0.0)

    # 8 overshootings in the window: factor 3.75, applied to the average only.
    b[100:108] = 1
    counts = hc.rolling_count(b)
    cap = hc.var_component(var, counts, book=1.0)
    assert cap["with_addend_eur"][0] == pytest.approx(3.75 * v10)
    assert cap["extra_eur"][0] == pytest.approx(0.75 * v10)

    # A spike yesterday larger than the scaled average wins, and then the addend adds nothing.
    spike = var.copy()
    spike[-2] = 0.20
    cap = hc.var_component(spike, hc.rolling_count(np.zeros(n, dtype=int)), book=1.0)
    assert cap["base_eur"][-1] == pytest.approx(0.20 * math.sqrt(10))


# --------------------------------------------------------------------- rules
E = 0.1078


def test_model_rule_observation_needs_both_limbs():
    assert hc.severity_model(True, 0.50, 0.01, 0.15, 0.0, E)[0] == "Observation"
    # rate inside the interval, but breaches cluster
    assert hc.severity_model(True, 0.01, 0.01, 0.15, 0.0, E)[0] == "Significant"


def test_model_rule_material_needs_the_zone_limb():
    assert hc.severity_model(False, 0.50, 0.01, 0.54, 0.28, E)[0] == "Material"
    # share above the floor but not distinguishable from a correct model
    assert hc.severity_model(False, 0.50, 0.01, 0.54, 0.05, E)[0] == "Significant"
    # distinguishable but below the floor
    assert hc.severity_model(False, 0.50, 0.01, 0.20, 0.15, E)[0] == "Significant"


def test_model_rule_scores_no_zone_away_from_the_99_level():
    sev, why = hc.severity_model(False, 0.50, 0.025, None, None, E)
    assert sev == "Significant" and "99% level only" in why


def test_test_rule():
    assert hc.severity_test(0.0, [])[0] == "Observation"
    assert hc.severity_test(0.3, [])[0] == "Significant"
    assert hc.severity_test(0.6, [])[0] == "Material"
    assert hc.severity_test(0.0, ["a control"])[0] == "Material"


def test_process_rule():
    assert hc.severity_process(False, 0.01, -0.10, False)[0] == "Observation"
    assert hc.severity_process(True, 0.01, -0.10, False)[0] == "Material"
    assert hc.severity_process(True, 0.01, -0.10, True)[0] == "Significant"
    assert hc.severity_process(True, 0.20, -0.10, False)[0] == "Significant"


def test_integrity_check_fires_on_an_impossible_pair():
    import pandas as pd

    n = 100
    g = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "var": np.full(n, -0.020), "es": np.full(n, -0.026),
        "realized_pnl": np.tile([0.004, -0.006, 0.009, -0.011], n // 4),
    })
    g["breached"] = g["realized_pnl"] < g["var"]
    assert hc.integrity(g)["passes"] is True
    bad = g.copy()
    bad.loc[:29, "es"] = -0.015                    # shallower than the VaR on 30 days
    out = hc.integrity(bad)
    assert out["passes"] is False and out["es_shallower_days"] == 30
    # a P&L in percent against a VaR in fractions
    units = g.copy()
    units["realized_pnl"] *= 100
    units["breached"] = units["realized_pnl"] < units["var"]
    assert hc.integrity(units)["scale_plausible"] is False


def test_integrity_rule():
    assert hc.severity_integrity(0.21)[0] == "Material"
    assert hc.severity_integrity(0.005)[0] == "Observation"


def test_bootstrap_stream_is_keyed_not_ordered():
    a = hc.stream("parametric", 0.01, "rate").integers(0, 1000, 5)
    hc.stream("something else").integers(0, 1000, 50)
    b = hc.stream("parametric", 0.01, "rate").integers(0, 1000, 5)
    assert np.array_equal(a, b)


# ------------------------------------------------------------------- lineage
def digest_of(directory: Path) -> str:
    d = hashlib.sha256()
    for f in sorted(p for p in directory.glob("*") if p.name != "meta.json"):
        d.update(f.name.encode())
        d.update(f.read_bytes())
    return d.hexdigest()


@needs_run
def test_digest_recomputes_from_the_shipped_files():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifact_digest_sha256"] == digest_of(ART)


@needs_run
def test_script_hash_identifies_the_code_that_ran():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    assert meta["script_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), (
        "audit_health_check.py has changed since the shipped artifacts were generated; re-run it")


@needs_run
def test_register_severities_are_the_rule_applied_to_the_evidence():
    """No severity in the register is typed by hand."""
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    cov = json.loads((ART / "coverage.json").read_text(encoding="utf-8"))
    for r in cov:
        sev, _ = hc.severity_model(
            r["stated_inside_ci"], r["independence_p"], r["alpha"],
            r.get("share_outside_green"), r.get("share_outside_lo"),
            meta["expected_share_outside_green"], r["direction"])
        assert sev == r["severity"], (r["method"], r["alpha"])


@needs_run
def test_integrity_passes_and_the_rule_revision_is_disclosed():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    # All 5 models pass. Where the 0.1.x output is present it is the control,
    # and it must fail: if it ever passes, the check has stopped working.
    assert all(meta["integrity_passes"].values())
    control = meta["integrity_control"]
    if control is not None:
        assert control["passes"] is False
        assert control["severity_if_reported"] == "Material"
    assert "revised once" in meta["rule_revision"]
    assert "8 July 2024" in meta["sources"]["rule_applied"]

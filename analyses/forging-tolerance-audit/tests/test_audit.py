"""The rules, the arithmetic and the lineage of the specimen Process Calibration Audit."""

import hashlib
import json
import pathlib

import numpy as np
import pytest

import audit_forging as af

HERE = pathlib.Path(__file__).resolve().parents[1]
ART = HERE / "artifacts" / "specimen"
needs_run = pytest.mark.skipif(not (ART / "meta.json").exists(), reason="specimen not generated")


def test_wilson_matches_a_known_value():
    lo, hi = af.wilson(6, 81)
    assert lo == pytest.approx(0.0343, abs=5e-4) and hi == pytest.approx(0.1523, abs=5e-4)
    assert af.wilson(0, 81)[0] == 0.0


def test_limit_rule():
    assert af.severity_limit(0.074, 0.034, 0.48, 1.4)[0] == "Material"
    assert af.severity_limit(0.0, 0.0, 0.98, 2.9)[0] == "Significant"
    sev, why = af.severity_limit(0.0, 0.0, 15.6, 47.0)
    assert sev == "Observation" and "never tested" in why
    assert af.severity_limit(0.0, 0.0, 1.8, 5.5)[0] == "Observation"
    # one part outside in 81 is not an established share
    assert af.severity_limit(1 / 81, af.wilson(1, 81)[0], 1.5, 4.5)[0] == "Observation"


def test_range_rule():
    assert af.severity_range(0.90, 0.86, 0.97, 1.0, 1.0)[0] == "Observation"
    assert af.severity_range(0.90, 0.62, 0.81, 0.67, 0.83)[0] == "Material"
    assert af.severity_range(0.90, 0.62, 0.81, 0.80, 0.95)[0] == "Significant"


def test_structure_rule():
    assert af.severity_structure(True, True)[0] == "Material"
    assert af.severity_structure(False, True)[0] == "Significant"
    assert af.severity_structure(False, False)[0] == "Observation"


def test_conformal_quantile_is_the_finite_sample_one():
    res = np.arange(1.0, 11.0)                      # 10 residuals
    assert af.conformal_q(res, 0.90) == 10.0        # ceil(11 * 0.9) = 10th smallest
    assert af.conformal_q(res, 0.50) == 6.0


def test_ridge_survives_a_constant_feature_and_recovers_a_line():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    X = np.column_stack([x, np.full(60, 243.85)])   # a channel stuck at one value
    y = 5.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1]
    f = af.ridge_fit(X, y)
    assert np.all(np.isfinite(f(X)))
    assert np.corrcoef(f(X), y)[0, 1] > 0.999


def digest_of(directory: pathlib.Path) -> str:
    d = hashlib.sha256()
    for f in sorted(p for p in directory.glob("*") if p.name != "meta.json"):
        d.update(f.name.encode())
        d.update(f.read_bytes())
    return d.hexdigest()


@needs_run
def test_digest_and_script_hash_recompute():
    meta = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
    assert meta["artifact_digest_sha256"] == digest_of(ART)
    assert meta["script_sha256"] == hashlib.sha256((HERE / "src" / "audit_forging.py").read_bytes()).hexdigest()
    assert meta["features_script_sha256"] == hashlib.sha256((HERE / "src" / "features.py").read_bytes()).hexdigest()


@needs_run
def test_register_severities_are_the_rules_applied_to_the_evidence():
    for r in json.loads((ART / "limits.json").read_text(encoding="utf-8")):
        assert af.severity_limit(r["share_outside"], r["outside_lo"], r["cpk"], r["nearest_limit_sd"])[0] == r["severity"]
    for r in json.loads((ART / "ranges.json").read_text(encoding="utf-8"))["constructions"]:
        assert af.severity_range(r["stated"], r["lo"], r["hi"], r["coverage_long"], r["long_hi"])[0] == r["severity"]


@needs_run
def test_decision_counts_add_up():
    d = json.loads((ART / "decision.json").read_text(encoding="utf-8"))
    for p in d["policies"] + d["curve"]:
        assert p["caught_per_100"] + p["escaped_per_100"] == pytest.approx(d["over_limit_per_100"])
        assert p["caught_per_100"] + p["flagged_in_limit_per_100"] == pytest.approx(p["flagged_per_100"])

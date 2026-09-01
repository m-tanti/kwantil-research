"""The pre-registered severity rule.

These exist because the rule is the specimen's main claim about itself: that
severity is derived and not chosen. A rule nobody tests is an assertion with
indentation.
"""

import audit_specimen as A


def test_nominal_inside_interval_is_never_worse_than_observation():
    # Sample size is allowed to win. A band whose stated level sits inside the
    # bootstrap interval scores Observation no matter how concentrated its
    # misses look, because there is no established shortfall to concentrate.
    sev, why = A.severity(nominal_inside_ci=True, conc=9.9, conc_lo=8.0)
    assert sev == "Observation"
    assert "inside" in why


def test_material_needs_both_limbs():
    sev, _ = A.severity(False, conc=A.CONCENTRATION_TRIGGER + 0.1, conc_lo=1.05)
    assert sev == "Material"


def test_concentration_indistinguishable_from_none_is_only_significant():
    # Large point estimate, but the interval covers 1: the concentration could
    # be nothing at all.
    sev, why = A.severity(False, conc=2.0, conc_lo=0.9)
    assert sev == "Significant"
    assert "not distinguishable" in why


def test_real_but_small_concentration_is_only_significant():
    # Distinguishable from none, below the materiality floor. This is the limb
    # that stops a 1.02x concentration with a tight interval being called
    # Material.
    sev, why = A.severity(False, conc=1.02, conc_lo=1.01)
    assert sev == "Significant"
    assert "materiality floor" in why


def test_the_floor_actually_binds_at_the_boundary():
    # Rolling split-conformal at 80% lands on 1.2499 against a 1.25 floor in the
    # published specimen and is recorded Significant. If the floor were ever
    # loosened to rescue that finding, this fails.
    just_under, _ = A.severity(False, conc=1.2499, conc_lo=1.10)
    just_over, _ = A.severity(False, conc=1.2500, conc_lo=1.10)
    assert just_under == "Significant"
    assert just_over == "Material"


def test_severity_ordering_is_total_and_known():
    # The register takes the most severe rating a construction reaches at any
    # level, which needs the ranks to be exactly these three.
    ranks = {"Observation", "Significant", "Material"}
    seen = {
        A.severity(True, 1.0, 1.0)[0],
        A.severity(False, 1.0, 0.5)[0],
        A.severity(False, 2.0, 1.5)[0],
    }
    assert seen == ranks

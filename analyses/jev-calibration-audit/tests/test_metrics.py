import numpy as np

from analysis import aps_sets, apply_temperature, auroc, decision_cost, ece_top, fit_temperature, rps


def test_ece_zero_for_perfect_bins():
    conf = np.repeat([0.2, 0.8], 100)
    correct = np.concatenate([np.r_[np.ones(20), np.zeros(80)], np.r_[np.ones(80), np.zeros(20)]])
    assert abs(ece_top(conf, correct, bins=2)) < 1e-9


def test_rps_bounds_and_ordering():
    y = np.array([4])
    near = np.array([[0, 0, 0, 1.0, 0]])
    far = np.array([[1.0, 0, 0, 0, 0]])
    exact = np.array([[0, 0, 0, 0, 1.0]])
    assert rps(exact, y) == 0
    assert rps(near, y) < rps(far, y)
    assert rps(far, y) == 1.0


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4000, 5))
    z = logits - logits.max(1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(5, p=p) for p in P])
    sharp = apply_temperature(P, 0.5)  # over-confident version
    T = fit_temperature(np.log(sharp), y)
    assert 1.6 < T < 2.6


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) - 0.5) < 1e-9


def test_aps_coverage_on_synthetic():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3000, 6)) * 2
    z = logits - logits.max(1, keepdims=True)
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(6, p=p) for p in P])
    S = aps_sets(P[:1500], y[:1500], P[1500:], alpha=0.1)
    cov = S[np.arange(1500), y[1500:]].mean()
    assert 0.87 < cov < 0.93


def test_decision_cost_threshold():
    p = np.array([0.05, 0.5, 0.95]); y = np.array([1, 1, 0])
    # threshold 1/11 = 0.091: the 0.05 positive is missed (false negative, cost 10) and the
    # 0.95 negative is acted on (false positive, cost 1): 11 per 3 decisions
    assert abs(decision_cost(p, y, 1, 10) - 1000 * 11 / 3) < 1e-9

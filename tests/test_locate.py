"""Locating the transmitter: exact recovery, Jacobian, efficiency against the Cramer-Rao bound, the
mirror minimum of a single pass, and the error-radius helpers."""

import numpy as np
import pytest
from cases import CASES, FC_NR, FC_SAR, GEO, MODES, truth_of
from scipy import stats

from ntn_pass import locate as L
from ntn_pass import orbit


@pytest.mark.parametrize("mode", MODES)
def test_noise_free_recovers_truth(mode):
    m = CASES[mode][0]
    truth = truth_of(mode, (12.0, -7.0))
    s = L.simulate(m, GEO, truth)
    fit = L.solve(s, GEO, L.nuisance_init(s, GEO, np.array([60.0, 45.0])))
    assert fit.converged
    assert L.horizontal_error_km(fit.ne_km, truth[:2]) < 1e-6  # 1 mm
    assert np.allclose(fit.theta[2:], truth[2:], atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_jacobian_matches_finite_differences(mode):
    m = CASES[mode][0]
    th = truth_of(mode, (3.0, 4.0))
    J = L.jacobian(m, GEO, th)
    for i in range(len(th)):
        e = np.zeros(len(th))
        e[i] = 1e-4
        num = (L.predict(m, GEO, th + e) - L.predict(m, GEO, th - e)) / 2e-4
        assert np.allclose(J[:, i], num, rtol=1e-5, atol=1e-6 * np.abs(J[:, i]).max())


@pytest.mark.parametrize("mode", MODES)
def test_estimator_reaches_crb(mode, rng):
    """Normalised estimation error squared over N trials is chi-square(k N) when the estimator is
    efficient; assert it lies inside the two-sided 99.9 % interval."""
    m = CASES[mode][0]
    truth = truth_of(mode)
    P = L.crb(m, GEO, truth)
    Pinv = np.linalg.inv(P)
    n, k = 1500, len(truth)
    nees = 0.0
    for _ in range(n):
        s = L.simulate(m, GEO, truth, rng)
        fit = L.solve(s, GEO, truth)  # start at truth: this test is about efficiency, not convergence
        e = fit.theta - truth
        nees += e @ Pinv @ e
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], k * n)
    assert lo < nees < hi


def test_single_pass_mirror_minimum():
    """One satellite: a second minimum exists across the ground track; the true side has lower cost."""
    leo = orbit.leo_pass_states(45.0)
    m = L.leosar(leo.r, leo.v, 5.0, FC_SAR)
    s = L.simulate(m, GEO, truth_of("leosar"))
    mirror = 2 * L.sub_satellite_offset(leo, GEO)
    fit = L.solve(s, GEO, L.nuisance_init(s, GEO, mirror))
    assert fit.converged
    assert L.horizontal_error_km(fit.ne_km, [0, 0]) > 500.0
    assert fit.cost > 0.0  # noise-free truth has cost 0


def test_per_row_sigmas_interleave():
    m = L.rel18(np.zeros((3, 3)) + 7e6, np.zeros((3, 3)), [1.0, 2.0, 3.0], 9.0, FC_NR)
    assert list(m.sigma) == [1.0, 9.0, 2.0, 9.0, 3.0, 9.0]


def test_position_information_matches_crb():
    """The projected 2x2 information is the inverse of the position block of the full CRB."""
    for mode in MODES:
        m = CASES[mode][0]
        th = truth_of(mode, (2.0, -1.0))
        P = L.crb(m, GEO, th)[:2, :2]
        assert np.allclose(np.linalg.inv(L.position_information(m, GEO, th)), P, rtol=1e-6)


def test_drift_needs_times():
    leo = orbit.leo_pass_states(45.0)
    with pytest.raises(ValueError):
        L.leosar(leo.r, leo.v, 1.0, FC_SAR, drift_unknown=True)


def test_p_within_circular():
    """Circular Gaussian: P(r <= R) = 1 - exp(-R^2 / (2 sigma^2))."""
    s = 2.0
    for R in (1.0, 3.0, 5.0):
        assert L.p_within(np.eye(2) * s**2, R, n=200000) == pytest.approx(1 - np.exp(-(R**2) / (2 * s**2)), abs=0.005)
        assert L.radius_km(np.eye(2) * s**2, 1 - np.exp(-(R**2) / (2 * s**2)), n=200000) == pytest.approx(R, rel=0.01)

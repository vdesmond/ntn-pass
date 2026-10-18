"""Is a claimed position consistent? False-alarm rate and noncentrality of the consistency test, the
smallest detectable false offset, and the near-mirror search."""

import numpy as np
import pytest
from cases import CASES, GEO, MODES, truth_of
from scipy import stats

from ntn_pass import integrity as I
from ntn_pass import locate as L
from ntn_pass import rel18


@pytest.mark.parametrize("mode", MODES)
def test_consistency_false_alarm_rate(mode, rng):
    """Under a truthful report the statistic is chi-square(m - nuisance): check the false-alarm rate."""
    m = CASES[mode][0]
    truth = truth_of(mode)
    pfa, n = 0.05, 3000
    thr = I.threshold(m, pfa)
    hits = sum(I.consistency_statistic(L.simulate(m, GEO, truth, rng), GEO, truth[:2]) > thr for _ in range(n))
    lo, hi = stats.binom.ppf([0.0005, 0.9995], n, pfa)
    assert lo <= hits <= hi


@pytest.mark.parametrize("mode", MODES)
def test_noncentrality_predicts_mean_statistic(mode, rng):
    """With a false report, E[T] = dof + lambda; compare the Monte Carlo mean against it."""
    m = CASES[mode][0]
    truth = truth_of(mode)
    reported = np.array([0.4, -0.3])  # km
    lam = I.noncentrality(m, GEO, truth[:2], reported)
    n = 2000
    t = np.array([I.consistency_statistic(L.simulate(m, GEO, truth, rng), GEO, reported) for _ in range(n)])
    expected_mean = I.dof(m) + lam
    sd_of_mean = np.sqrt(2 * (I.dof(m) + 2 * lam) / n)
    assert abs(t.mean() - expected_mean) < 4 * sd_of_mean


def test_min_detectable_hits_required_noncentrality():
    meas, _ = rel18.measurements(45.0, 60.0)
    a = np.radians(30.0)
    d = I.min_detectable_km(meas, GEO, a, 0.01, 0.9)
    off = d * np.array([np.cos(a), np.sin(a)])
    lam = I.noncentrality(meas, GEO, np.zeros(2), off)
    assert lam == pytest.approx(I.required_noncentrality(0.9, I.dof(meas), 0.01), rel=1e-3)
    assert I.detection_probability(meas, GEO, np.zeros(2), off, 0.01) == pytest.approx(0.9, abs=1e-3)


def test_mirror_search_beats_geometric_mirror():
    """The near-mirror point found by the solver explains the data at least as well as the geometric
    mirror, and a longer window separates it better."""
    short, st = rel18.measurements(45.0, 10.0)
    long_, st_long = rel18.measurements(45.0, 120.0)
    mirror = 2 * L.sub_satellite_offset(st, GEO)
    m = I.mirror_check(short, GEO, st)
    assert m["noncentrality"] <= I.noncentrality(short, GEO, np.zeros(2), mirror) + 1e-9
    assert I.mirror_check(long_, GEO, st_long)["noncentrality"] > m["noncentrality"]


def test_required_noncentrality():
    lam = I.required_noncentrality(0.9, 5, 0.01)
    assert stats.ncx2.sf(stats.chi2.isf(0.01, 5), 5, lam) == pytest.approx(0.9)
    assert I.detection_pd(lam, 5, 0.01) == pytest.approx(0.9)
    assert I.detection_pd(0.0, 5, 0.01) == pytest.approx(0.01)

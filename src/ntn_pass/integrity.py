"""Is a claimed transmitter position consistent with what the satellites measured?

The test statistic is the whitened residual energy with the position held at the claim and the nuisance
parameters fitted: chi-square with len(meas) - nuisance degrees of freedom when the claim is true, and
noncentral chi-square when it is not. That gives the false-alarm threshold, the probability of catching a
false claim, the smallest false offset caught with a given probability, and whether the mirror image of the
true position across the satellite ground track can be told apart (the one ambiguity a single satellite
leaves).
"""

import numpy as np
from scipy import optimize, stats

from . import locate as L


def _projector_out_nuisance(meas, geo, th):
    """Whitened-space projector removing the nuisance directions (identity when there are none)."""
    m = len(meas)
    if not meas.nuisance:
        return np.eye(m)
    Jn = L.jacobian(meas, geo, th)[:, 2:] / meas.sigma[:, None]
    return np.eye(m) - Jn @ np.linalg.pinv(Jn)


def consistency_statistic(meas: L.Measurements, geo: L.Geo, ne_reported):
    """Whitened residual energy with the position fixed at the reported location and the nuisance
    parameters fitted. Chi-square with len(meas) - nuisance degrees of freedom if the report is true."""
    th = L.nuisance_init(meas, geo, np.asarray(ne_reported, dtype=float))
    r = (meas.y - L.predict(meas, geo, th)) / meas.sigma
    return float(r @ r)


def dof(meas: L.Measurements):
    return len(meas) - meas.nuisance


def threshold(meas: L.Measurements, pfa):
    return float(stats.chi2.isf(pfa, dof(meas)))


def noncentrality(meas: L.Measurements, geo: L.Geo, ne_true, ne_reported):
    """Non-centrality of the statistic when the transmitter is at ne_true but ne_reported is claimed.

    Linearised around the reported point for the nuisance projection; the position mismatch itself
    enters through the exact model difference.
    """
    th_rep = np.concatenate([ne_reported, np.zeros(meas.nuisance)])
    th_true = np.concatenate([ne_true, np.zeros(meas.nuisance)])
    dy = (L.predict(meas, geo, th_true) - L.predict(meas, geo, th_rep)) / meas.sigma
    P = _projector_out_nuisance(meas, geo, th_rep)
    e = P @ dy
    return float(e @ e)


def detection_probability(meas: L.Measurements, geo: L.Geo, ne_true, ne_reported, pfa=0.01):
    lam = noncentrality(meas, geo, ne_true, ne_reported)
    return float(stats.ncx2.sf(threshold(meas, pfa), dof(meas), lam))


def detection_pd(lam, dof_, pfa):
    """Probability the test rejects a claim whose statistic has noncentrality lam."""
    return float(stats.ncx2.sf(stats.chi2.isf(pfa, dof_), dof_, lam))


def required_noncentrality(pd, dof_, pfa):
    """Noncentrality at which the test detects with probability pd."""
    thr = stats.chi2.isf(pfa, dof_)
    return optimize.brentq(lambda x: stats.ncx2.sf(thr, dof_, x) - pd, 1e-6, 1e6)


def min_detectable_km(meas: L.Measurements, geo: L.Geo, direction, pfa=0.01, pd=0.9, d_max=3000.0):
    """Smallest offset along `direction` (rad from north) whose false report is detected with pd."""
    u = np.array([np.cos(direction), np.sin(direction)])
    lam_req = required_noncentrality(pd, dof(meas), pfa)

    def lam(d):
        return noncentrality(meas, geo, np.zeros(2), d * u)

    ds = np.geomspace(1e-3, d_max, 80)
    vals = np.array([lam(d) for d in ds])
    k = np.flatnonzero(vals >= lam_req)
    if not len(k):
        return np.inf
    lo, hi = (ds[k[0] - 1] if k[0] else 0.0), ds[k[0]]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lam(mid) < lam_req else (lo, mid)
    return hi


def mirror_check(meas: L.Measurements, geo: L.Geo, states, pfa=0.01):
    """The near-mirror point that best explains noise-free data from the true site (solver started at the
    geometric mirror across the ground track of `states`), and the probability the test rejects it."""
    mirror = 2 * L.sub_satellite_offset(states, geo)
    truth = np.zeros(2 + meas.nuisance)
    s = L.simulate(meas, geo, truth)
    fit = L.solve(s, geo, L.nuisance_init(s, geo, mirror))
    thr = stats.chi2.isf(pfa, dof(meas))
    return {
        "mirror_offset_km": round(float(np.hypot(*fit.ne_km)), 1),
        "noncentrality": round(float(fit.cost), 2),
        "pd": round(float(stats.ncx2.sf(thr, dof(meas), fit.cost)), 4),
    }

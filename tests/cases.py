"""Measurement sets shared by test_locate and test_integrity: one per system, with nuisance values."""

import numpy as np

from ntn_pass import locate as L
from ntn_pass import orbit
from ntn_pass.sites import SOPHIA

GEO = L.Geo.at(SOPHIA)
FC_NR, FC_SAR = 2e9, 406e6


def _cases():
    leo = orbit.leo_pass_states(45.0)
    meo = orbit.meo_burst_states()
    t = leo.t_s - leo.t_s.mean()
    sig = np.linspace(15e-9, 45e-9, len(leo.r))  # per-instant sigmas
    return {
        "rel18": (L.rel18(leo.r, leo.v, sig, 20.0, FC_NR), [100.0]),
        "rel18_known_carrier": (L.rel18(leo.r, leo.v, 30e-9, 20.0, FC_NR, carrier_unknown=False), []),
        "leosar": (L.leosar(leo.r, leo.v, 5.0, FC_SAR), [37.0]),
        "leosar_drift": (L.leosar(leo.r, leo.v, 1.2, FC_SAR, t=t, drift_unknown=True), [37.0, 0.007]),
        "meosar": (L.meosar(meo.r, meo.v, np.linspace(0.5e-6, 2e-6, len(meo.r)), 0.2, FC_SAR), [3.0, 37.0]),
    }


CASES = _cases()
MODES = list(CASES)


def truth_of(mode, ne=(0.0, 0.0)):
    return np.concatenate([ne, CASES[mode][1]])

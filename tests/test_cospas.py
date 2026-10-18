"""COSPAS-SARSAT: the link-budget C/N0 lands where T.019 specifies accuracy, the MEOLUT G/T threshold, and
the Cramer-Rao bounds against what T.019 requires of a MEOLUT."""

import pytest

from ntn_pass import beacon, bounds, cospas
from ntn_pass import link as lb
from ntn_pass.analytic import slant_range


def test_worst_fgb_lands_in_t019_accuracy_band():
    """A worst-compliant FGB (35 dBm, -3 dBi) at 5 deg into a Galileo FOC repeater gives a system C/N0
    inside the 34.8-37.8 dB-Hz band where T.019 specifies the TOA / FOA accuracy."""
    r = lb.galileo_slant_range_m(5.0)
    up = lb.cn0_up_dbhz(beacon.FGB_WORST.eirp_dbw(5.0), r, lb.sat_gt_db(5.0))
    sys = lb.cn0_sys_dbhz(up, lb.cn0_dn_dbhz(r), lb.cn0_im_dbhz())
    assert 34.8 <= sys <= 37.8


def test_gt_threshold_costs_half_a_db():
    up = 46.0
    gt = cospas.gt_threshold_db(up)
    dn = lb.cn0_dn_dbhz(slant_range(lb.GALILEO_ALT_M, cospas.MASK_DEG), gt_db=gt)
    assert up - lb.cn0_sys_dbhz(up, dn) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("kind,rise", [("fgb", 150e-6), ("fgb", 250e-6), ("sgb", None)])
def test_crb_below_t019_requirement_at_sensitivity(kind, rise):
    fs = 200e3 if kind == "fgb" else 307.2e3
    s = beacon.fgb_waveform(fs, rise) if kind == "fgb" else beacon.sgb_waveform(fs)
    toa, foa = bounds.waveform_crb(s, fs, cospas.SENSITIVITY_DBHZ[kind])
    req_toa, req_foa = cospas.T019_SIGMA[kind]
    assert toa < req_toa
    assert foa < req_foa

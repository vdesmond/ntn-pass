"""COSPAS-SARSAT: how the system measures a distress beacon, and what it must achieve.

MEOSAR: Galileo FOC repeaters relay each burst to a MEOLUT, which measures its time and frequency of
arrival (TOA, FOA) through every satellite it tracks and locates the beacon from one burst. LEOSAR: one
low satellite measures the Doppler of successive bursts over a pass.

Sources (March 2021 issues):
- C/S T.016 Issue 1 Rev.5, Galileo FOC repeater (Table 4.3): group delay uncertainty < 163 ns (95 %) and slope
  2.5 us per 4 kHz; translation frequency 1,138,049,997.6 Hz, short-term stability 4e-12 over 100 ms.
  Section 2 interoperability: group delay stability < 200 ns peak-to-peak over any 3 minutes.
- C/S T.019 Issue 2 Rev.5, MEOLUT: sensitivity 34.8 dB-Hz (FGB) / 30.55 dB-Hz (SGB) (5.2); TOA / FOA
  standard deviation 25 us / 0.20 Hz for FGB at 34.8-37.8 dB-Hz, [1] us / [0.20] Hz for SGB at
  [30.55-33.55] dB-Hz, bracketed = provisional (5.8.1); location accuracy (5.6.1 / 5.6.2); 5 deg minimum
  elevation at both ends (Annex F); a stand-alone MEOLUT with four antennas (Annex F example).
"""

import itertools

import numpy as np

from . import beacon, orbit, tle
from . import link as lb
from . import locate as L
from .analytic import slant_range
from .sites import SITES

MASK_DEG = 5.0
BEACON_SITE, MEOLUT_SITE = SITES["sophia"], SITES["toulouse"]
GEO = L.Geo.at(BEACON_SITE)
MAX_SATS = 4

SENSITIVITY_DBHZ = {"fgb": 34.8, "sgb": 30.55}
T019_SIGMA = {"fgb": (25e-6, 0.20), "sgb": (1e-6, 0.20)}  # (TOA s, FOA Hz); SGB values provisional
T019_TARGETS = {  # (time from first burst s, radius km, required fraction), T.019 5.6.1 / 5.6.2
    "fgb": [(0, 5, 0.90), (600, 5, 0.95), (600, 10, 0.98)],
    "sgb": [(0, 5, 0.95), (30, 5, 0.97), (300, 1, 0.97)],
}

GROUP_DELAY_UNC_S = 163e-9  # 95 %, Galileo FOC
GROUP_DELAY_SLOPE_S_PER_HZ = 2.5e-6 / 4e3  # FOC, normal mode, in the 1 dB band
GROUP_DELAY_STABILITY_PP_S = 200e-9  # interoperability requirement, any 3 minutes
TRANSLATION_HZ = 1_138_049_997.6  # FOC
TRANSLATION_STABILITY_100MS = 4e-12  # FOC (IOV 2e-11, interoperability requirement 1e-11)

# Satellite-side floors on one satellite's TOA / FOA, independent per satellite (so the emission-time and
# carrier nuisances do not absorb them): the group-delay uncertainty (95 % -> 1 sigma) and its 3-minute
# stability (peak-to-peak taken as uniform), and the translation oscillator. The group-delay slope is a
# deterministic function of the beacon frequency, characterised per satellite (C/S R.018); uncalibrated it
# would add up to +-0.75 us at the SGB's +-1.2 kHz carrier tolerance.
TOA_FLOOR_S = float(np.hypot(GROUP_DELAY_UNC_S / 1.96, GROUP_DELAY_STABILITY_PP_S / np.sqrt(12)))
FOA_FLOOR_HZ = TRANSLATION_STABILITY_100MS * TRANSLATION_HZ


def link_limited_sigma(crb_toa_s, crb_foa_hz):
    """Per-satellite sigma: link CRB combined with the satellite-side floors."""
    return np.hypot(crb_toa_s, TOA_FLOOR_S), np.hypot(crb_foa_hz, FOA_FLOOR_HZ)


def gt_threshold_db(up_max_dbhz, el_meolut_deg=MASK_DEG, cost_db=0.5):
    """MEOLUT G/T above which the downlink lowers the system C/N0 by less than cost_db at the best uplink."""
    need_dn = up_max_dbhz + 10 * np.log10(1 / (1 - 10 ** (-cost_db / 10)) - 1)
    return float(lb.GT_MEOLUT_DB + need_dn - lb.cn0_dn_dbhz(slant_range(lb.GALILEO_ALT_M, el_meolut_deg)))


class GalileoView:
    """Galileo satellites as seen from the beacon and the MEOLUT."""

    def __init__(self):
        self.f = tle.load("galileo")

    def at(self, t_s):
        """States at one instant, both looks, and which satellites both ends see above the mask."""
        r, v = orbit.propagate_at(self.f.sats, t_s)
        r, v = r[:, 0], v[:, 0]
        lb_ = orbit.look(BEACON_SITE, r, v, accel=False)
        lm = orbit.look(MEOLUT_SITE, r, v, accel=False)
        ok = (lb_.el_deg >= MASK_DEG) & (lm.el_deg >= MASK_DEG)
        return r, v, lb_, lm, ok


def sigmas(kind, case, lb_, lm):
    """Per-satellite (TOA s, FOA Hz) for one beacon type: "t019" is a just-compliant MEOLUT, "link" the bound
    for the weakest compliant beacon through each satellite plus the satellite floors."""
    if case == "t019":
        return T019_SIGMA[kind]
    cn0 = lb.bent_pipe_cn0_dbhz(beacon.worst(kind).eirp_dbw(lb_.el_deg), lb_, lm)
    return link_limited_sigma(*beacon.crb_at(kind, cn0))


def burst_information(view, t, kind, case, subset=None):
    """Position information of one burst at time t through the best <= MAX_SATS satellites (min CRB trace)
    unless a subset is given. Returns (information, satellites used, satellites visible)."""
    r, v, lb_, lm, ok = view.at(t)
    idx = np.flatnonzero(ok)
    st, sf = sigmas(kind, case, lb_, lm)
    st = np.broadcast_to(st, len(r))
    sf = np.broadcast_to(sf, len(r))
    th = np.array([0.0, 0.0, 0.0, 0.0])

    def info(sel):
        m = L.meosar(r[sel], v[sel], st[sel], sf[sel], lb.F_UP)
        return L.position_information(m, GEO, th)

    if subset is None:
        best = None
        for comb in itertools.combinations(idx, min(MAX_SATS, len(idx))):
            F = info(list(comb))
            tr = np.trace(np.linalg.inv(F))
            if best is None or tr < best[0]:
                best = (tr, list(comb), F)
        return best[2], best[1], len(idx)
    sel = [i for i in subset if ok[i]]
    return info(sel), sel, len(idx)


def leosar_sigma_hz():
    """Per-burst FOA: the FGB bound at the T.019 sensitivity with the beacon's 3e-9 residual frequency
    variation (T.001 2.3.1) as independent noise."""
    foa = beacon.crb_at("fgb", SENSITIVITY_DBHZ["fgb"])[1]
    return float(np.hypot(foa, beacon.FGB_RESIDUAL_FREQ * lb.F_UP))


def leosar_burst_times(p: orbit.Pass, rng):
    """FGB bursts over a pass: random phase in the first period, then 47.5-52.5 s apart (T.001 2.2.1)."""
    t = p.aos_s + np.cumsum(np.r_[rng.uniform(0, 50), rng.uniform(47.5, 52.5, 40)])
    return t[t < p.los_s]

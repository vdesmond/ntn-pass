"""Rel-18 NR NTN network-verified UE location: what the network measures, and how well.

TS 38.305 V18.8.0 4.3.11: the network checks a UE-reported location against round-trip time (RTT) and
Doppler from the serving satellite at several instants. TR 38.882 V18.0.0 counts 5-10 km as verified.

Noise per instant:
- RTT: TS 38.133 V19.0.0 Table 10.1.25C.2-1 (FR1-NTN, 15 kHz PRS, >= 52 PRB, 0 dB) bounds the UE Rx-Tx time
  difference to +-(59 + delta) Tc with delta = 80 Tc from Table 10.1.25C.2-2. Read as a 90 % two-sided
  bound (an assumption: the table gives a +- limit, not a sigma). The network leg (gNB Rx-Tx) has no NTN
  requirement in the sources, so it is taken as equal. The ``ssb_only`` case uses the SSB's own TOA bound instead.
- Doppler: the SSB FOA bound at the link-budget SNR (TR 38.821 Set-1, LEO-600 payload, S-band handheld),
  which the fine estimator in ``acquire`` reaches (TOA and FOA both) from -2 dB per resource element, below
  every SNR on these passes.
"""

import numpy as np
from scipy import stats

from . import bounds, nr, orbit
from . import link as lb
from . import locate as L
from .constants import C
from .sites import SITES

SITE = SITES["sophia"]
GEO = L.Geo.at(SITE)
FC = lb.FC_S
UE_RXTX_90_S = (59 + 80) * nr.TC
SIGMA_RTT_S = np.sqrt(2.0) * UE_RXTX_90_S / stats.norm.ppf(0.95)
STEP_S = 10.0  # one RTT + Doppler pair every 10 s (assumed, Rel-18 leaves it to the LMF)


def measurements(max_el, window_s, ssb_only=False, floor_m=0.0, step_s=STEP_S):
    """RTT + Doppler every step_s over a window centred on the closest approach of the Starlink pass over
    the site whose maximum elevation is nearest max_el. floor_m adds a range error (troposphere,
    ionosphere, ephemeris) treated as noise. Returns (Measurements, States)."""
    n = max(2, int(round(window_s / step_s)) + 1)
    st = orbit.leo_pass_states(max_el, window_s, n)
    lk = orbit.look(SITE, st.r, st.v, accel=False)
    snr = lb.link("set1", "LEO-600", "S", "DL").cnr_db(lk.range_m, lk.el_deg)
    crb = np.array([bounds.ssb_crb(10 ** (x / 10)) for x in snr])
    sig_rtt = np.sqrt(2.0) * crb[:, 0] if ssb_only else np.full(n, SIGMA_RTT_S)
    sig_rtt = np.hypot(sig_rtt, 2 * floor_m / C)
    return L.rel18(st.r, st.v, sig_rtt, crb[:, 1], FC, t=st.t_s - st.t_s.mean()), st

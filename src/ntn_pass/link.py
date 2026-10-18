"""Link budgets: how much signal reaches the receiver, for both systems.

NR NTN service link, TR 38.821 parameter sets. CNR (TR 38.821 V16.2.0 eq. 6.1.3.1-2):
    CNR = EIRP + G/T - k - FSPL - PL_A - PL_SM - PL_SL - PL_pol - PL_add - 10 log10(B)
with k = -228.6 dBW/K/Hz and, along a pass, the elevation-dependent terms recomputed per sample:

- FSPL from the real slant range (TR 38.811 eq. 6.6-2),
- gaseous absorption, ITU-R P.676 Annex 2 with the P.835 mean annual reference atmosphere at sea
  level (TR 38.811 6.6.4: T = 288.15 K, p = 1013.25 hPa, water vapour 7.5 g/m^3),
- scintillation: 2.2 dB ionospheric below 6 GHz (TR 38.821 Table 6.1.3.2-1, note 1); tropospheric above
  6 GHz from TR 38.811 Table 6.6.6.2.1-1 (99 % level at 20 GHz, Toulouse), interpolated in elevation,
- shadowing margin 3 dB for a handheld, 0 dB for a VSAT; polarisation and additional losses 0 dB.

Beams: the satellite beam is assumed steered at the UE (Earth-fixed beams, scenarios C1/D1), so the
satellite antenna stays at peak gain. ``beam_edge_loss_db`` adds the loss of a UE at the edge of a moving
beam (C2/D2; 3 dB at the -3 dB contour).

406 MHz distress link through a Galileo MEOSAR repeater to a MEOLUT, from C/S T.016 Issue 1 Rev.5 (Galileo
FOC repeater, Table 4.3) and C/S T.019 Issue 2 Rev.5 (MEOLUT, Annex A link budgets):
(C/N0)sys = 1 / [(C/N0)up^-1 + (C/N0)dn^-1 + (C/N0)im^-1]. C/N0 is the unmodulated-carrier power over noise
density (T.019 5.2); for the constant-envelope beacon signals that is the total received power.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .analytic import slant_range

K_BOLTZMANN_DB = -228.6  # dBW/K/Hz
T0 = 290.0  # K

# ITU-R P.835 mean annual global reference atmosphere at sea level (TR 38.811 6.6.4).
P_HPA, T_K, RHO_GM3 = 1013.25, 288.15, 7.5

# TR 38.811 Table 6.6.6.2.1-1: tropospheric scintillation, 99 % level at 20 GHz in Toulouse.
# The elevation column did not survive text extraction; 10, 20, ..., 90 deg is the column the
# TR 38.821 link budget uses (0.3 dB at 30 deg, 1.1 dB at the 12.5 deg GEO case).
_TROPO_EL = np.arange(10.0, 91.0, 10.0)
_TROPO_DB = np.array([1.08, 0.48, 0.30, 0.22, 0.17, 0.13, 0.12, 0.12, 0.12])
IONO_SCINT_DB = 2.2


def fspl_db(d_m, f_hz):
    """Free-space path loss, TR 38.811 eq. 6.6-2: 32.45 + 20 log10(f_GHz) + 20 log10(d_m)."""
    return 32.45 + 20.0 * np.log10(f_hz / 1e9) + 20.0 * np.log10(d_m)


def gaseous_db(f_hz, el_deg, mode="approx"):
    """Gaseous absorption on the slant path (ITU-R P.676; Annex 2 = "approx", Annex 1 = "exact")."""
    from itur.models import itu676

    el = np.clip(np.asarray(el_deg, dtype=float), 5.0, 90.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = itu676.gaseous_attenuation_slant_path(f_hz / 1e9, el, RHO_GM3, P_HPA, T_K, h=0.0, mode=mode)
    return np.asarray(a.value, dtype=float)


def scintillation_db(f_hz, el_deg):
    el = np.asarray(el_deg, dtype=float)
    if f_hz < 6e9:
        return np.full(el.shape, IONO_SCINT_DB)
    return np.interp(el, _TROPO_EL, _TROPO_DB)  # held at 1.08 dB below 10 deg


def g_over_t_db(gain_dbi, nf_db, t_ant_k):
    """TR 38.821 eq. 6.1.3.1-3: G/T = G - 10 log10(T_a + T0 (10^(NF/10) - 1))."""
    return gain_dbi - 10.0 * np.log10(t_ant_k + T0 * (10.0 ** (nf_db / 10.0) - 1.0))


def dish_gain_dbi(diameter_m, f_hz, efficiency):
    lam = 299_792_458.0 / f_hz
    return 10.0 * np.log10(efficiency * (np.pi * diameter_m / lam) ** 2)


@dataclass(frozen=True)
class SatPayload:
    """Per-beam satellite parameters, TR 38.821 Tables 6.1.1.1-1 (Set-1) and 6.1.1.1-2 (Set-2)."""

    name: str
    altitude_km: float
    band: str  # "S" or "Ka"
    eirp_density_dbw_mhz: float  # DL
    g_t_db: float  # UL
    beam_diameter_km: float


PAYLOADS = {
    # (set, orbit, band)
    ("set1", "LEO-600", "S"): SatPayload("Set-1 LEO-600 S", 600, "S", 34.0, 1.1, 50),
    ("set1", "LEO-600", "Ka"): SatPayload("Set-1 LEO-600 Ka", 600, "Ka", 4.0, 13.0, 20),
    ("set1", "LEO-1200", "S"): SatPayload("Set-1 LEO-1200 S", 1200, "S", 40.0, 1.1, 90),
    ("set1", "LEO-1200", "Ka"): SatPayload("Set-1 LEO-1200 Ka", 1200, "Ka", 10.0, 13.0, 40),
    ("set2", "LEO-600", "S"): SatPayload("Set-2 LEO-600 S", 600, "S", 28.0, -4.9, 90),
    ("set2", "LEO-600", "Ka"): SatPayload("Set-2 LEO-600 Ka", 600, "Ka", -4.0, 5.0, 50),
    ("set2", "LEO-1200", "S"): SatPayload("Set-2 LEO-1200 S", 1200, "S", 34.0, -4.9, 190),
    ("set2", "LEO-1200", "Ka"): SatPayload("Set-2 LEO-1200 Ka", 1200, "Ka", 2.0, 5.0, 90),
}


@dataclass(frozen=True)
class Terminal:
    """TR 38.821 Table 6.1.1.1-3."""

    name: str
    band: str
    tx_power_dbm: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    nf_db: float
    t_ant_k: float
    shadow_margin_db: float

    @property
    def g_t_db(self):
        return g_over_t_db(self.rx_gain_dbi, self.nf_db, self.t_ant_k)

    @property
    def eirp_dbw(self):
        return self.tx_power_dbm - 30.0 + self.tx_gain_dbi


# VSAT: 60 cm equivalent aperture; Tx gain as tabulated (43.2 dBi at 30 GHz). The Rx gain at 20 GHz is
# not tabulated. The aperture efficiency 0.6 is fitted to the G/T of 15.9 dB/K that Table 6.1.3.3-1 uses.
# The TR is not self-consistent here: 43.2 dBi at 30 GHz implies an efficiency of 0.587, which would give
# G/T 15.83 (printed 15.8), and with 0.6 the Tx gain would be 43.3 dBi.
HANDHELD = Terminal("Handheld", "S", 23.0, 0.0, 0.0, 7.0, 290.0, 3.0)
VSAT = Terminal("VSAT", "Ka", 33.0, 43.2, float(dish_gain_dbi(0.6, 20e9, 0.6)), 1.2, 150.0, 0.0)

# TR 38.821 Table 6.1.3.2-1
SYSTEM_BW_HZ = {"S": 30e6, "Ka": 400e6}
FC_S, FC_KA = 2e9, 20e9  # TR 38.821 example service-link carriers (DL); n256 DL is 2170-2200 MHz
FREQ_HZ = {("S", "DL"): FC_S, ("S", "UL"): FC_S, ("Ka", "DL"): FC_KA, ("Ka", "UL"): 30e9}
UL_HANDHELD_BW_HZ = 360e3


@dataclass(frozen=True)
class Link:
    payload: SatPayload
    terminal: Terminal
    direction: str  # "DL" or "UL"
    reuse: int = 1
    beam_edge_loss_db: float = 0.0

    @property
    def f_hz(self):
        return FREQ_HZ[(self.payload.band, self.direction)]

    @property
    def bandwidth_hz(self):
        if self.direction == "UL" and self.terminal is HANDHELD:
            return UL_HANDHELD_BW_HZ
        return SYSTEM_BW_HZ[self.payload.band] / self.reuse

    @property
    def eirp_dbw(self):
        if self.direction == "DL":
            return self.payload.eirp_density_dbw_mhz + 10.0 * np.log10(self.bandwidth_hz / 1e6)
        return self.terminal.eirp_dbw

    @property
    def g_t_db(self):
        return self.terminal.g_t_db if self.direction == "DL" else self.payload.g_t_db

    def losses_db(self, el_deg):
        el = np.asarray(el_deg, dtype=float)
        return (
            gaseous_db(self.f_hz, el)
            + scintillation_db(self.f_hz, el)
            + self.terminal.shadow_margin_db
            + self.beam_edge_loss_db
        )

    def cn0_dbhz(self, range_m, el_deg):
        """Carrier-to-noise density along the pass (independent of the channel bandwidth for the UL,
        and of the reuse factor for the DL, since the EIRP density is fixed)."""
        return self.eirp_dbw + self.g_t_db - K_BOLTZMANN_DB - fspl_db(range_m, self.f_hz) - self.losses_db(el_deg)

    def cnr_db(self, range_m, el_deg):
        return self.cn0_dbhz(range_m, el_deg) - 10.0 * np.log10(self.bandwidth_hz)


# The TR defines payloads for 600 km and 1200 km; each is applied to the nearest real constellation:
# LEO-600 on Starlink (~466 km mean altitude, so ~2 dB more favourable than the TR geometry) and
# LEO-1200 on OneWeb (~1200 km).
PAYLOAD_ORBIT = {"starlink": "LEO-600", "oneweb": "LEO-1200"}
SETS = ("set1", "set2")


def link(set_name, orbit, band, direction, reuse=1, **kw) -> Link:
    term = HANDHELD if band == "S" else VSAT
    return Link(PAYLOADS[(set_name, orbit, band)], term, direction, reuse, **kw)


F_UP = 406.05e6  # middle of the 406.0-406.1 MHz band (T.019 Annex F)
F_DN_GALILEO = 1544.1e6
GALILEO_ALT_M = 23_222e3  # T.019 Table A.1 / T.016
EOC_EL_DEG = 5.0  # T.016 edge of coverage: beacon at 5 deg elevation

POL_LOSS_DB = 3.0  # linear beacon antenna into a circularly polarised satellite antenna
FADING_DB = 2.5  # T.019 Table A.1 / A.2 nominal ("between -20 dB and -0.5 dB")
SHARING_DB = 10.0  # downlink power shared by 8 signals + 1 dB for noise (T.019 Table A.1)

GT_SAT_EOC_DB, GT_SAT_CENTRE_DB = -15.3, -13.6  # Galileo FOC, T.016 Table 4.3
EIRP_DN_DBW = 17.8  # Galileo FOC minimum downlink EIRP, T.016 Table 4.3
C_I3_DB = 28.0  # Galileo FOC transponder linearity, T.016 Table 4.3
PASSBAND_HZ = 90e3  # 3 dB, normal mode

# MEOLUT receive G/T is design dependent (T.019 A.5). 4 dB/K is an assumption; the same value appears in
# "Preliminary Analysis of Quality of Link for Cospas-Sarsat Second Generation Beacon in MEOSAR"
# (researchgate.net/publication/384103902), seen only through a search summary, not verified.
GT_MEOLUT_DB = 4.0
LO_MEOLUT_DB = 1.0  # local losses at the MEOLUT, also design dependent


def galileo_slant_range_m(el_deg):
    return slant_range(GALILEO_ALT_M, el_deg)


def sat_gt_db(beacon_el_deg, alt_m=GALILEO_ALT_M, r_e=6_371e3):
    """Galileo FOC G/T, interpolated linearly in the satellite's off-nadir angle between the edge of
    coverage (beacon at 5 deg) and the centre (nadir)."""

    def nadir(el):
        return np.degrees(np.arcsin(r_e / (r_e + alt_m) * np.cos(np.radians(el))))

    x = nadir(np.asarray(beacon_el_deg, dtype=float)) / nadir(EOC_EL_DEG)
    return GT_SAT_CENTRE_DB + (GT_SAT_EOC_DB - GT_SAT_CENTRE_DB) * np.clip(x, 0.0, None)


def cn0_up_dbhz(eirp_dbw, range_m, gt_db, f_hz=F_UP, pol_db=POL_LOSS_DB, fading_db=FADING_DB):
    """T.019 Table A.2: C/N0 = EIRP - FSL - Pol - Lf + G/T - k."""
    return eirp_dbw - fspl_db(range_m, f_hz) - pol_db - fading_db + gt_db - K_BOLTZMANN_DB


def cn0_dn_dbhz(range_m, gt_db=GT_MEOLUT_DB, lo_db=LO_MEOLUT_DB, eirp_dbw=EIRP_DN_DBW, f_hz=F_DN_GALILEO):
    """Per-beacon downlink: satellite EIRP less the power sharing, at the MEOLUT."""
    return eirp_dbw - SHARING_DB - fspl_db(range_m, f_hz) - FADING_DB - lo_db + gt_db - K_BOLTZMANN_DB


def cn0_im_dbhz():
    """Third-order intermodulation as noise: products at C/I3 below the total output, spread over the
    passband, against one beacon's share of that output. A crude conversion; it stays ~20 dB above the
    uplink either way."""
    return C_I3_DB - SHARING_DB + 10.0 * np.log10(PASSBAND_HZ)


def cn0_sys_dbhz(*terms_dbhz):
    lin = sum(10.0 ** (-np.asarray(t) / 10.0) for t in terms_dbhz)
    return -10.0 * np.log10(lin)


def bent_pipe_cn0_dbhz(eirp_dbw, beacon_look, meolut_look):
    """System C/N0 of one beacon through one Galileo repeater, from the two ends' geometry."""
    up = cn0_up_dbhz(eirp_dbw, beacon_look.range_m, sat_gt_db(beacon_look.el_deg))
    return cn0_sys_dbhz(up, cn0_dn_dbhz(meolut_look.range_m), cn0_im_dbhz())

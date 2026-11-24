"""Link budget vs TR 38.821 V16.2.0 Table 6.1.3.3-1 (LEO rows), computed at the 30 deg target elevation
with the slant range of the spherical-Earth reference geometry (Table 6.1.3.2-1).

Every input column is checked to the TR's printed precision. The end-to-end CNR matches in the 26 cases
where the TR is self-consistent. In the other 14 (all Ka-band DL rows, and SC7/SC22 UL) the TR's own
printed inputs give a CNR 0.05-0.09 dB higher than the printed result (SC6 DL: inputs give 8.58, printed
8.5), i.e. the printed CNR looks truncated. Those are strict xfails.
"""

import numpy as np
import pytest

from ntn_pass import analytic as ref
from ntn_pass import link as lb

EL = 30.0

# case: (set, orbit, band, reuse, DL CNR dB, UL CNR dB)
TABLE = {
    "SC6": ("set1", "LEO-600", "Ka", 1, 8.5, 18.4),
    "SC7": ("set1", "LEO-600", "Ka", 3, 8.5, 23.1),
    "SC8": ("set1", "LEO-600", "Ka", 2, 8.5, 21.4),
    "SC9": ("set1", "LEO-600", "S", 1, 6.6, 2.8),
    "SC10": ("set1", "LEO-600", "S", 3, 6.6, 2.8),
    "SC11": ("set1", "LEO-1200", "Ka", 1, 9.1, 13.0),
    "SC12": ("set1", "LEO-1200", "Ka", 3, 9.1, 17.8),
    "SC13": ("set1", "LEO-1200", "Ka", 2, 9.1, 16.0),
    "SC14": ("set1", "LEO-1200", "S", 1, 7.2, -2.6),
    "SC15": ("set1", "LEO-1200", "S", 3, 7.2, -2.6),
    "SC21": ("set2", "LEO-600", "Ka", 1, 0.5, 10.4),
    "SC22": ("set2", "LEO-600", "Ka", 3, 0.5, 15.1),
    "SC23": ("set2", "LEO-600", "Ka", 2, 0.5, 13.4),
    "SC24": ("set2", "LEO-600", "S", 1, 0.6, -3.2),
    "SC25": ("set2", "LEO-600", "S", 3, 0.6, -3.2),
    "SC26": ("set2", "LEO-1200", "Ka", 1, 1.1, 5.0),
    "SC27": ("set2", "LEO-1200", "Ka", 3, 1.1, 9.8),
    "SC28": ("set2", "LEO-1200", "Ka", 2, 1.1, 8.0),
    "SC29": ("set2", "LEO-1200", "S", 1, 1.2, -8.6),
    "SC30": ("set2", "LEO-1200", "S", 3, 1.2, -8.6),
}


def _cnr(case, direction):
    s, orbit, band, reuse, *_ = TABLE[case]
    ln = lb.link(s, orbit, band, direction, reuse)
    d = ref.slant_range(ln.payload.altitude_km * 1e3, EL)
    return float(ln.cnr_db(d, EL))


TRUNCATED = {(c, "DL") for c, row in TABLE.items() if row[2] == "Ka"} | {("SC7", "UL"), ("SC22", "UL")}

CNR_CASES = [
    pytest.param(
        c,
        d,
        marks=[pytest.mark.xfail(strict=True, reason="TR printed CNR below its own printed inputs (truncated)")]
        if (c, d) in TRUNCATED
        else [],
        id=f"{c}-{d}",
    )
    for c in TABLE
    for d in ("DL", "UL")
]


@pytest.mark.parametrize("case, direction", CNR_CASES)
def test_cnr_tr38821(case, direction):
    tr = TABLE[case][4 if direction == "DL" else 5]
    assert round(_cnr(case, direction), 1) == tr


# Printed DL EIRP (dBm) per case. UL EIRP is 23.0 dBm (handheld) / 76.2 dBm (VSAT) everywhere.
DL_EIRP_DBM = {
    "SC6": 60.0, "SC7": 55.2, "SC8": 57.0, "SC9": 78.8, "SC10": 74.0,
    "SC11": 66.0, "SC12": 61.2, "SC13": 63.0, "SC14": 84.8, "SC15": 80.0,
    "SC21": 52.0, "SC22": 47.2, "SC23": 49.0, "SC24": 72.8, "SC25": 68.0,
    "SC26": 58.0, "SC27": 53.2, "SC28": 55.0, "SC29": 78.8, "SC30": 74.0,
}  # fmt: skip


@pytest.mark.parametrize("case", TABLE)
def test_eirp_columns(case):
    s, orbit, band, reuse, *_ = TABLE[case]
    assert round(float(lb.link(s, orbit, band, "DL", reuse).eirp_dbw) + 30.0, 1) == DL_EIRP_DBM[case]
    assert round(float(lb.link(s, orbit, band, "UL", reuse).eirp_dbw) + 30.0, 1) == (23.0 if band == "S" else 76.2)


def test_terminal_g_over_t():
    """Handheld: derived from 0 dBi, NF 7 dB, 290 K. VSAT: the efficiency is fitted to this value (see
    link.VSAT), so only the handheld half is an independent check."""
    assert round(float(lb.HANDHELD.g_t_db), 1) == -31.6
    assert round(float(lb.VSAT.g_t_db), 1) == 15.9


@pytest.mark.parametrize(
    "h_km, f_ghz, fspl, atm, scint",
    [
        (600, 2, 159.1, 0.1, 2.2),
        (600, 20, 179.1, 0.5, 0.3),
        (600, 30, 182.6, 0.5, 0.3),
        (1200, 2, 164.5, 0.1, 2.2),
        (1200, 20, 184.5, 0.5, 0.3),
        (1200, 30, 188.0, 0.5, 0.3),
    ],
)
def test_path_loss_columns(h_km, f_ghz, fspl, atm, scint):
    d = ref.slant_range(h_km * 1e3, EL)
    assert round(float(lb.fspl_db(d, f_ghz * 1e9)), 1) == fspl
    assert round(float(lb.gaseous_db(f_ghz * 1e9, EL)), 1) == atm
    assert round(float(lb.scintillation_db(f_ghz * 1e9, EL)), 1) == scint


def test_gaseous_annex1_matches_geo_row():
    """The GEO 12.5 deg row (20 GHz: 1.2 dB) is matched by P.676 Annex 1, not by the Annex 2 baseline."""
    assert round(float(lb.gaseous_db(20e9, 12.5, mode="exact")), 1) == 1.2
    assert round(float(lb.gaseous_db(20e9, 12.5)), 1) == 1.1


# 406 MHz: C/S T.019 Annex A and T.016 values reproduced from their own inputs.
def test_t019_slant_ranges():
    assert ref.slant_range(23_222e3, 5.0) / 1e3 == pytest.approx(28_350, abs=1.0)
    # GLONASS / GPS columns: within 10 km (0.003 dB of path loss). The TR rounds or uses another radius
    assert ref.slant_range(19_100e3, 5.0) / 1e3 == pytest.approx(24_120, abs=10)
    assert ref.slant_range(20_850e3, 5.0) / 1e3 == pytest.approx(25_920, abs=10)


def test_t016_horizon_to_zenith_path_difference():
    assert 20 * np.log10(ref.slant_range(lb.GALILEO_ALT_M, 0.0) / lb.GALILEO_ALT_M) == pytest.approx(1.9, abs=0.05)


def test_t019_table_a2_galileo_uplink():
    fsl = lb.fspl_db(28_350e3, 406.034e6)
    assert round(float(fsl), 1) == 173.7
    assert lb.cn0_up_dbhz(9.0, 28_350e3, -12.7, f_hz=406.034e6) == pytest.approx(45.7, abs=0.05)


@pytest.mark.xfail(strict=True, reason="T.019 Table A.1 prints 185.4 dB; 28,350 km at 1544.1 MHz gives 185.27")
def test_t019_table_a1_galileo_fsl():
    assert round(float(lb.fspl_db(28_350e3, 1544.1e6)), 1) == 185.4


@pytest.mark.xfail(strict=True, reason="T.019 Table A.1 prints G/T - Lo + 23.7; its own inputs sum to +20.7")
def test_t019_table_a1_galileo_ebn0():
    assert 7.0 - 185.4 - 2.5 - lb.K_BOLTZMANN_DB - 29.0 + 2.0 == pytest.approx(23.7, abs=0.05)


def test_satellite_gt_endpoints():
    assert lb.sat_gt_db(5.0) == pytest.approx(lb.GT_SAT_EOC_DB)
    assert lb.sat_gt_db(90.0) == pytest.approx(lb.GT_SAT_CENTRE_DB)
    assert lb.GT_SAT_EOC_DB < lb.sat_gt_db(30.0) < lb.GT_SAT_CENTRE_DB


def test_system_cn0_reciprocal_sum():
    assert lb.cn0_sys_dbhz(40.0, 40.0) == pytest.approx(40.0 - 10 * np.log10(2))
    assert lb.cn0_sys_dbhz(40.0, 200.0) == pytest.approx(40.0, abs=1e-9)

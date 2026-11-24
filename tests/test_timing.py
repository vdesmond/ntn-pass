"""Uplink timing: TA-Info encoding against the TS 38.213 decoding formula, ASN.1 ranges, the
reference values from TR 38.821 Table 4.2-2, two-body propagation, and HARQ counting."""

import numpy as np
import pytest

from ntn_pass import analytic, nr, orbit, tle
from ntn_pass import timing as T
from ntn_pass.constants import MU_EARTH, OMEGA_EARTH, C


def test_numerology():
    assert nr.TC == pytest.approx(1 / 1.96608e9)
    assert nr.cp_normal_s(0) == pytest.approx(4.6875e-6)  # 144 * 64 Tc
    assert nr.TE_NTN_S[(15, 15)] == pytest.approx(29 * 64 / 1.96608e9)


@pytest.mark.parametrize("d, ddot, dddot", [(1.5e6, 6000.0, 40.0), (8e5, -7000.0, 90.0), (40_581e3, 3.0, 0.0)])
def test_ta_info_round_trip(d, ddot, dddot):
    """Encode at the epoch, decode with TS 38.213 4.2: Delay_common and its derivatives match d/c within
    the field granularities."""
    info = T.encode_ta_info(d, ddot, dddot)
    assert info.in_range()
    # Delay_common is quadratic in dt, so central differences give its derivatives exactly.
    h = 1.0
    m, z, p = T.delay_common_s(info, np.array([-h, 0.0, h]))
    assert abs(z - d / C) <= nr.TA_COMMON_LSB_S / 4 + 1e-15
    assert abs((p - m) / (2 * h) - ddot / C) <= nr.TA_DRIFT_LSB / 4 + 1e-15
    assert abs((p - 2 * z + m) / h**2 - dddot / C) <= nr.TA_DRIFT_VAR_LSB / 2 + 1e-15


def test_ta_info_variant_is_clamped_when_negative():
    info = T.encode_ta_info(1e6, 0.0, -20.0)
    assert info.variant_clamped and info.ta_common_drift_variant == 0


def test_fit_recovers_a_quadratic():
    dt = np.arange(0.0, 60.0, 1.0)
    d = 1.2e6 + 5000.0 * dt + 0.5 * 30.0 * dt**2
    info = T.fit_ta_info(dt, d)
    err = 2 * T.delay_common_s(info, dt) - 2 * d / C
    # only quantisation is left: half an LSB of each field, growing with dt and dt^2
    bound = (nr.TA_COMMON_LSB_S + nr.TA_DRIFT_LSB * dt + nr.TA_DRIFT_VAR_LSB * dt**2) / 2
    assert np.all(np.abs(err) <= bound + 1e-12)


def test_tr38821_reference_values_fit_the_fields():
    """TR 38.821 Table 4.2-2 worst cases (10 deg) fit the TS 38.331 ranges."""
    geo = analytic.slant_range(35_786e3, 10.0)
    leo600 = analytic.slant_range(600e3, 10.0)
    assert T.encode_ta_info(geo, 0.0, 0.0).in_range()  # GEO feeder RTT 270.73 ms in ta-Common
    assert T.slots_15khz(4 * leo600 / C) == 26  # LEO-600 transparent RTT 25.77 ms -> K_offset 26
    assert T.slots_15khz(4 * geo / C) == 542 <= nr.K_OFFSET_RANGE[1]  # GEO 541.46 ms
    assert T.slots_15khz(2 * geo / C) == 271 <= nr.K_MAC_RANGE[1]  # GEO feeder 270.73 ms in kmac


@pytest.mark.parametrize("key", ["starlink", "oneweb", "galileo"])
def test_ephemeris_codes_in_range(key):
    f = tle.load(key)
    jd, fr = tle.jd_split([0.0])
    r, v, _ = orbit.propagate(f.sats, jd, fr)
    for i in range(len(f)):
        assert T.quantize_state(r[i, 0], v[i, 0])[2]


def test_two_body_circular_orbit():
    """Circular equatorial orbit given as an ECEF state: after a quarter period the propagated ECEF
    position equals the inertial rotation n t followed by the Earth rotation -omega t."""
    a = 6_900e3
    n = np.sqrt(MU_EARTH / a**3)
    r_eci = np.array([a, 0.0, 0.0])
    v_eci = np.array([0.0, a * n, 0.0])
    v0 = v_eci - np.cross([0.0, 0.0, OMEGA_EARTH], r_eci)
    tq = 0.25 * 2 * np.pi / n
    r, _ = T.propagate_two_body(r_eci, v0, np.array([tq]))
    th = n * tq - OMEGA_EARTH * tq
    assert np.allclose(r[0], [a * np.cos(th), a * np.sin(th), 0.0], atol=1e-3)


def test_harq_tr_anchor():
    """TR 38.821 6.1.1.3 assumption N = RTT / T_slot: RTT 32 ms at 15 kHz needs 32 processes."""
    assert T.harq_processes_needed(32e-3, 0) == 32
    assert T.harq_processes_needed(32e-3, 1) == 64
    assert T.harq_duty_cycle(32e-3, 0, 16) == pytest.approx(0.5)


def test_rp_choice_keeps_the_ue_gnb_rtt():
    """TS 38.321: UE-gNB RTT = TA + kmac. RP at the gNB puts the feeder round trip in the TA, while RP at the
    satellite puts it in kmac (whole 1 ms slots), so the RTT is the same up to that rounding."""
    service, feeder = 9.3e-3, 7.6e-3
    rtt_gnb = service + feeder + 0 * 1e-3
    rtt_sat = service + T.slots_15khz(feeder) * 1e-3
    assert 0 <= rtt_sat - rtt_gnb < 1e-3

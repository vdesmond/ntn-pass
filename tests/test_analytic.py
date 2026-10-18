"""Closed-form circular-orbit model vs the 3GPP NTN reference tables.

Sources (values typed from the documents, not from memory):
- TR 38.821 V16.2.0, Table 4.2-2 "Reference scenario parameters" (min elevation 10 deg).
- TR 38.811 V15.4.0, Table 5.3.4.3.2-7 "Summary of Doppler shift and shift variation for different
  altitudes" (2 GHz column).
"""

import math

import pytest

from ntn_pass import analytic as ref

GEO, LEO600, LEO1200 = 35_786e3, 600e3, 1200e3


@pytest.mark.parametrize(
    "h, km",
    [(GEO, 40_581), (LEO600, 1_932), (LEO1200, 3_131)],
)
def test_max_slant_range_tr38821(h, km):
    assert round(ref.slant_range(h, 10.0) / 1e3) == km


@pytest.mark.parametrize(
    "h, links, ms",
    [
        (GEO, 2, 541.46),  # Scenario A, transparent: service + feeder link
        (GEO, 1, 270.73),  # Scenario B, regenerative: service link only
        (LEO600, 2, 25.77),  # Scenario C
        (LEO1200, 2, 41.77),
        (LEO600, 1, 12.89),  # Scenario D
        (LEO1200, 1, 20.89),
    ],
)
def test_max_rtt_tr38821(h, links, ms):
    assert round(ref.max_rtt_s(h, 10.0, links) * 1e3, 2) == ms


@pytest.mark.parametrize(
    "h, tr_ppm",
    [
        (LEO600, 24.0),  # TR 38.821 Table 4.2-2 and TR 38.811 (0.0024 %)
        (LEO1200, 21.0),  # TR 38.821 Table 4.2-2
        (1500e3, 20.0),  # TR 38.811 (0.002 %)
    ],
)
def test_max_doppler_tr(h, tr_ppm):
    """The tabulated max Doppler is the closed-form maximum (0 deg elevation) rounded up to the next ppm."""
    assert math.ceil(ref.doppler_ppm(h, 0.0)) == tr_ppm


@pytest.mark.xfail(strict=True, reason="TR value not reproduced by the closed form (6.41 ppm vs 7.5 ppm)")
def test_max_doppler_10000km_tr38811():
    assert math.ceil(ref.doppler_ppm(10_000e3, 0.0) * 2) / 2 == 7.5


@pytest.mark.parametrize(
    "h, tr_hz_s",
    [(1500e3, 180.0), (10_000e3, 6.0)],
)
def test_doppler_rate_tr38811(h, tr_hz_s):
    """Max |Doppler rate| at 2 GHz matches the TR to its printed precision (2 / 1 significant figures)."""
    model = ref.doppler_rate_ppm_s_zenith(h) * 2e9 / 1e6
    half_ulp = 5.0 if tr_hz_s >= 100 else 0.5
    assert abs(model - tr_hz_s) <= half_ulp


@pytest.mark.xfail(strict=True, reason="TR value not reproduced by the closed form (581 Hz/s vs 544 Hz/s)")
def test_doppler_rate_600km_tr38811():
    assert abs(ref.doppler_rate_ppm_s_zenith(LEO600) * 2e3 - 544.0) <= 5.0


@pytest.mark.xfail(strict=True, reason="TR value not reproduced by the closed form (0.123 vs 0.13 ppm/s)")
def test_doppler_rate_1200km_tr38821():
    assert round(ref.doppler_rate_ppm_s_zenith(LEO1200), 2) == 0.13

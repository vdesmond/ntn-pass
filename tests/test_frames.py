import numpy as np
from sgp4.propagation import gstime
from skyfield.api import wgs84

from ntn_pass.frames import enu_matrix, geodetic_to_ecef, gmst82
from ntn_pass.sites import SITES


def test_gmst82_matches_sgp4_library():
    for jd in np.linspace(2451545.0, 2461700.0, 50):
        assert abs(gmst82(jd) - gstime(jd)) < 1e-10


def test_geodetic_matches_skyfield():
    for s in SITES.values():
        sky = wgs84.latlon(s.lat_deg, s.lon_deg, s.alt_m).itrs_xyz.m
        assert np.allclose(geodetic_to_ecef(s.lat_deg, s.lon_deg, s.alt_m), sky, atol=1e-3)


def test_enu_orthonormal_and_up_is_geodetic_normal():
    s = SITES["sophia"]
    m = enu_matrix(s.lat_deg, s.lon_deg)
    assert np.allclose(m @ m.T, np.eye(3))
    # moving along "up" changes only the height
    p = geodetic_to_ecef(s.lat_deg, s.lon_deg, s.alt_m + 1000.0) - geodetic_to_ecef(s.lat_deg, s.lon_deg, s.alt_m)
    assert np.allclose(p / 1000.0, m[2], atol=1e-9)

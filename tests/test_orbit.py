import numpy as np
import pytest

from ntn_pass import analytic, orbit
from ntn_pass.constants import OMEGA_EARTH, R_EARTH_TR, C
from ntn_pass.sites import SOPHIA
from ntn_pass.tle import mean_altitude_km

MASK = 10.0
FC = 2e9


@pytest.mark.parametrize("key", ["starlink", "oneweb"])
def test_pass_edges_and_tca(survey, key):
    s = survey[key]
    for p in s["passes"][:: max(1, len(s["passes"]) // 40)]:
        sat = s["fleet"].sats[p.sat]
        for t in (p.aos_s, p.los_s):
            assert abs(orbit._geom(sat, SOPHIA, t).el_deg[0] - MASK) < 1e-3
        g = orbit._geom(sat, SOPHIA, p.tca_s)
        assert abs(g.range_rate[0]) < 1.0  # m/s
        assert MASK <= p.max_el_deg <= 90.0


@pytest.mark.parametrize("key", ["starlink", "oneweb"])
def test_doppler_sign_and_monotone(survey, key):
    """Approaching -> positive Doppler; a LEO pass sweeps it downwards through zero at TCA."""
    s = survey[key]
    p = max(s["passes"], key=lambda p: p.max_el_deg)
    t, lk = orbit.pass_track(s["fleet"].sats[p.sat], SOPHIA, p, 1.0)
    d = lk.doppler_hz(FC)
    assert d[0] > 0 > d[-1]
    assert np.all(np.diff(d) < 0)
    assert abs(t[np.argmin(np.abs(d))] - p.tca_s) <= 1.0


@pytest.mark.parametrize("key", ["starlink", "oneweb", "galileo"])
def test_range_accel_analytic_vs_numeric(survey, key):
    s = survey[key]
    p = max(s["passes"], key=lambda p: p.max_el_deg)
    t, lk = orbit.pass_track(s["fleet"].sats[p.sat], SOPHIA, p, 1.0)
    num = np.gradient(lk.range_rate, t)[2:-2]
    assert np.max(np.abs(lk.range_accel[2:-2] - num)) < 5e-3 * np.max(np.abs(lk.range_accel))


@pytest.mark.parametrize("key", ["starlink", "oneweb"])
def test_sgp4_passes_within_closed_form_bounds(survey, key):
    """Real passes stay inside the circular-orbit envelope (+ Earth rotation and ellipsoid margins)."""
    s = survey[key]
    h_max = max(mean_altitude_km(s["fleet"].sats[p.sat]) for p in s["passes"]) * 1e3 + 30e3  # + ecc
    h_min = min(mean_altitude_km(s["fleet"].sats[p.sat]) for p in s["passes"]) * 1e3 - 30e3
    rot_ppm = 1e6 * OMEGA_EARTH * R_EARTH_TR * np.cos(np.deg2rad(SOPHIA.lat_deg)) / C
    dop_bound = analytic.doppler_ppm(h_min, MASK) + rot_ppm
    delay_bound = analytic.slant_range(h_max, MASK) * 1.005 / C
    for p in s["passes"][:: max(1, len(s["passes"]) // 100)]:
        _, lk = orbit.pass_track(s["fleet"].sats[p.sat], SOPHIA, p, 2.0)
        assert np.max(np.abs(lk.doppler_ppm())) <= dop_bound
        assert np.max(lk.delay_s) <= delay_bound

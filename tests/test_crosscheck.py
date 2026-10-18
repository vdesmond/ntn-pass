"""Independent pipeline check against Skyfield (same SGP4 core, independent time scales and frames).

With Skyfield's UT1 - UTC passed in, the remaining difference is numerical (cm level). Without it the
budget is ~50 m per 0.1 s of UT1 - UTC at LEO, which would hide a frame-convention error (using GAST
instead of GMST alone is ~0.5 km), so the tolerances below are set on the dUT1-corrected comparison.
"""

import numpy as np
import pytest
from skyfield.api import EarthSatellite, load, wgs84

from ntn_pass import orbit, tle
from ntn_pass.sites import SOPHIA

TS = load.timescale()


def _compare(fleet, p):
    t = np.arange(p.aos_s, p.los_s, 5.0)
    T = TS.from_datetimes([tle.utc(x) for x in t])
    jd, fr = tle.jd_split(t)
    r, v, err = orbit.propagate([fleet.sats[p.sat]], jd, fr, dut1_s=float(np.mean(T.dut1)))
    assert not err.any()
    lk = orbit.look(SOPHIA, r[0], v[0])
    topo = wgs84.latlon(SOPHIA.lat_deg, SOPHIA.lon_deg, SOPHIA.alt_m)
    rel = (EarthSatellite(*fleet.lines[p.sat], ts=TS) - topo).at(T)
    alt, az, dist = rel.altaz()
    *_, rate = rel.frame_latlon_and_rates(topo)
    daz = np.abs((lk.az_deg - az.degrees + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(lk.el_deg))
    return (
        np.abs(lk.el_deg - alt.degrees).max(),
        daz.max(),
        np.abs(lk.range_m - dist.m).max(),
        np.abs(lk.range_rate - rate.m_per_s).max(),
    )


@pytest.mark.parametrize("key", ["starlink", "oneweb", "galileo"])
@pytest.mark.parametrize("which", ["highest", "lowest"])
def test_pass_matches_skyfield(survey, key, which):
    s = survey[key]
    p = (max if which == "highest" else min)(s["passes"], key=lambda p: p.max_el_deg)
    d_el, d_az, d_rng, d_rdot = _compare(s["fleet"], p)
    assert d_el < 1e-5  # deg
    assert d_az < 1e-5  # deg, scaled by cos(el)
    assert d_rng < 0.05  # m
    assert d_rdot < 1e-3  # m/s

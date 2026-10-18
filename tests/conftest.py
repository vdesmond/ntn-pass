import numpy as np
import pytest

from ntn_pass import orbit, tle
from ntn_pass.sites import SOPHIA

WINDOWS = {"starlink": (3 * 3600, 10.0), "oneweb": (6 * 3600, 20.0), "galileo": (24 * 3600, 60.0)}


@pytest.fixture(scope="session")
def survey():
    """Per constellation: fleet, coarse grid, elevations and refined passes over Sophia (10 deg mask)."""
    out = {}
    for key, (dur, step) in WINDOWS.items():
        f = tle.load(key)
        t, jd, fr = tle.time_grid(dur, step)
        el = orbit.elevation_grid(f.sats, SOPHIA, jd, fr)
        out[key] = dict(fleet=f, t=t, el=el, passes=orbit.find_passes(f, SOPHIA, t, el, 10.0))
    return out


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)

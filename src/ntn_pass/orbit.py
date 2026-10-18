"""Where the satellites are, as seen from a ground site: propagation, look geometry and passes.

Range acceleration is analytic: two-body gravity plus the centrifugal and Coriolis terms of the
rotating frame. J2 and drag are left out of the acceleration (well under 0.1 % of the range
acceleration), so the Doppler rate does not depend on the time step.

Passes are found on a coarse elevation grid and refined by bisection: rise (AOS), closest approach (TCA,
the range-rate zero crossing) and set (LOS) above the mask.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from sgp4.api import Satrec, SatrecArray

from . import tle
from .constants import MU_EARTH, OMEGA_EARTH, C
from .frames import teme_to_ecef
from .sites import SITES, Site

MASK_DEG = 10.0  # TR 38.821 Table 4.2-2 minimum elevation (service and feeder links)

# Survey window and grid step per constellation: long enough for many passes, step well below a pass.
SURVEY = {"starlink": (6.0, 10.0), "oneweb": (24.0, 20.0), "galileo": (24.0, 60.0)}  # (hours, step s)


def propagate(sats, jd, fr, dut1_s=0.0):
    """ECEF position (m) and velocity (m/s), shape (n_sat, n_t, 3); err (n_sat, n_t) != 0 marks SGP4 failures."""
    arr = sats if isinstance(sats, SatrecArray) else SatrecArray(list(sats))
    err, r, v = arr.sgp4(np.asarray(jd), np.asarray(fr))
    r_ecef, v_ecef = teme_to_ecef(r * 1e3, v * 1e3, jd + fr, dut1_s)
    return r_ecef, v_ecef, err


def propagate_at(sats, t_s):
    """ECEF states (n_sat, n_t, 3) at seconds from tle.T0."""
    jd, fr = tle.jd_split(np.atleast_1d(t_s))
    r, v, _ = propagate(sats, jd, fr)
    return r, v


def ecef_acceleration(r, v):
    """Two-body + rotating-frame acceleration in ECEF (m/s^2)."""
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    a = -MU_EARTH * r / rn**3
    w2 = OMEGA_EARTH**2
    a[..., 0] += w2 * r[..., 0] + 2.0 * OMEGA_EARTH * v[..., 1]
    a[..., 1] += w2 * r[..., 1] - 2.0 * OMEGA_EARTH * v[..., 0]
    return a


@dataclass
class Look:
    """Per-sample geometry; arrays share the (n_sat, n_t) shape of the inputs."""

    az_deg: np.ndarray
    el_deg: np.ndarray
    range_m: np.ndarray
    range_rate: np.ndarray  # m/s, positive = receding
    range_accel: np.ndarray  # m/s^2

    @property
    def delay_s(self):
        """One-way propagation delay (geometric, vacuum)."""
        return self.range_m / C

    def doppler_hz(self, fc_hz):
        """Doppler shift on a carrier fc received at the site (positive when approaching)."""
        return -fc_hz * self.range_rate / C

    def doppler_rate_hz_s(self, fc_hz):
        return -fc_hz * self.range_accel / C

    def doppler_ppm(self):
        return -1e6 * self.range_rate / C


def look(site: Site, r, v, accel=True) -> Look:
    d = r - site.ecef
    rng = np.linalg.norm(d, axis=-1)
    u = d / rng[..., None]
    enu = d @ site.enu.T
    el = np.rad2deg(np.arcsin(np.clip(enu[..., 2] / rng, -1.0, 1.0)))
    az = np.mod(np.rad2deg(np.arctan2(enu[..., 0], enu[..., 1])), 360.0)
    rdot = np.sum(u * v, axis=-1)
    if accel:
        a = ecef_acceleration(r, v)
        rdd = (np.sum(v * v, axis=-1) - rdot**2 + np.sum(d * a, axis=-1)) / rng
    else:
        rdd = np.full_like(rng, np.nan)
    return Look(az, el, rng, rdot, rdd)


def elevation_grid(fleet_sats, site: Site, jd, fr, chunk=500):
    """Elevation (deg, float32) for many satellites over a time grid, chunked to bound memory."""
    sats = list(fleet_sats)
    out = np.empty((len(sats), len(jd)), dtype=np.float32)
    for i in range(0, len(sats), chunk):
        r, v, err = propagate(sats[i : i + chunk], jd, fr)
        el = look(site, r, v, accel=False).el_deg
        el[err != 0] = -90.0
        out[i : i + chunk] = el
    return out


@dataclass
class Pass:
    sat: int  # index into the fleet
    name: str
    aos_s: float  # seconds from T0
    tca_s: float  # minimum range (= rate zero crossing)
    los_s: float
    max_el_deg: float

    @property
    def duration_s(self):
        return self.los_s - self.aos_s


def _geom(sat: Satrec, site: Site, t):
    jd, fr = tle.jd_split(np.atleast_1d(t))
    r, v, _ = propagate([sat], jd, fr)
    return look(site, r[0], v[0])


def _bisect(f, a, b, tol=0.01):
    fa = f(a)
    while b - a > tol:
        m = 0.5 * (a + b)
        fm = f(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def find_passes(fleet, site: Site, t_s, el_grid, mask_deg=MASK_DEG) -> list[Pass]:
    """Refine passes found on a coarse elevation grid (n_sat, n_t). Passes cut by the window edge are dropped."""
    above = el_grid >= mask_deg
    out = []
    for i in np.flatnonzero(above.any(axis=1)):
        sat = fleet.sats[i]
        edges = np.diff(above[i].astype(np.int8))
        rises, sets = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for k in rises:
            later = sets[sets > k]
            if not len(later):
                break
            j = later[0]

            def el(t, sat=sat):
                return _geom(sat, site, t).el_deg[0] - mask_deg

            def rdot(t, sat=sat):
                return _geom(sat, site, t).range_rate[0]

            aos = _bisect(el, t_s[k], t_s[k + 1])
            los = _bisect(lambda t: -el(t), t_s[j], t_s[j + 1])
            tca = _bisect(rdot, aos, los)
            out.append(Pass(int(i), fleet.names[i], aos, tca, los, float(el(tca) + mask_deg)))
    return sorted(out, key=lambda p: p.aos_s)


def pass_track(sat: Satrec, site: Site, p: Pass, step_s=1.0):
    """Dense geometry over one pass: (t_s, Look)."""
    t = np.arange(p.aos_s, p.los_s, step_s)
    return t, _geom(sat, site, t)


@cache
def leo_passes(key="starlink", site_key="sophia", hours=3.0, step=10.0):
    """Fleet and refined passes over a site (cached: several analyses draw on the same survey)."""
    site = SITES[site_key]
    f = tle.load(key)
    t, jd, fr = tle.time_grid(hours * 3600.0, step)
    el = elevation_grid(f.sats, site, jd, fr)
    return f, find_passes(f, site, t, el, MASK_DEG)


def survey_passes(key, site_key="sophia"):
    """leo_passes over the constellation's SURVEY window."""
    return leo_passes(key, site_key, *SURVEY[key])


def nearest_pass(passes, max_el_deg):
    return min(passes, key=lambda p: abs(p.max_el_deg - max_el_deg))


PASS_EL = {"overhead": 90.0, "low": 30.0}  # representative passes: maximum elevation nearest these


def representative_passes(key, site_key="sophia", step_s=1.0):
    """Dense geometry of the overhead and the low pass of a LEO constellation over the site:
    {tag: (Pass, t_s, Look)}."""
    f, ps = survey_passes(key, site_key)
    out = {}
    for tag, target in PASS_EL.items():
        p = nearest_pass(ps, target)
        t, lk = pass_track(f.sats[p.sat], SITES[site_key], p, step_s)
        out[tag] = (p, t, lk)
    return out


@dataclass
class States:
    t_s: np.ndarray
    r: np.ndarray  # (m, 3) ECEF m
    v: np.ndarray  # (m, 3) ECEF m/s
    label: str


def pass_states(f, p: Pass, t_s) -> States:
    """One satellite's states at the given instants of a pass."""
    r, v = propagate_at([f.sats[p.sat]], t_s)
    return States(np.asarray(t_s), r[0], v[0], p.name)


def leo_pass_states(max_el_deg=45.0, window_s=240.0, n=9, key="starlink", site_key="sophia") -> States:
    """n instants spread over window_s, centred on the closest approach of the pass whose maximum
    elevation is nearest max_el_deg (clipped to the pass)."""
    f, ps = leo_passes(key, site_key)
    p = nearest_pass(ps, max_el_deg)
    half = min(window_s / 2.0, (p.los_s - p.aos_s) / 2.0 - 1.0, p.tca_s - p.aos_s - 1.0, p.los_s - p.tca_s - 1.0)
    t = p.tca_s + np.linspace(-half, half, n)
    st = pass_states(f, p, t)
    st.label = f"{p.name}, max elevation {p.max_el_deg:.1f} deg"
    return st


def meo_burst_states(t_s=40000.0, key="galileo", site_key="sophia", max_sats=None) -> States:
    """All satellites above the mask at one instant (one burst). max_sats keeps the highest ones."""
    site = SITES[site_key]
    f = tle.load(key)
    r, v = propagate_at(f.sats, [t_s])
    r, v = r[:, 0], v[:, 0]
    el = look(site, r, v, accel=False).el_deg
    idx = np.argsort(-el)
    idx = idx[el[idx] >= MASK_DEG]
    if max_sats:
        idx = idx[:max_sats]
    return States(np.full(len(idx), t_s), r[idx], v[idx], f"{len(idx)} {f.constellation.name} satellites")

"""Reference frames: TEME (SGP4 output) -> Earth-fixed, geodetic -> ECEF, ECEF -> ENU.

TEME -> PEF uses GMST (IAU-82), the convention SGP4 output assumes. UT1 is taken equal to UTC
by default (pass ``dut1_s`` = UT1 - UTC otherwise) and polar motion is neglected (PEF ~ ITRF).
UT1 - UTC stays within 0.9 s; 0.1 s of it rotates a LEO satellite ~50 m in the Earth-fixed frame.
"""

import numpy as np

from .constants import OMEGA_EARTH, WGS84_A, WGS84_F


def gmst82(jd_ut1):
    """Greenwich mean sidereal time (IAU-82) in rad, for a Julian date (UT1). Vectorised."""
    t = (np.asarray(jd_ut1, dtype=float) - 2451545.0) / 36525.0
    sec = -6.2e-6 * t**3 + 0.093104 * t**2 + (876600.0 * 3600.0 + 8640184.812866) * t + 67310.54841
    return np.mod(np.deg2rad(sec / 240.0), 2.0 * np.pi)


def teme_to_ecef(r_teme, v_teme, jd_utc, dut1_s=0.0):
    """Rotate TEME position/velocity (..., 3) to Earth-fixed. jd_utc broadcasts over the leading axes."""
    th = gmst82(jd_utc + dut1_s / 86400.0)
    c, s = np.cos(th), np.sin(th)
    x, y, z = r_teme[..., 0], r_teme[..., 1], r_teme[..., 2]
    r = np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)
    vx, vy, vz = v_teme[..., 0], v_teme[..., 1], v_teme[..., 2]
    v = np.stack([c * vx + s * vy, -s * vx + c * vy, vz], axis=-1)
    # v_ecef = R v_teme - omega x r_ecef, omega = (0, 0, OMEGA_EARTH)
    v[..., 0] += OMEGA_EARTH * r[..., 1]
    v[..., 1] -= OMEGA_EARTH * r[..., 0]
    return r, v


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_m) * np.cos(lat) * np.cos(lon),
            (n + alt_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + alt_m) * np.sin(lat),
        ]
    )


def enu_matrix(lat_deg, lon_deg):
    """Rows are the local East, North, Up unit vectors in ECEF (geodetic vertical)."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )

"""Locating a ground transmitter from delay and Doppler measured at moving satellites.

Three measurement sets share the same model:

- ``rel18``:   NR NTN network verification (TS 38.305 V18.8.0 4.3.11): round-trip time and Doppler from a
               single satellite at several instants. The only nuisance is the satellite's carrier error.
- ``leosar``:  Doppler curve from one satellite pass (LEOSAR / Argos). The transmitter carrier is unknown,
               and optionally its linear drift.
- ``meosar``:  one burst received by several satellites (MEOSAR / MEOLUT): time and frequency of arrival,
               with the emission time and carrier unknown, which is the same as using their differences
               (TDOA / FDOA).

The transmitter is at a known height, so the position has two unknowns, solved as north/east offsets
in km from a reference point. Unknowns are scaled (km, us, Hz) and residuals whitened by the noise sigmas,
so the solver works on a well-conditioned problem.

Approximations shared by the simulator and the estimator: no light time (the satellite state is taken
at the reception instant, ~300 m of satellite motion at MEO), straight-line propagation in the Earth-fixed
frame (Sagnac effect, tens of metres), vacuum propagation, and no bent-pipe relay to a ground station.

Whether a claimed position is consistent with the measurements is in ``integrity``.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import C
from .frames import geodetic_to_ecef
from .sites import Site

R_M_KM = 6371.0  # metres-per-radian scale for the north/east parametrisation (exact mapping, any R works)

KINDS = ("rtt", "toa", "doppler")


@dataclass
class Measurements:
    """Satellite states and what is measured from each, one row per observation.

    kind: "rtt" (2 rho / c, s), "toa" (t_e + rho / c, s) or "doppler" (-fc rho_dot / c + df + df' t, Hz).
    t is each row's time from a reference instant (s), which only matters for the carrier drift df'.
    """

    kind: np.ndarray  # (m,) str
    r: np.ndarray  # (m, 3) satellite ECEF position, m
    v: np.ndarray  # (m, 3) satellite ECEF velocity, m/s
    sigma: np.ndarray  # (m,) noise std in the measurement unit (s or Hz)
    fc: float  # carrier, Hz
    clock_unknown: bool = False  # emission time t_e unknown (TOA rows)
    carrier_unknown: bool = False  # carrier offset df unknown (Doppler rows)
    drift_unknown: bool = False  # carrier drift df' unknown (Doppler rows, linear in t)
    t: np.ndarray | None = field(default=None, repr=False)
    y: np.ndarray | None = field(default=None, repr=False)

    def __len__(self):
        return len(self.kind)

    @property
    def nuisance(self):
        return int(self.clock_unknown) + int(self.carrier_unknown) + int(self.drift_unknown)


def _pairs(a, b, m):
    """Interleave two per-instant quantities (scalars or length-m arrays) into 2m rows."""
    return np.column_stack([np.broadcast_to(a, m), np.broadcast_to(b, m)]).ravel().astype(float)


def rel18(r, v, sigma_rtt_s, sigma_dop_hz, fc, carrier_unknown=True, t=None):
    """RTT and Doppler at each instant from one satellite (rows interleaved).

    carrier_unknown: the satellite access node's own frequency error (TS 38.108: 0.05 ppm) offsets every
    Doppler row by the same amount, so it is estimated with the position."""
    m = len(r)
    return Measurements(
        np.array(["rtt", "doppler"] * m),
        np.repeat(r, 2, axis=0),
        np.repeat(v, 2, axis=0),
        _pairs(sigma_rtt_s, sigma_dop_hz, m),
        fc,
        carrier_unknown=carrier_unknown,
        t=None if t is None else np.repeat(np.asarray(t, dtype=float), 2),
    )


def leosar(r, v, sigma_dop_hz, fc, t=None, drift_unknown=False):
    """Doppler of each burst from one satellite. drift_unknown adds a linear beacon-frequency drift (needs t)."""
    m = len(r)
    if drift_unknown and t is None:
        raise ValueError("a drift nuisance needs the burst times")
    return Measurements(
        np.array(["doppler"] * m),
        r,
        v,
        np.broadcast_to(sigma_dop_hz, m).astype(float),
        fc,
        carrier_unknown=True,
        drift_unknown=drift_unknown,
        t=None if t is None else np.asarray(t, dtype=float),
    )


def meosar(r, v, sigma_toa_s, sigma_foa_hz, fc):
    """One burst through several satellites: TOA and FOA per satellite, sigmas per satellite or scalar."""
    m = len(r)
    return Measurements(
        np.array(["toa", "doppler"] * m),
        np.repeat(r, 2, axis=0),
        np.repeat(v, 2, axis=0),
        _pairs(sigma_toa_s, sigma_foa_hz, m),
        fc,
        clock_unknown=True,
        carrier_unknown=True,
    )


@dataclass(frozen=True)
class Geo:
    """Transmitter position as north/east offsets (km) from a reference point, at a fixed height."""

    lat0_deg: float
    lon0_deg: float
    h_m: float

    @classmethod
    def at(cls, site: Site):
        return cls(site.lat_deg, site.lon_deg, site.alt_m)

    def latlon(self, ne_km):
        lat = self.lat0_deg + np.rad2deg(ne_km[0] / R_M_KM)
        lon = self.lon0_deg + np.rad2deg(ne_km[1] / (R_M_KM * np.cos(np.deg2rad(self.lat0_deg))))
        return lat, lon

    def ecef(self, ne_km):
        return geodetic_to_ecef(*self.latlon(ne_km), self.h_m)

    def jac(self, ne_km, d=1e-4):
        """d(ECEF m)/d(north km, east km), central differences (the mapping is smooth)."""
        cols = []
        for i in range(2):
            e = np.zeros(2)
            e[i] = d
            cols.append((self.ecef(ne_km + e) - self.ecef(ne_km - e)) / (2 * d))
        return np.stack(cols, axis=1)

    def offset_of(self, lat_deg, lon_deg):
        """Inverse of latlon: (north, east) km of a point relative to the reference."""
        return np.array(
            [
                np.deg2rad(lat_deg - self.lat0_deg) * R_M_KM,
                np.deg2rad(lon_deg - self.lon0_deg) * R_M_KM * np.cos(np.deg2rad(self.lat0_deg)),
            ]
        )


# Parameter vector theta = [north_km, east_km, (t_e in us), (df in Hz), (df' in Hz/s)].
US = 1e-6


def _split(meas: Measurements, theta):
    i = 2
    te = df = dd = 0.0
    if meas.clock_unknown:
        te = theta[i] * US
        i += 1
    if meas.carrier_unknown:
        df = theta[i]
        i += 1
    if meas.drift_unknown:
        dd = theta[i]
    return te, df, dd


def predict(meas: Measurements, geo: Geo, theta):
    p = geo.ecef(theta[:2])
    d = meas.r - p
    rho = np.linalg.norm(d, axis=1)
    rdot = np.sum(d * meas.v, axis=1) / rho
    te, df, dd = _split(meas, theta)
    t = 0.0 if meas.t is None else meas.t
    return np.where(
        meas.kind == "rtt",
        2 * rho / C,
        np.where(meas.kind == "toa", te + rho / C, -meas.fc * rdot / C + df + dd * t),
    )


def jacobian(meas: Measurements, geo: Geo, theta):
    p = geo.ecef(theta[:2])
    d = meas.r - p
    rho = np.linalg.norm(d, axis=1)
    u = d / rho[:, None]
    udotv = np.sum(u * meas.v, axis=1)
    drho_dp = -u
    drdot_dp = -(meas.v - u * udotv[:, None]) / rho[:, None]
    dp = geo.jac(theta[:2])  # (3, 2)
    jr = drho_dp @ dp
    jrd = drdot_dp @ dp
    pos = np.where(
        (meas.kind == "rtt")[:, None],
        2 * jr / C,
        np.where((meas.kind == "toa")[:, None], jr / C, -meas.fc * jrd / C),
    )
    cols = [pos]
    if meas.clock_unknown:
        cols.append(((meas.kind == "toa") * US)[:, None])
    if meas.carrier_unknown:
        cols.append((meas.kind == "doppler").astype(float)[:, None])
    if meas.drift_unknown:
        cols.append(((meas.kind == "doppler") * meas.t)[:, None])
    return np.hstack(cols)


def n_params(meas: Measurements):
    return 2 + meas.nuisance


def simulate(meas: Measurements, geo: Geo, theta_true, rng=None):
    """Return a copy of meas with y = model(theta_true) + Gaussian noise (noise-free when rng is None)."""
    y = predict(meas, geo, theta_true)
    if rng is not None:
        y = y + rng.normal(0.0, meas.sigma)
    return Measurements(**{**meas.__dict__, "y": y})


@dataclass
class Fit:
    theta: np.ndarray
    cost: float  # whitened sum of squared residuals
    iterations: int
    converged: bool

    @property
    def ne_km(self):
        return self.theta[:2]


def solve(meas: Measurements, geo: Geo, theta0, max_iter=100, tol_km=1e-7) -> Fit:
    """Levenberg-Marquardt on whitened residuals."""
    w = 1.0 / meas.sigma
    th = np.asarray(theta0, dtype=float).copy()
    r = (meas.y - predict(meas, geo, th)) * w
    cost = float(r @ r)
    lam = 1e-3
    for it in range(1, max_iter + 1):
        J = jacobian(meas, geo, th) * w[:, None]
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
            r_new = (meas.y - predict(meas, geo, th + step)) * w
            c_new = float(r_new @ r_new)
            if c_new <= cost or lam > 1e12:
                break
            lam *= 10.0
        if c_new > cost:
            return Fit(th, cost, it, False)
        th, r, cost = th + step, r_new, c_new
        lam = max(lam / 10.0, 1e-9)
        if np.linalg.norm(step[:2]) < tol_km:
            return Fit(th, cost, it, True)
    return Fit(th, cost, max_iter, False)


def nuisance_init(meas: Measurements, geo: Geo, ne_km):
    """Least-squares nuisance parameters with the position held at ne_km (they enter linearly)."""
    th = np.concatenate([ne_km, np.zeros(meas.nuisance)])
    if not meas.nuisance:
        return th
    w = 1.0 / meas.sigma
    J = jacobian(meas, geo, th)[:, 2:] * w[:, None]
    r = (meas.y - predict(meas, geo, th)) * w
    th[2:] = np.linalg.lstsq(J, r, rcond=None)[0]
    return th


def crb(meas: Measurements, geo: Geo, theta_true):
    """Cramer-Rao bound (inverse Fisher information) on theta for Gaussian noise."""
    J = jacobian(meas, geo, theta_true) / meas.sigma[:, None]
    return np.linalg.inv(J.T @ J)


def position_information(meas: Measurements, geo: Geo, theta):
    """Fisher information on (north, east) km with the nuisance parameters projected out (Schur complement).
    Information from independent bursts, each with its own nuisances, adds."""
    J = jacobian(meas, geo, theta) / meas.sigma[:, None]
    F = J.T @ J
    if not meas.nuisance:
        return F
    A, B, D = F[:2, :2], F[:2, 2:], F[2:, 2:]
    return A - B @ np.linalg.solve(D, B.T)


def horizontal_error_km(ne_est, ne_true):
    return float(np.hypot(*(np.asarray(ne_est) - np.asarray(ne_true))))


def sub_satellite_offset(states, geo: Geo):
    """North/east (km) of the sub-satellite point at the middle instant of `states`, relative to geo."""
    x = states.r[len(states.r) // 2]
    lat = np.degrees(np.arctan2(x[2], np.hypot(x[0], x[1])))  # geocentric is fine at this scale
    lon = np.degrees(np.arctan2(x[1], x[0]))
    return geo.offset_of(lat, lon)


def _radii(P, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.zeros(2), P, n)
    return np.hypot(x[:, 0], x[:, 1])


def p_within(P, radius_km, n=20000, seed=0):
    """Probability that a zero-mean 2D Gaussian error with covariance P falls inside radius_km."""
    return float(np.mean(_radii(P, n, seed) <= radius_km))


def radius_km(P, q, n=20000, seed=0):
    """Radius containing a fraction q of a zero-mean 2D Gaussian error with covariance P."""
    return float(np.quantile(_radii(P, n, seed), q))

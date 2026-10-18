"""Rel-17 NR NTN uplink timing: timing advance, broadcast assistance information, K_offset, HARQ.

TS 38.211 4.3.1: T_TA = (N_TA + N_TA,offset + N_TA,adj^common + N_TA,adj^UE) T_c.
- N_TA,adj^UE: the service-link round trip, computed by the UE from its position and the broadcast
  satellite ephemeris (SIB19 ephemerisInfo).
- N_TA,adj^common: twice Delay_common(t) from ta-Common / ta-CommonDrift / ta-CommonDriftVariant
  (TS 38.213 4.2), i.e. the round trip between the satellite and the uplink time synchronisation
  reference point (RP). RP at the gNB/gateway: the feeder-link round trip. RP at the satellite: 0, and
  the feeder round trip goes into kmac instead.
TS 38.321: UE-gNB RTT = timing advance + kmac.

N_TA (closed-loop TA commands) and N_TA,offset (0 or 13 us depending on duplexing) are left out.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from . import nr, tle
from .analytic import slant_range
from .constants import MU_EARTH, OMEGA_EARTH
from .constants import C as C_LIGHT

EPOCH_STEP_S = 10.0  # SIB19 broadcast epochs analysed along a pass
MAX_DT_S = 900.0  # largest ntn-UlSyncValidityDuration


@dataclass(frozen=True)
class TAInfo:
    ta_common: int
    ta_common_drift: int
    ta_common_drift_variant: int
    variant_clamped: bool  # the wanted variant was negative and had to be sent as 0

    def in_range(self):
        return (
            nr.TA_COMMON_RANGE[0] <= self.ta_common <= nr.TA_COMMON_RANGE[1]
            and nr.TA_DRIFT_RANGE[0] <= self.ta_common_drift <= nr.TA_DRIFT_RANGE[1]
            and nr.TA_DRIFT_VAR_RANGE[0] <= self.ta_common_drift_variant <= nr.TA_DRIFT_VAR_RANGE[1]
        )


def encode_ta_info(d_m, d_dot, d_ddot) -> TAInfo:
    """Fields for a one-way RP distance d (m) and its derivatives at the epoch.

    Delay_common(t) = TAC/2 + Drift/2 dt + Var/2 dt^2 (TS 38.213 4.2) must equal d(t)/c, so
    TAC = 2d/c, Drift = 2 d'/c and Var = d''/c (the second-order Taylor term d''/2 dt^2 over c, doubled).
    """
    var = d_ddot / C_LIGHT / nr.TA_DRIFT_VAR_LSB
    return TAInfo(
        round(2 * d_m / C_LIGHT / nr.TA_COMMON_LSB_S),
        round(2 * d_dot / C_LIGHT / nr.TA_DRIFT_LSB),
        max(0, round(var)),
        var < -0.5,
    )


def fit_ta_info(dt_s, d_m) -> TAInfo:
    """Least-squares TA-Info over a window instead of a Taylor expansion at the epoch.

    2 Delay_common(dt) = TAC + Drift dt + Var dt^2 is fitted to 2 d(dt) / c over the samples the network
    wants the fields to cover, with Var >= 0 (the ASN.1 range of ta-CommonDriftVariant), then quantised.
    """
    dt = np.asarray(dt_s, dtype=float)
    y = 2 * np.asarray(d_m, dtype=float) / C_LIGHT
    A = np.stack([np.ones_like(dt), dt, dt**2], axis=1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    clamped = coef[2] < 0
    if clamped:
        coef = np.append(np.linalg.lstsq(A[:, :2], y, rcond=None)[0], 0.0)
    return TAInfo(
        round(coef[0] / nr.TA_COMMON_LSB_S),
        round(coef[1] / nr.TA_DRIFT_LSB),
        round(coef[2] / nr.TA_DRIFT_VAR_LSB),
        bool(clamped),
    )


def delay_common_s(info: TAInfo, dt_s):
    """TS 38.213 4.2: Delay_common(t), one way, seconds."""
    tac = info.ta_common * nr.TA_COMMON_LSB_S
    drift = info.ta_common_drift * nr.TA_DRIFT_LSB
    var = info.ta_common_drift_variant * nr.TA_DRIFT_VAR_LSB
    dt = np.asarray(dt_s, dtype=float)
    return tac / 2 + drift / 2 * dt + var / 2 * dt**2


def quantize_state(r, v):
    """SIB19 positionVelocity: ECEF state rounded to 1.3 m / 0.06 m/s steps; also returns the codes."""
    rc = np.round(np.asarray(r) / nr.POS_LSB_M).astype(np.int64)
    vc = np.round(np.asarray(v) / nr.VEL_LSB_MS).astype(np.int64)
    ok = np.all((rc >= nr.POS_RANGE[0]) & (rc <= nr.POS_RANGE[1])) and np.all(
        (vc >= nr.VEL_RANGE[0]) & (vc <= nr.VEL_RANGE[1])
    )
    return rc * nr.POS_LSB_M, vc * nr.VEL_LSB_MS, bool(ok)


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def propagate_constant_velocity(r0, v0, dt_s):
    """Straight-line extrapolation of the ECEF state (the simplest UE implementation)."""
    dt = np.asarray(dt_s, dtype=float)[:, None]
    return r0 + v0 * dt, np.repeat(v0[None, :], len(dt), axis=0)


def propagate_two_body(r0, v0, dt_s):
    """Two-body (Keplerian) propagation of an ECEF state.

    ECI and ECEF coincide at the epoch (TS 38.331 EphemerisInfo), so r_eci = r0 and
    v_eci = v0 + omega x r0. Integrate in ECI, then rotate back by -omega dt.
    """
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    y0 = np.concatenate([r0, v0 + np.cross(w, r0)])

    def f(_, y):
        r = y[:3]
        return np.concatenate([y[3:], -MU_EARTH * r / np.linalg.norm(r) ** 3])

    dt = np.asarray(dt_s, dtype=float)
    sol = solve_ivp(f, (0.0, float(dt.max())), y0, t_eval=dt, method="DOP853", rtol=1e-12, atol=1e-6)
    r_eci, v_eci = sol.y[:3].T, sol.y[3:].T
    r = np.empty_like(r_eci)
    v = np.empty_like(v_eci)
    for k, t in enumerate(dt):
        R = _rot_z(-OMEGA_EARTH * t)
        r[k] = R @ r_eci[k]
        v[k] = R @ (v_eci[k] - np.cross(w, r_eci[k]))
    return r, v


PROPAGATORS = {"constant-velocity": propagate_constant_velocity, "two-body": propagate_two_body}


def los_range_and_rate(site_ecef, r, v):
    d = r - site_ecef
    rho = np.linalg.norm(d, axis=-1)
    return rho, np.sum(d * v, axis=-1) / rho


def slots_15khz(t_s):
    """Smallest whole number of 15 kHz slots (1 ms) covering t."""
    return int(np.ceil(np.max(t_s) / 1e-3 - 1e-9))


def harq_processes_needed(rtt_s, mu, t_proc_s=0.0):
    """ceil((RTT + T_proc) / T_slot). T_proc = 0 is the TR 38.821 6.1.1.3 simulation assumption."""
    return np.ceil((np.asarray(rtt_s) + t_proc_s) / nr.slot_s(mu) - 1e-9).astype(int)


def harq_duty_cycle(rtt_s, mu, n_processes, t_proc_s=0.0):
    """Fraction of slots a single UE can fill with HARQ feedback enabled: each process is reusable one
    cycle (RTT + T_proc) after it was used, the same cycle harq_processes_needed counts."""
    return np.minimum(1.0, n_processes * nr.slot_s(mu) / (np.asarray(rtt_s) + t_proc_s))


def staleness(t, r, v, lu, lg, k, ue_ecef, fc_hz):
    """TA error (s) and Doppler error (Hz at fc_hz) against time since epoch, per broadcast epoch (every
    EPOCH_STEP_S samples of a 1 s track) and UE propagation model. lu / lg: UE and gateway looks, k: the
    samples where both see the satellite.

    Returns dict model -> dict(dt, ta_ue, ta_common, ta_total, doppler) with arrays (n_epochs, n_dt),
    NaN where the satellite is no longer visible from both ends.
    """
    rho_ue, rdot_ue = lu.range_m[k], lu.range_rate[k]
    rho_gw, rdot_gw, rdd_gw = lg.range_m[k], lg.range_rate[k], lg.range_accel[k]
    dt_grid = np.arange(1.0, MAX_DT_S + 1.0, 1.0)
    epochs = np.arange(0, len(t), int(EPOCH_STEP_S))
    out = {}
    for name, prop in PROPAGATORS.items():
        res = {
            key: np.full((len(epochs), len(dt_grid)), np.nan) for key in ("ta_ue", "ta_common", "ta_total", "doppler")
        }
        for ei, e in enumerate(epochs):
            later = np.round(t[e:] - t[e], 6)
            m = np.isin(dt_grid, later)
            if not m.any():
                continue
            idx = e + np.searchsorted(later, dt_grid[m])
            rq, vq, _ = quantize_state(r[e], v[e])
            rp, vp = prop(rq, vq, dt_grid[m])
            rho_p, rdot_p = los_range_and_rate(ue_ecef, rp, vp)
            ta_ue = 2 * (rho_p - rho_ue[idx]) / C_LIGHT
            info = encode_ta_info(rho_gw[e], rdot_gw[e], rdd_gw[e])
            ta_c = 2 * delay_common_s(info, dt_grid[m]) - 2 * rho_gw[idx] / C_LIGHT
            res["ta_ue"][ei, m] = ta_ue
            res["ta_common"][ei, m] = ta_c
            res["ta_total"][ei, m] = ta_ue + ta_c
            res["doppler"][ei, m] = -(rdot_p - rdot_ue[idx]) / C_LIGHT * fc_hz
        res["dt"] = dt_grid
        out[name] = res
    return out


def validity(t, r, v, lu, lg, k, ue_ecef, limit):
    """Largest enumerated validity duration D such that, for every epoch, |TA error| <= limit over
    (0, D] while the satellite stays visible from both ends. TA error = UE part (ephemeris propagated
    by the UE) + common part (TA-Info). Common part either Taylor at the epoch or fitted over [0, D].
    Returns {(model, common_mode): D or None}."""
    rho_ue = lu.range_m[k]
    rho_gw, rdot_gw, rdd_gw = lg.range_m[k], lg.range_rate[k], lg.range_accel[k]
    epochs = np.arange(0, len(t), int(EPOCH_STEP_S))
    out = {}
    for name, prop in PROPAGATORS.items():
        ue_err = {}
        for e in epochs:
            later = np.round(t[e:] - t[e], 6)
            sel = (later > 0) & (later <= MAX_DT_S)
            if not sel.any():
                continue
            rq, vq, _ = quantize_state(r[e], v[e])
            rp, vp = prop(rq, vq, later[sel])
            ue_err[e] = (
                later[sel],
                e + np.flatnonzero(sel),
                2 * (los_range_and_rate(ue_ecef, rp, vp)[0] - rho_ue[e + np.flatnonzero(sel)]) / C_LIGHT,
            )
        for mode in ("taylor", "fit"):
            best = None
            for D in nr.UL_SYNC_VALIDITY_S:
                ok = True
                for e, (dt, idx, eu) in ue_err.items():
                    w = dt <= D
                    if not w.any():
                        continue
                    if mode == "taylor":
                        info = encode_ta_info(rho_gw[e], rdot_gw[e], rdd_gw[e])
                    else:
                        info = fit_ta_info(
                            np.concatenate([[0.0], dt[w]]), np.concatenate([[rho_gw[e]], rho_gw[idx[w]]])
                        )
                    ec = 2 * delay_common_s(info, dt[w]) - 2 * rho_gw[idx[w]] / C_LIGHT
                    if np.max(np.abs(eu[w] + ec)) > limit:
                        ok = False
                        break
                if not ok:
                    break
                best = D
            out[(name, mode)] = best
    return out


def max_validity(err, dt, limit):
    """Largest enumerated ntn-UlSyncValidityDuration for which |err| <= limit at every epoch."""
    worst = np.nanmax(np.abs(err), axis=0)
    ok = None
    for d in nr.UL_SYNC_VALIDITY_S:
        span = dt <= d
        cols = worst[span]
        if len(cols) and np.all(cols[~np.isnan(cols)] <= limit):
            ok = d
        else:
            break
    return ok


def cell_k_offset(fleet, mask_deg):
    """Cell-wide K_offset from the worst geometry: UE and gateway both at the mask, highest
    satellite in the fleet (spherical Earth; the UE-specific remainder goes in the differential MAC CE)."""
    h = max(tle.mean_altitude_km(s) for s in fleet.sats) * 1e3
    rtt = 2 * 2 * slant_range(h, mask_deg) / C_LIGHT
    return slots_15khz(rtt), float(rtt), h

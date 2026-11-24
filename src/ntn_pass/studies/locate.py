"""Where is the transmitter, and can a false claim be caught? Locating a transmitter at Sophia Antipolis
from satellite delay and Doppler, with the measurement noise from ``acquisition`` (NR) and ``beacon``
(406 MHz).

All three systems use the same solver:
- rel18: NR NTN network verification of a UE-reported location (TS 38.305 V18.8.0 4.3.11) from one LEO
  satellite: RTT and Doppler at several instants. The question is how far off a reported location must be
  before the network can tell, by direction, against TR 38.882's 5-10 km.
- meosar: one 406 MHz burst through up to four Galileo satellites (TOA + FOA), as a MEOLUT locates a
  distress beacon, against the T.019 5.6 location accuracy requirements.
- leosar: the Doppler curve of one LEO pass (bursts every ~50 s), with the beacon oscillator's drift.

Accuracy is evaluated from the Cramer-Rao bound of the solver, which tests/test_locate.py shows the solver
reaches, plus a Monte Carlo of the solver itself at one MEOSAR epoch.
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import beacon, cospas, orbit, rel18
from .. import integrity as I
from .. import link as lb
from .. import locate as L
from . import Output
from .plot import INK2, SERIES, save, style

PFA, PD = 0.01, 0.9
GEO = rel18.GEO


WINDOWS_S = (10.0, 20.0, 30.0, 60.0, 120.0, 240.0)
PASS_ELS = (25.0, 45.0, 70.0, 88.0)
DIRECTIONS = np.radians(np.arange(0.0, 360.0, 10.0))
RANGE_FLOOR_M = 10.0  # sensitivity case: troposphere / ionosphere / ephemeris bias treated as noise


def rel18_sweep():
    out = []
    for el in PASS_ELS:
        for w in WINDOWS_S:
            row = {"pass_max_el_deg": el, "window_s": w}
            for case, kw in {
                "ts38133": {},
                "ssb_only": {"ssb_only": True},
                "ts38133_floor10m": {"floor_m": RANGE_FLOOR_M},
            }.items():
                meas, st = rel18.measurements(el, w, **kw)
                th = np.array([0.0, 0.0, 0.0])
                P = np.linalg.inv(L.position_information(meas, GEO, th))
                d = np.array([I.min_detectable_km(meas, GEO, a, PFA, PD) for a in DIRECTIONS])
                row[case] = {
                    "instants": len(st.t_s),
                    "crb_semi_axes_km": np.sqrt(np.linalg.eigvalsh(P)).round(4).tolist(),
                    "min_detectable_km": {"best": round(float(d.min()), 3), "worst": round(float(d.max()), 3)},
                    "mirror": I.mirror_check(meas, GEO, st, PFA),
                    "_d": d,
                }
            out.append(row)
    return out


def meosar_day(step_s=1800.0, hours=24.0):
    view = cospas.GalileoView()
    epochs = np.arange(0.0, hours * 3600.0, step_s)
    out = {}
    for kind in ("fgb", "sgb"):
        for case in ("t019", "link"):
            res = {f"{t}s_{r}km": [] for t, r, _ in cospas.T019_TARGETS[kind]}
            r95 = []
            for t0 in epochs:
                F0, sel, _ = cospas.burst_information(view, t0, kind, case)
                cache = {}
                for t, rad, _ in cospas.T019_TARGETS[kind]:
                    if t not in cache:
                        F = F0.copy()
                        for tb in beacon.burst_times(kind, t)[1:]:
                            F = F + cospas.burst_information(view, t0 + tb, kind, case, subset=sel)[0]
                        cache[t] = np.linalg.inv(F)
                    res[f"{t}s_{rad}km"].append(L.p_within(cache[t], rad))
                r95.append(L.radius_km(np.linalg.inv(F0), 0.95))
            out[f"{kind}_{case}"] = {
                "fraction_within": {k: round(float(np.mean(v)), 4) for k, v in res.items()},
                "worst_epoch_fraction_within": {k: round(float(np.min(v)), 4) for k, v in res.items()},
                "single_burst_r95_km": {"median": round(float(np.median(r95)), 3), "max": round(float(np.max(r95)), 3)},
                "_r95": np.array(r95),
            }
    return epochs, out


def meosar_monte_carlo(t0=40000.0, kind="fgb", trials=500, seed=1):
    """Run the solver itself on one epoch (T.019-compliant noise) and compare its 5 km rate with the CRB."""
    view = cospas.GalileoView()
    F, sel, _ = cospas.burst_information(view, t0, kind, "t019")
    r, v, *_ = view.at(t0)
    m = L.meosar(r[sel], v[sel], *cospas.T019_SIGMA[kind], lb.F_UP)
    truth = np.array([0.0, 0.0, 3.0, 37.0])
    rng = np.random.default_rng(seed)
    err = []
    for _ in range(trials):
        s = L.simulate(m, GEO, truth, rng)
        fit = L.solve(s, GEO, L.nuisance_init(s, GEO, np.array([20.0, -20.0])))
        err.append(L.horizontal_error_km(fit.ne_km, truth[:2]))
    err = np.array(err)
    return {
        "epoch_s": t0,
        "satellites": len(sel),
        "solver_within_5km": round(float(np.mean(err <= 5.0)), 3),
        "crb_within_5km": round(L.p_within(np.linalg.inv(F), 5.0), 3),
        "solver_r95_km": round(float(np.quantile(err, 0.95)), 3),
        "crb_r95_km": round(L.radius_km(np.linalg.inv(F), 0.95), 3),
    }


def meosar_uniqueness(step_s=1800.0, hours=24.0, kind="fgb"):
    """Is one burst's fix unique? Noise-free data from the beacon, solver started from every used satellite's
    ground-track mirror and from rings 300 / 1000 / 3000 km out in 8 directions, at every epoch. Exact fits
    are compared with the beacon in ECEF (the north/east parametrisation wraps). Every other end point is a
    false minimum, scored by the noncentrality of the consistency test there."""
    view = cospas.GalileoView()
    truth = GEO.ecef(np.zeros(2))
    dirs = np.radians(range(0, 360, 45))
    ring = [d * np.array([np.cos(a), np.sin(a)]) for d in (300.0, 1000.0, 3000.0) for a in dirs]
    n_starts, at_beacon, false_cost = 0, 0, []
    for t0 in np.arange(0.0, hours * 3600.0, step_s):
        _, sel, _ = cospas.burst_information(view, t0, kind, "t019")
        r, v, *_ = view.at(t0)
        m = L.simulate(L.meosar(r[sel], v[sel], *cospas.T019_SIGMA[kind], lb.F_UP), GEO, np.zeros(4))
        one = [orbit.States(np.zeros(1), r[i : i + 1], v[i : i + 1], "") for i in sel]
        mirrors = [2 * L.sub_satellite_offset(st, GEO) for st in one]
        for x0 in mirrors + ring:
            fit = L.solve(m, GEO, L.nuisance_init(m, GEO, x0))
            n_starts += 1
            if np.linalg.norm(GEO.ecef(fit.ne_km) - truth) < 1.0:
                at_beacon += 1
            else:
                false_cost.append(fit.cost)
    lam = min(false_cost) if false_cost else None
    return {
        "starts": n_starts,
        "ended_at_beacon": at_beacon,
        "false_minima": len(false_cost),
        "lowest_false_noncentrality": None if lam is None else round(float(lam), 1),
        "pd_reject_lowest": None if lam is None else round(float(I.detection_pd(lam, I.dof(m), PFA)), 4),
    }


def leosar_cases():
    f, ps = orbit.leo_passes()
    sigma = cospas.leosar_sigma_hz()
    rng = np.random.default_rng(3)
    out = []
    for el in PASS_ELS:
        p = orbit.nearest_pass(ps, el)
        t = cospas.leosar_burst_times(p, rng)
        st = orbit.pass_states(f, p, t)
        row = {"pass_max_el_deg": round(p.max_el_deg, 1), "bursts": len(t), "sigma_foa_hz": round(sigma, 3)}
        for drift in (False, True):
            m = L.leosar(st.r, st.v, sigma, lb.F_UP, t=t - t.mean(), drift_unknown=drift)
            th = np.zeros(2 + m.nuisance)
            P = np.linalg.inv(L.position_information(m, GEO, th))
            row["drift_estimated" if drift else "drift_ignored"] = {
                "crb_semi_axes_km": np.sqrt(np.linalg.eigvalsh(P)).round(3).tolist(),
                "within_5km": round(L.p_within(P, 5.0), 3),
                "mirror": I.mirror_check(m, GEO, st, PFA),
            }
        # what an unmodelled drift does: bias from the T.001 mean slope (1e-9 per minute)
        m0 = L.leosar(st.r, st.v, sigma, lb.F_UP, t=t - t.mean())
        J = L.jacobian(m0, GEO, np.zeros(3)) / m0.sigma[:, None]
        dy = beacon.FGB_MEAN_SLOPE_PER_MIN / 60.0 * lb.F_UP * (t - t.mean()) / m0.sigma
        row["bias_km_if_drift_ignored"] = round(float(np.hypot(*np.linalg.lstsq(J, dy, rcond=None)[0][:2])), 3)
        out.append(row)
    return out


def _plot_rel18(sweep, path):
    fig = plt.figure(figsize=(11.5, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    rows = [r for r in sweep if r["pass_max_el_deg"] == 45.0]
    for k, r in enumerate(rows):
        d = r["ts38133"]["_d"]
        ax.plot(
            np.r_[DIRECTIONS, DIRECTIONS[0]],
            np.r_[d, d[0]] * 1e3,
            color=plt.cm.viridis(k / len(rows)),
            label=f"{r['window_s']:.0f} s",
        )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rscale("log")
    ax.set_rgrids([10, 30, 100, 300, 1000], labels=["10", "30", "100", "300", "1000 m"], fontsize=7, angle=22)
    ax.set_title(
        "Smallest detectable false offset (m) by direction\n45 deg pass, RTT at TS 38.133 accuracy", fontsize=10
    )
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.3, -0.15), fontsize=8, title="window")
    ax2 = fig.add_subplot(1, 2, 2)
    for k, el in enumerate(PASS_ELS):
        rs = [r for r in sweep if r["pass_max_el_deg"] == el]
        w = [r["window_s"] for r in rs]
        pd = [r["ts38133"]["mirror"]["pd"] for r in rs]
        km = rs[0]["ts38133"]["mirror"]["mirror_offset_km"]
        ax2.semilogx(
            w,
            pd,
            marker="o",
            markersize=3,
            color=plt.cm.plasma(k / len(PASS_ELS)),
            label=f"max el {el:.0f} deg (mirror {km:,.0f} km away)",
        )
    ax2.axhline(PD, color=INK2, linewidth=1, linestyle="--")
    ax2.set_xlabel("Observation window (s), one RTT + Doppler every 10 s")
    ax2.set_ylabel(f"Pd of the mirror claim (Pfa {PFA})")
    ax2.set_title("A claim mirrored across the ground track", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="center right")
    fig.text(
        0.01,
        0.01,
        "Starlink passes over Sophia, Set-1 S-band link, noise only. TR 38.882 counts 5-10 km as verified.",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path)


def _plot_meosar(epochs, mo, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = {
        "fgb_t019": (SERIES[0], "-"),
        "fgb_link": (SERIES[0], "--"),
        "sgb_t019": (SERIES[2], "-"),
        "sgb_link": (SERIES[2], "--"),
    }
    names = {
        "fgb_t019": "FGB, T.019-compliant TOA/FOA",
        "fgb_link": "FGB, link-limited",
        "sgb_t019": "SGB, T.019-compliant",
        "sgb_link": "SGB, link-limited",
    }
    for k, (c, ls) in styles.items():
        ax.semilogy(epochs / 3600.0, mo[k]["_r95"], color=c, linestyle=ls, label=names[k])
    ax.axhline(5.0, color=INK2, linewidth=1)
    ax.text(23.8, 5.5, "5 km (T.019 radius)", color=INK2, fontsize=8, ha="right")
    ax.set_ylim(0.03, 20)
    ax.set_xlabel("Hours from 2026-10-13 00:00 UTC")
    ax.set_ylabel("95 % error radius, one burst (km)")
    ax.set_title("Beacon at Sophia, MEOLUT at Toulouse, best 4 Galileo satellites", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fig.tight_layout()
    save(fig, path)


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def run(**kw):
    style()
    out = Output("locate", **kw)
    sweep = rel18_sweep()
    epochs, mo = meosar_day()
    res = {
        "rel18": {
            "sigma_rtt_ns_ts38133": round(rel18.SIGMA_RTT_S * 1e9, 2),
            "measurement_step_s": rel18.STEP_S,
            "pfa": PFA,
            "pd": PD,
            "sweep": sweep,
        },
        "meosar": {
            "targets": cospas.T019_TARGETS,
            "max_satellites": cospas.MAX_SATS,
            "results": mo,
            "solver_check": meosar_monte_carlo(),
            "uniqueness": meosar_uniqueness(),
        },
        "leosar": leosar_cases(),
    }
    _plot_rel18(sweep, out.figure("rel18"))
    _plot_meosar(epochs, mo, out.figure("meosar"))
    clean = _strip(res)
    out.write(clean)
    for r in sweep:
        a, b = r["ts38133"], r["ssb_only"]
        print(
            f"rel18 el {r['pass_max_el_deg']:4.0f} win {r['window_s']:5.0f}s: min detect {a['min_detectable_km']} km "
            f"(SSB-only {b['min_detectable_km']}), mirror {a['mirror']}"
        )
    for k, v in clean["meosar"]["results"].items():
        print("meosar", k, v["fraction_within"], v["single_burst_r95_km"])
    print("solver check", clean["meosar"]["solver_check"])
    print("uniqueness", clean["meosar"]["uniqueness"])
    for r in clean["leosar"]:
        print("leosar", r)
    return clean

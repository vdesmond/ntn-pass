"""How well can a distress beacon be heard and timed? The 406 MHz link from a beacon at Sophia Antipolis
through Galileo FOC repeaters to a MEOLUT at Toulouse, over 24 h.

For every minute, the satellites seen at >= 5 deg from both the beacon and the MEOLUT (T.019 Annex F),
their uplink / downlink / system C/N0 for three beacons (FGB T.019 nominal, FGB worst compliant, SGB
minimum EIRP) and the TOA / FOA Cramer-Rao bounds. Also the single-pass (LEOSAR-style) case on the
Starlink stand-in pass: the payload G/T it would need, and the beacon oscillator floor on its Doppler.
The T.019 / T.016 values are reproduced from their own inputs first.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from .. import beacon as B
from .. import cospas, orbit, tle
from .. import link as lb
from ..analytic import slant_range
from ..link import K_BOLTZMANN_DB, fspl_db
from . import Output
from .plot import INK2, SERIES, save, style

BEACONS = {"fgb_nominal": B.fgb(), "fgb_worst": B.FGB_WORST, "sgb_min": B.sgb_min()}


def reproduction():
    r_gal = slant_range(lb.GALILEO_ALT_M, 5.0)
    return {
        "slant_range_5deg_km": {
            k: {"tr": tr, "model": round(slant_range(h * 1e3, 5.0) / 1e3, 1)}
            for k, (h, tr) in {"galileo": (23222, 28350), "glonass": (19100, 24120), "gps": (20850, 25920)}.items()
        },
        "horizon_vs_zenith_path_db": {
            "t016": 1.9,
            "model": round(20 * np.log10(slant_range(lb.GALILEO_ALT_M, 0.0) / lb.GALILEO_ALT_M), 3),
        },
        "t019_a2_galileo_uplink_dbhz": {
            "t019": 45.7,
            "model": round(float(9.0 - fspl_db(28350e3, 406.034e6) - 3 - 2.5 - 12.7 - K_BOLTZMANN_DB), 2),
        },
        "t019_a1_galileo_fsl_db": {"t019": 185.4, "model": round(float(fspl_db(r_gal, lb.F_DN_GALILEO)), 2)},
        "t019_a1_galileo_ebn0_constant_db": {
            "t019": 23.7,
            "model": round(7.0 - 185.4 - 2.5 - K_BOLTZMANN_DB - 29.0 + 2.0, 2),
        },
    }


def meosar(hours=24.0, step=60.0):
    f = tle.load("galileo")
    t, jd, fr = tle.time_grid(hours * 3600.0, step)
    r, v, _ = orbit.propagate(f.sats, jd, fr)
    lk_b = orbit.look(cospas.BEACON_SITE, r, v, accel=False)
    lk_m = orbit.look(cospas.MEOLUT_SITE, r, v, accel=False)
    both = (lk_b.el_deg >= cospas.MASK_DEG) & (lk_m.el_deg >= cospas.MASK_DEG)
    up_gt = lb.sat_gt_db(lk_b.el_deg)
    dn = lb.cn0_dn_dbhz(lk_m.range_m)
    im = lb.cn0_im_dbhz()
    out = {"t": t, "both": both, "el_beacon": lk_b.el_deg, "el_meolut": lk_m.el_deg, "dn": dn, "sys": {}, "up": {}}
    for k, b in BEACONS.items():
        up = lb.cn0_up_dbhz(b.eirp_dbw(lk_b.el_deg), lk_b.range_m, up_gt)
        out["up"][k] = up
        out["sys"][k] = lb.cn0_sys_dbhz(up, dn, im)
    return out


def leosar():
    """Single LEO pass (Starlink stand-in): range at 5 deg, payload G/T needed for the FGB worst case to reach
    the T.019 sensitivity there, bursts every 50 s, and the beacon oscillator against the Doppler noise."""
    leo = orbit.leo_pass_states(45.0)
    h = float(np.median(np.linalg.norm(leo.r, axis=1))) - 6_371e3
    rng = slant_range(h, cospas.MASK_DEG)
    eirp = float(B.FGB_WORST.eirp_dbw(5.0))
    gt_needed = cospas.SENSITIVITY_DBHZ["fgb"] - (
        eirp - fspl_db(rng, lb.F_UP) - lb.POL_LOSS_DB - lb.FADING_DB - K_BOLTZMANN_DB
    )
    pass_min = 10.0
    return {
        "pass": leo.label,
        "altitude_km": round(h / 1e3, 1),
        "slant_range_5deg_km": round(rng / 1e3, 1),
        "payload_gt_needed_db": round(float(gt_needed), 2),
        "bursts_per_10_min_pass": int(pass_min * 60 // B.FGB_REPETITION_S),
        "crb_foa_hz_at_sensitivity": round(float(B.crb_at("fgb", cospas.SENSITIVITY_DBHZ["fgb"])[1]), 4),
        "oscillator_residual_hz": round(B.FGB_RESIDUAL_FREQ * lb.F_UP, 3),
        "oscillator_drift_over_pass_hz": round(B.FGB_MEAN_SLOPE_PER_MIN * pass_min * lb.F_UP, 2),
    }


def _plot(m, crb, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for key, c in zip(BEACONS, SERIES, strict=False):
        x = m["el_beacon"][m["both"]]
        y = m["sys"][key][m["both"]]
        ax.scatter(x[::7], y[::7], s=4, color=c, label=BEACONS[key].name, alpha=0.6, linewidths=0)
    for kind, ls in (("fgb", "--"), ("sgb", ":")):
        ax.axhline(cospas.SENSITIVITY_DBHZ[kind], color=INK2, linestyle=ls, linewidth=1)
        ax.text(
            6,
            cospas.SENSITIVITY_DBHZ[kind] + 0.2,
            f"T.019 sensitivity, {kind.upper()}",
            ha="left",
            fontsize=8,
            color=INK2,
        )
    ax.set_xlabel("Satellite elevation at the beacon (deg)")
    ax.set_ylabel("System C/N0 at the MEOLUT (dB-Hz)")
    ax.set_title("Galileo FOC, beacon Sophia, MEOLUT Toulouse, 24 h", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", markerscale=3)

    ax = axes[1]
    cn0 = np.linspace(28, 50, 100)
    for (key, label), c in zip(
        {
            "fgb_150us": "FGB, 150 us edges",
            "fgb_250us": "FGB, 250 us edges",
            "sgb": "SGB, half-sine chips, 90 kHz",
        }.items(),
        [SERIES[0], SERIES[0], SERIES[2]],
        strict=True,
    ):
        ax.semilogy(
            cn0,
            crb[key][0] * 10 ** (-cn0 / 20) * 1e6,
            color=c,
            linestyle="-" if key != "fgb_250us" else "--",
            label=label,
        )
    for kind, c in (("fgb", SERIES[0]), ("sgb", SERIES[2])):
        lo = cospas.SENSITIVITY_DBHZ[kind]
        ax.plot([lo, lo + 3], [cospas.T019_SIGMA[kind][0] * 1e6] * 2, color=c, linewidth=5, alpha=0.4)
    ax.text(36.5, 30, "T.019 required σ (FGB)", fontsize=8, color=INK2)
    ax.text(30.6, 1.25, "T.019 required σ (SGB, provisional)", fontsize=8, color=INK2)
    ax.set_xlabel("C/N0 (dB-Hz)")
    ax.set_ylabel("TOA standard deviation, one burst (µs)")
    ax.set_title("Cramér-Rao bound on TOA against T.019", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, path)


def run(**kw):
    style()
    out = Output("beacon", **kw)
    crb = B.unit_crb()
    m = meosar()
    both = m["both"]
    n_vis = both.sum(axis=0)
    res = {
        "sites": {"beacon": cospas.BEACON_SITE.name, "meolut": cospas.MEOLUT_SITE.name, "mask_deg": cospas.MASK_DEG},
        "reproduction": reproduction(),
        "meolut_gt_db": lb.GT_MEOLUT_DB,
        "cn0_im_dbhz": round(float(lb.cn0_im_dbhz()), 1),
        "satellites_visible_both": {
            "min": int(n_vis.min()),
            "mean": round(float(n_vis.mean()), 2),
            "max": int(n_vis.max()),
        },
        "cn0_dn_dbhz": [round(float(m["dn"][both].min()), 2), round(float(m["dn"][both].max()), 2)],
        "beacons": {},
        "crb_at_0_dbhz": {
            k: {"toa_s": v[0], "foa_hz": v[1], "rms_bandwidth_hz": round(v[2], 1)} for k, v in crb.items()
        },
        "t019_required_sigma": {k: {"toa_us": v[0] * 1e6, "foa_hz": v[1]} for k, v in cospas.T019_SIGMA.items()},
        "satellite_floors": {
            "toa_ns": round(cospas.TOA_FLOOR_S * 1e9, 1),
            "foa_mhz": round(cospas.FOA_FLOOR_HZ * 1e3, 2),
            "uncalibrated_group_delay_slope_at_1200hz_us": round(cospas.GROUP_DELAY_SLOPE_S_PER_HZ * 1200 * 1e6, 2),
        },
        "leosar": leosar(),
    }
    for key, b in BEACONS.items():
        sysv = m["sys"][key][both]
        upv = m["up"][key][both]
        w = "fgb_150us" if b.kind == "fgb" else "sgb"
        toa = crb[w][0] * 10 ** (-sysv / 20)
        foa = crb[w][1] * 10 ** (-sysv / 20)
        res["beacons"][key] = {
            "name": b.name,
            "cn0_up_dbhz": [round(float(upv.min()), 2), round(float(upv.max()), 2)],
            "cn0_sys_dbhz": [round(float(sysv.min()), 2), round(float(sysv.max()), 2)],
            "share_below_sensitivity": round(float(np.mean(sysv < cospas.SENSITIVITY_DBHZ[b.kind])), 4),
            "crb_toa_us": [round(float(toa.min() * 1e6), 3), round(float(toa.max() * 1e6), 3)],
            "crb_foa_hz": [round(float(foa.min()), 4), round(float(foa.max()), 4)],
            "link_limited_toa_us": [
                round(float(cospas.link_limited_sigma(x, 0)[0] * 1e6), 3) for x in (toa.min(), toa.max())
            ],
            "link_limited_foa_hz": [
                round(float(cospas.link_limited_sigma(0, x)[1]), 4) for x in (foa.min(), foa.max())
            ],
            "meolut_gt_for_0.5db_downlink_cost_t019_fixed_share": round(cospas.gt_threshold_db(float(upv.max())), 1),
        }
    _plot(m, crb, out.figure("link"))
    out.write(res)
    print(json.dumps({k: res[k] for k in ("satellites_visible_both", "cn0_dn_dbhz", "beacons", "leosar")}, indent=1))
    return res

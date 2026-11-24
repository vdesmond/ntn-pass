"""How long does the broadcast timing assistance stay good? What the pass does to NR NTN uplink timing
(timing advance, SIB19 validity, K_offset) and, as an aside, to HARQ.

UE at Sophia Antipolis, transparent payload with its gateway at Toulouse (an assumed site), uplink
time synchronisation reference point (RP) at the gateway unless stated. Only samples where both the UE
and the gateway see the satellite at >= 10 deg are used (TR 38.821 Table 4.2-2 minimum elevation for the
service and feeder links).

For every broadcast epoch along a pass, the network sends (SIB19):
- the serving satellite's ECEF state, quantised to 1.3 m / 0.06 m/s, and
- TA-Info for the feeder link (ta-Common, ta-CommonDrift, ta-CommonDriftVariant).
The UE then computes its timing advance for later times from these alone. The error against the SGP4
truth is compared with two limits:
- Te_NTN (TS 38.133 Table 7.1C.2-1) bounds the UE's own transmit timing against the TA it computes. It
  is a budget the propagation error has to share with the UE implementation, not the propagation
  error's limit.
- The normal cyclic prefix is what the gNB receiver absorbs without inter-symbol interference.
The two-body residual here is also the frequency error left for the warm SSB search in ``acquisition``.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from .. import nr, orbit, tle
from .. import timing as T
from ..constants import C
from ..link import FC_S as FC
from ..orbit import MASK_DEG as MASK
from ..sites import SITES
from . import Output
from .plot import GRID, INK2, SERIES, save, style

GATEWAY = "toulouse"
MIN_PASS_EL = 20.0
N_PASSES = 12
CONSTELLATIONS = ("starlink", "oneweb")


def _visible_track(f, p, ue, gw, step=1.0):
    t = np.arange(p.aos_s, p.los_s, step)
    r, v = orbit.propagate_at([f.sats[p.sat]], t)
    r, v = r[0], v[0]
    lu, lg = orbit.look(ue, r, v), orbit.look(gw, r, v)
    k = (lu.el_deg >= MASK) & (lg.el_deg >= MASK)
    return t[k], r[k], v[k], lu, lg, k


def run(**kw):
    style()
    out = Output("timing", **kw)
    ue, gw = SITES["sophia"], SITES[GATEWAY]
    te, cp = nr.TE_NTN_S[(15, 15)], nr.cp_normal_s(0)
    results = dict(
        ue=ue.name, gateway=gw.name, mask_deg=MASK,
        limits_us=dict(te_ntn_15khz=round(te * 1e6, 4), cp_normal_15khz=round(cp * 1e6, 4),
                       te_ntn_30khz=round(nr.TE_NTN_S[(30, 30)] * 1e6, 4),
                       cp_normal_30khz=round(nr.cp_normal_s(1) * 1e6, 4)),
    )  # fmt: skip
    curves = {}
    harq_example = {}
    for key in CONSTELLATIONS:
        name = tle.CONSTELLATIONS[key].name
        f, ps = orbit.survey_passes(key)
        chosen = [p for p in ps if p.max_el_deg >= MIN_PASS_EL]
        chosen = chosen[:: max(1, len(chosen) // N_PASSES)][:N_PASSES]
        agg = {m: {k: [] for k in ("ta_ue", "ta_common", "ta_total", "doppler")} for m in T.PROPAGATORS}
        rtt_all, feeder_all, clamped, in_range, n_info = [], [], 0, True, 0
        joint = {}
        for p in chosen:
            t, r, v, lu, lg, k = _visible_track(f, p, ue, gw)
            if len(t) < 30:
                continue
            st = T.staleness(t, r, v, lu, lg, k, ue.ecef, FC)
            for lim_name, lim in (("te_ntn_15khz", te), ("cp_15khz", cp)):
                for key2, D in T.validity(t, r, v, lu, lg, k, ue.ecef, lim).items():
                    slot = f"{key2[0]} + common {key2[1]}"
                    prev = joint.setdefault(lim_name, {}).get(slot, 10**9)
                    joint[lim_name][slot] = min(prev, D if D is not None else 0)
            for m in st:
                for q in agg[m]:
                    agg[m][q].append(st[m][q])
            rtt_all.append(2 * (lu.range_m[k] + lg.range_m[k]) / C)
            feeder_all.append(2 * lg.range_m[k] / C)
            for e in range(0, len(t), int(T.EPOCH_STEP_S)):
                info = T.encode_ta_info(lg.range_m[k][e], lg.range_rate[k][e], lg.range_accel[k][e])
                clamped += info.variant_clamped
                in_range &= info.in_range()
                n_info += 1
                in_range &= T.quantize_state(r[e], v[e])[2]
            if not harq_example or p.max_el_deg > harq_example["max_el"]:
                harq_example = dict(
                    max_el=p.max_el_deg,
                    name=p.name,
                    t=t - p.tca_s,
                    rtt=2 * (lu.range_m[k] + lg.range_m[k]) / C,
                    key=name,
                )
        dt = st[next(iter(st))]["dt"]
        r_key = {"passes": len(rtt_all), "ta_info_in_range": bool(in_range),
                 "drift_variant_clamped": f"{clamped}/{n_info}", "validity_s": {}}  # fmt: skip
        curves[name] = {}
        for m in agg:
            ta = np.vstack(agg[m]["ta_total"])
            dop = np.vstack(agg[m]["doppler"])
            curves[name][m] = (dt, np.nanmax(np.abs(ta), axis=0), np.nanmedian(np.abs(ta), axis=0))
            r_key["validity_s"][m] = dict(
                te_ntn_15khz=T.max_validity(ta, dt, te),
                cp_15khz=T.max_validity(ta, dt, cp),
                ue_part_only_te=T.max_validity(np.vstack(agg[m]["ta_ue"]), dt, te),
                common_part_only_te=T.max_validity(np.vstack(agg[m]["ta_common"]), dt, te),
            )
            for s in (10, 60, 120):
                if s <= len(dt):
                    r_key["validity_s"][m][f"worst_ta_err_us_at_{s}s"] = round(
                        float(np.nanmax(np.abs(ta[:, s - 1]))) * 1e6, 4
                    )
                    r_key["validity_s"][m][f"worst_doppler_err_hz_at_{s}s"] = round(
                        float(np.nanmax(np.abs(dop[:, s - 1]))), 2
                    )
        rtt = np.concatenate(rtt_all)
        ko, ko_rtt, h = T.cell_k_offset(f, MASK)
        r_key["validity_joint_s"] = joint  # worst over passes, 0 = not even the shortest value holds
        r_key["rtt_ms"] = dict(min=round(float(rtt.min()) * 1e3, 3), max=round(float(rtt.max()) * 1e3, 3))
        r_key["k_offset"] = dict(
            cell_specific_slots=ko, cell_worst_rtt_ms=round(ko_rtt * 1e3, 3), highest_sat_km=round(h / 1e3, 1),
            ue_specific_max_slots=T.slots_15khz(rtt),
            rp_at_satellite_kmac_slots=T.slots_15khz(np.concatenate(feeder_all)),  # covers the feeder round trip
        )  # fmt: skip
        r_key["harq"] = {}
        for mu in (0, 1):
            n = T.harq_processes_needed(rtt, mu)
            r_key["harq"][f"scs_{15 * 2**mu}khz"] = dict(
                needed_max=int(n.max()), needed_median=int(np.median(n)),
                enough_16=round(float(np.mean(n <= 16)), 3), enough_32=round(float(np.mean(n <= 32)), 3),
                duty_16=round(float(np.mean(T.harq_duty_cycle(rtt, mu, 16))), 3),
                duty_32=round(float(np.mean(T.harq_duty_cycle(rtt, mu, 32))), 3),
            )  # fmt: skip
        results[key] = r_key
        print(name, json.dumps(r_key, indent=1))
    _plot_staleness(curves, te, cp, out.figure("ta-staleness"))
    _plot_harq(harq_example, out.figure("harq"))
    out.write(results)
    return results


def _plot_staleness(curves, te, cp, path):
    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4.2), sharey=True, squeeze=False)
    for j, (name, cm) in enumerate(curves.items()):
        ax = axes[0, j]
        for i, (m, (dt, worst, med)) in enumerate(cm.items()):
            ax.plot(dt, worst * 1e6, color=SERIES[i], label=f"{m}, worst epoch")
            ax.plot(dt, med * 1e6, color=SERIES[i], linewidth=1.0, linestyle="--", label=f"{m}, median")
        ax.axhline(te * 1e6, color=INK2, linewidth=1, linestyle=":")
        ax.text(dt[-1], te * 1e6, "Te_NTN 15 kHz", color=INK2, fontsize=8, ha="right", va="bottom")
        ax.axhline(cp * 1e6, color=INK2, linewidth=1, linestyle="-.")
        ax.text(dt[-1], cp * 1e6, "normal CP 15 kHz", color=INK2, fontsize=8, ha="right", va="bottom")
        for d in nr.UL_SYNC_VALIDITY_S:
            ax.axvline(d, color=GRID, linewidth=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 900)
        ax.set_title(
            f"{name}: TA error vs age of SIB19 data\n(UE ephemeris model + common TA as Taylor at epoch)",
            loc="left",
            fontsize=10,
        )
        ax.set_xlabel("Seconds since epochTime (grid: ntn-UlSyncValidityDuration values)")
    axes[0, 0].set_ylabel("|TA error| (us)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, path)


def _plot_harq(ex, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, mu in enumerate((0, 1)):
        n = T.harq_processes_needed(ex["rtt"], mu)
        ax.plot(ex["t"] / 60.0, n, color=SERIES[i], label=f"{15 * 2**mu} kHz SCS", drawstyle="steps-post")
    for lim, lab in ((16, "16 (legacy max)"), (32, "32 (Rel-17 n32)")):
        ax.axhline(lim, color=INK2, linewidth=1, linestyle=":")
        ax.text(ex["t"][0] / 60.0, lim, lab, color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("Minutes from closest approach")
    ax.set_ylabel("HARQ processes for no stalling")
    ax.set_title(f"{ex['key']} {ex['name']} (max elevation {ex['max_el']:.0f} deg), transparent payload", loc="left")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    save(fig, path)

"""How strong is the signal along a pass? CNR and C/N0 along real passes with the TR 38.821 parameter sets.
The TR's LEO-600 payload is applied to Starlink and LEO-1200 to OneWeb (``link.PAYLOAD_ORBIT``).
"""

import matplotlib.pyplot as plt

from .. import link as lb
from .. import orbit, tle
from ..link import PAYLOAD_ORBIT, SETS
from . import Output
from .plot import GRID, INK2, SERIES, save, style

LINKS = [("S", "DL"), ("S", "UL"), ("Ka", "DL"), ("Ka", "UL")]


def _summary(ln, lk):
    cnr = ln.cnr_db(lk.range_m, lk.el_deg)
    cn0 = ln.cn0_dbhz(lk.range_m, lk.el_deg)
    return dict(
        bandwidth_mhz=round(ln.bandwidth_hz / 1e6, 3),
        cnr_db=dict(min=round(float(cnr.min()), 2), max=round(float(cnr.max()), 2)),
        cn0_dbhz=dict(min=round(float(cn0.min()), 2), max=round(float(cn0.max()), 2)),
        swing_db=round(float(cnr.max() - cnr.min()), 2),
    )


def _plot(tracks, path):
    fig, axes = plt.subplots(len(LINKS), len(tracks), figsize=(5 * len(tracks), 10), sharex="col", squeeze=False)
    for j, (key, (payload, tr)) in enumerate(tracks.items()):
        p, t, lk = tr["overhead"]
        m = (t - p.tca_s) / 60.0
        for i, (band, d) in enumerate(LINKS):
            ax = axes[i, j]
            for k, s in enumerate(SETS):
                ln = lb.link(s, payload, band, d)
                ax.plot(m, ln.cnr_db(lk.range_m, lk.el_deg), color=SERIES[k], label=s.replace("set", "Set-"))
            ax.axhline(0.0, color=GRID, linewidth=1)
            term = "handheld" if band == "S" else "VSAT"
            bw = lb.link("set1", payload, band, d).bandwidth_hz / 1e6
            head = (
                f"{tle.CONSTELLATIONS[key].name}, TR {payload} payload, max elevation {p.max_el_deg:.0f} deg\n"
                if i == 0
                else ""
            )
            ax.set_title(f"{head}{band}-band {d}, {term}, {bw:g} MHz", loc="left", fontsize=10)
            if j == 0:
                ax.set_ylabel("CNR (dB)")
        axes[-1, j].set_xlabel("Minutes from closest approach")
    axes[0, 0].legend(frameon=False, loc="lower center")
    fig.text(
        0.01,
        0.005,
        "Satellite beam steered at the UE (Earth-fixed beams). DL bandwidth at frequency reuse 1.",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, path)


def run(**kw):
    style()
    out = Output("link", **kw)
    results = {}
    tracks = {}
    for key, payload in PAYLOAD_ORBIT.items():
        tr = orbit.representative_passes(key)
        tracks[key] = (payload, tr)
        r = {"payload_orbit": payload, "passes": {}}
        for tag, (p, _, lk) in tr.items():
            rp = {"sat": p.name, "max_el_deg": round(p.max_el_deg, 1), "links": {}}
            for s in SETS:
                for band, d in LINKS:
                    rp["links"][f"{s} {band} {d}"] = _summary(lb.link(s, payload, band, d), lk)
            r["passes"][tag] = rp
        results[key] = r
        print(key, payload)
        for tag, rp in r["passes"].items():
            print(f"  {tag} pass ({rp['max_el_deg']} deg):")
            for k, v in rp["links"].items():
                print(
                    f"    {k:16s} CNR {v['cnr_db']['min']:6.1f} .. {v['cnr_db']['max']:6.1f} dB"
                    f"  C/N0 {v['cn0_dbhz']['min']:5.1f} .. {v['cn0_dbhz']['max']:5.1f} dB-Hz"
                )
    _plot(tracks, out.figure("cnr"))
    out.write(results)
    return results

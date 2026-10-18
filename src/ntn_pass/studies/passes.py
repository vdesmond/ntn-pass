"""What does a pass look like from the ground? Pass geometry (elevation, delay, Doppler and its rate),
the serving-satellite timeline and handover rates over the site, for Starlink, OneWeb and Galileo."""

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from .. import orbit, serving, tle
from ..constants import C
from ..link import FC_KA, FC_S
from ..orbit import MASK_DEG as MASK
from ..sites import SITES
from . import Output
from .plot import INK2, SERIES, save, style

LOW_PASS_MIN_EL = 15.0
PLOT_HOURS = {"starlink": 2, "oneweb": 2, "galileo": 24}
POLICIES = [("max-elevation", 0.0), ("max-elevation", 5.0), ("until-mask", 0.0)]
PLOT_POLICY = ("until-mask", 0.0)  # readable sawtooth; max-elevation switches every few tens of seconds


def _pass_summary(p, lk):
    return dict(
        sat=p.name,
        aos_utc=tle.utc(p.aos_s).isoformat(),
        duration_s=round(p.duration_s, 1),
        max_el_deg=round(p.max_el_deg, 2),
        delay_ms=[round(float(lk.delay_s.min() * 1e3), 3), round(float(lk.delay_s.max() * 1e3), 3)],
        max_doppler_ppm=round(float(np.abs(lk.doppler_ppm()).max()), 3),
        max_doppler_khz_s_band=round(float(np.abs(lk.doppler_hz(FC_S)).max() / 1e3), 2),
        max_doppler_khz_ka=round(float(np.abs(lk.doppler_hz(FC_KA)).max() / 1e3), 1),
        max_doppler_rate_ppm_s=float(f"{np.abs(lk.range_accel).max() / C * 1e6:.4g}"),
        max_delay_drift_us_s=round(float(np.abs(lk.range_rate).max() / C * 1e6), 3),
    )


def _plot_passes(tracks, path):
    """Small multiples, one column per constellation: elevation, delay, Doppler, Doppler rate."""
    rows = [
        ("Elevation (deg)", lambda lk: lk.el_deg),
        ("One-way delay (ms)", lambda lk: lk.delay_s * 1e3),
        ("Doppler (ppm)", lambda lk: lk.doppler_ppm()),
        ("Doppler rate (ppm/s)", lambda lk: -1e6 * lk.range_accel / C),
    ]
    fig, axes = plt.subplots(len(rows), len(tracks), figsize=(4.2 * len(tracks), 9), sharex="col", squeeze=False)
    for j, (title, p, t, lk) in enumerate(tracks):
        m = (t - p.tca_s) / 60.0
        for i, (label, f) in enumerate(rows):
            ax = axes[i, j]
            ax.plot(m, f(lk), color=SERIES[0])
            if j == 0:
                ax.set_ylabel(label)
        axes[0, j].set_title(f"{title}: {p.name}\nmax elevation {p.max_el_deg:.1f} deg", loc="left")
        axes[-1, j].set_xlabel("Minutes from closest approach")
    fig.text(
        0.01,
        0.005,
        "Doppler in ppm of the carrier: 1 ppm = 2 kHz at 2 GHz (S-band), 20 kHz at 20 GHz (Ka-band).",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, path)


def _plot_serving(t, el, tl, site, title, path, plot_hours):
    k = t <= plot_hours * 3600.0
    t, el = t[k], el[:, k]
    h = t / 3600.0
    idx = tl.serving[k]
    serv_el = np.where(idx >= 0, el[np.maximum(idx, 0), np.arange(len(t))], np.nan)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(h, serv_el, color=SERIES[0], linewidth=1.2)
    ax[0].axhline(MASK, color=INK2, linewidth=1, linestyle="--")
    ax[0].set_ylabel("Serving satellite\nelevation (deg)")
    ax[0].set_title(
        f"{title} over {site.name}: {tl.handovers_per_hour:.1f} handovers per hour ({tl.describe()})", loc="left"
    )
    ax[1].plot(h, serving.in_view(el, MASK), color=SERIES[1], linewidth=1.2, drawstyle="steps-post")
    ax[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[1].set_ylabel(f"In view\n(> {MASK:g} deg)")
    ax[1].set_xlabel(f"Hours from {tle.T0:%Y-%m-%d %H:%M} UTC")
    fig.tight_layout()
    save(fig, path)


def run(site_key="sophia", **kw):
    style()
    out = Output("passes", **kw)
    site = SITES[site_key]
    results = dict(
        site=dict(name=site.name, lat_deg=site.lat_deg, lon_deg=site.lon_deg, alt_m=site.alt_m),
        t0_utc=tle.T0.isoformat(),
        snapshot=tle.SNAPSHOT,
        mask_deg=MASK,
    )
    tracks = []
    for key, (hours, step_s) in orbit.SURVEY.items():
        f = tle.load(key)
        t, jd, fr = tle.time_grid(hours * 3600.0, step_s)
        el = orbit.elevation_grid(f.sats, site, jd, fr)
        ps = orbit.find_passes(f, site, t, el, MASK)
        nv = serving.in_view(el, MASK)
        r = dict(
            satellites=len(f),
            window_h=hours,
            step_s=step_s,
            passes=len(ps),
            mean_pass_s=round(float(np.mean([p.duration_s for p in ps])), 1),
            in_view=dict(min=int(nv.min()), mean=round(float(nv.mean()), 2), max=int(nv.max())),
            serving={},
        )
        for pol, hyst in POLICIES:
            tl = serving.timeline(t, el, MASK, pol, hyst)
            dw = tl.dwell_s()
            r["serving"][tl.describe()] = dict(
                handovers_per_hour=round(float(tl.handovers_per_hour), 2),
                median_dwell_s=round(float(np.median(dw)), 1) if len(dw) else None,
                outage_fraction=round(tl.outage_fraction, 4),
            )
            if (pol, hyst) == PLOT_POLICY:
                _plot_serving(
                    t,
                    el,
                    tl,
                    site,
                    f.constellation.name,
                    out.figure(f"serving-{key}"),
                    PLOT_HOURS[key],
                )
        best = max(ps, key=lambda p: p.max_el_deg)
        # lowest pass that is not a graze of the mask
        low = min((p for p in ps if p.max_el_deg >= LOW_PASS_MIN_EL), key=lambda p: p.max_el_deg)
        step = 1.0 if key != "galileo" else 30.0
        for tag, p in (("highest", best), ("lowest", low)):
            tt, lk = orbit.pass_track(f.sats[p.sat], site, p, step)
            r[f"{tag}_pass"] = _pass_summary(p, lk)
            if tag == "highest":
                tracks.append((f.constellation.name, p, tt, lk))
        results[key] = r
        print(f"{f.constellation.name}: {len(f)} sats, {len(ps)} passes in {hours:g} h, in view {r['in_view']}")
        for k, v in r["serving"].items():
            print(f"  {k}: {v}")
    _plot_passes(tracks, out.figure("tracks"))
    out.write(results)
    return results

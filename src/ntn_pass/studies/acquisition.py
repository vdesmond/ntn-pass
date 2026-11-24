"""Can the UE find the cell, and how precisely does one SSB time the satellite? Acquiring the SSB under LEO
Doppler, and the per-burst delay and Doppler it measures.

Downlink, band n256 (S-band, SSB at 15 kHz), carrier 2 GHz as everywhere else (n256 DL is 2170-2200 MHz,
so Hz values there are ~9 % higher). The UE's reference oscillator is GNSS-disciplined (an NR NTN UE has
GNSS), so the absolute carrier offset it sees is the service-link Doppler plus the satellite access node's
frequency error (TS 38.108 V17.5.0 6.5.1.2: 0.05 ppm). TS 38.300 V17.6.0 16.14.2.2 leaves the feeder-link
Doppler to the network and the service-link Doppler to the UE, so the downlink arrives with the full
service-link Doppler.

Two searches:
- cold: no ephemeris yet (SIB19 comes after the SSB). +-(24 ppm + 0.05 ppm) in 5 kHz steps, 20 ms of
  timing (one SSB period), 3 N_ID2. 24 ppm is the TR 38.811 LEO-600 maximum.
- warm: valid ephemeris and GNSS position (connected mode, or idle with a stored SIB19). The UE
  pre-compensates the Doppler and the delay. What is left is the two-body ephemeris error (``timing``:
  below 12 Hz at 120 s) plus the 0.05 ppm SAN error, so one frequency hypothesis and +-1 symbol of timing.
"""

import matplotlib.pyplot as plt
import numpy as np

from .. import acquire as D
from .. import bounds, nr, orbit, tle
from .. import link as lb
from ..constants import C
from . import Output
from .plot import GRID, INK2, SERIES, save, style

FC = lb.FC_S
PFA = 0.01
SAN_PPM = 0.05
COLD_SPAN_HZ = (24.0 + SAN_PPM) * 1e-6 * FC
COLD_STEP_HZ = 5e3
WARM_SPAN_HZ = 12.0 + SAN_PPM * 1e-6 * FC
N_ID1, N_ID2 = 17, 1
K_COMBINE = 4
MIN_DETECTIONS = 50  # accuracy points with fewer detected trials are not reported

SEARCHES = {
    "cold": dict(freqs=D.freq_grid(COLD_SPAN_HZ, COLD_STEP_HZ), window=round(nr.BURST_PERIOD_S * nr.FS)),
    "warm": dict(freqs=np.array([0.0]), window=2 * nr.NSYM),
}
MODES = {  # label -> (search, bursts, drift compensation)
    "cold, 1 burst": ("cold", 1, True),
    f"cold, {K_COMBINE} bursts, drift compensated": ("cold", K_COMBINE, True),
    f"cold, {K_COMBINE} bursts, no drift compensation": ("cold", K_COMBINE, False),
    "warm, 1 burst": ("warm", 1, True),
    f"warm, {K_COMBINE} bursts": ("warm", K_COMBINE, True),
}


def thresholds():
    out = {}
    for label, (s, k, _) in MODES.items():
        n = D.n_cells(SEARCHES[s]["window"], len(SEARCHES[s]["freqs"]))
        out[label] = D.threshold(PFA, n, k)
    return out


def _draw(search, rng):
    """Residual frequency offset and the delay drift that goes with it."""
    if search == "cold":
        dop = rng.uniform(-24e-6, 24e-6) * FC
        df = rng.uniform(-1, 1) * SAN_PPM * 1e-6 * FC
        return dop + df, -dop / FC, df
    df = rng.uniform(-WARM_SPAN_HZ, WARM_SPAN_HZ)
    return df, 0.0, df  # Doppler and drift pre-compensated by the UE


def _bursts(k, d1, df, snr_db, rng, n=700, lead=40.0):
    dl = nr.Delay(2e-3, d1)
    out, t0s, s2 = [], [], None
    lead += rng.uniform(0, 1)
    for i in range(k):
        t_tx0 = i * nr.BURST_PERIOD_S
        t0 = t_tx0 + dl.d0_s + nr.NCP / nr.FS - (lead + nr.NCP) / nr.FS
        t = t0 + np.arange(n) / nr.FS
        rx = nr.received(nr.ssb_grid(N_ID1, N_ID2, rng), t, t_tx0, dl, FC, df, rng.uniform(0, 2 * np.pi))
        w, s2 = nr.awgn(n, snr_db, rng)
        out.append(rx + w)
        t0s.append(t0)
    return out, np.array(t0s), s2, dl


def pd_curves(snrs, trials, rng):
    thr = thresholds()
    pd = {label: np.zeros(len(snrs)) for label in MODES}
    for i, snr in enumerate(snrs):
        for _ in range(trials):
            for search in ("cold", "warm"):
                _, d1, df = _draw(search, rng)
                bursts, t0s, s2, _ = _bursts(K_COMBINE, d1, df, snr, rng)
                freqs = SEARCHES[search]["freqs"]
                for label, (s, k, comp) in MODES.items():
                    if s != search:
                        continue
                    T = D.search(bursts[:k], s2, freqs, FC, (N_ID2,), comp, t0s[:k])
                    pd[label][i] += T.max() > thr[label]
        for label in MODES:
            pd[label][i] /= trials
    return pd


def accuracy(snrs, trials, rng):
    """Cold acquisition (1 burst) then fine estimation. Errors over the detected trials (NaN when fewer
    than MIN_DETECTIONS were detected: the RMSE would rest on a handful of samples)."""
    thr = thresholds()["cold, 1 burst"]
    freqs = SEARCHES["cold"]["freqs"]
    out = {k: [] for k in ("pd", "n_detected", "rmse_toa_s", "rmse_foa_hz", "outlier_rate")}
    for snr in snrs:
        et, ef, det = [], [], 0
        for _ in range(trials):
            _, d1, df = _draw("cold", rng)
            (rx,), t0s, s2, dl = _bursts(1, d1, df, snr, rng, n=1100)
            T = D.search([rx], s2, freqs, FC, (N_ID2,), t0s=t0s)[0]
            if T.max() <= thr:
                continue
            det += 1
            b, lag = np.unravel_index(np.argmax(T), T.shape)
            e = D.fine(rx, lag, freqs[b], N_ID1, N_ID2, FC, t0_s=t0s[0])
            et.append(e.toa_s - nr.NCP / nr.FS - dl(e.t_toa_s))
            ef.append(e.foa_hz - (-FC * dl.rate(e.t_foa_s) + df))
        et, ef = np.array(et), np.array(ef)
        crb = np.array(bounds.ssb_crb(10 ** (snr / 10)))
        enough = det >= MIN_DETECTIONS
        out["pd"].append(det / trials)
        out["n_detected"].append(det)
        out["rmse_toa_s"].append(float(np.sqrt(np.mean(et**2))) if enough else np.nan)
        out["rmse_foa_hz"].append(float(np.sqrt(np.mean(ef**2))) if enough else np.nan)
        out["outlier_rate"].append(
            float(np.mean((np.abs(et) > 5 * crb[0]) | (np.abs(ef) > 5 * crb[1]))) if enough else np.nan
        )
    return {k: np.array(v) for k, v in out.items()}


def crb_region_start(snrs, acc, factor=1.2, which=("toa", "foa")):
    """Lowest SNR from which the RMSE stays within `factor` of the CRB (for TOA and/or FOA)."""
    crb = np.array([bounds.ssb_crb(10 ** (s / 10)) for s in snrs])
    ok = np.ones(len(snrs), dtype=bool)
    if "toa" in which:
        ok &= acc["rmse_toa_s"] <= factor * crb[:, 0]
    if "foa" in which:
        ok &= acc["rmse_foa_hz"] <= factor * crb[:, 1]
    for i in range(len(snrs)):
        if ok[i:].all():
            return float(snrs[i])
    return None


def snr_at(snrs, curve, level=0.9):
    above = np.flatnonzero(curve >= level)
    if not len(above):
        return None
    i = above[0]
    if i == 0:
        return float(snrs[0])
    return float(np.interp(level, [curve[i - 1], curve[i]], [snrs[i - 1], snrs[i]]))


def _interp_log(x, xs, ys):
    good = np.isfinite(ys)
    return 10 ** np.interp(x, xs[good], np.log10(ys[good]), left=np.nan, right=np.nan)


def sigma_along(snr, snrs, acc, crb_from):
    """Per-SSB (TOA s, FOA Hz) sigma at the given SNRs: the CRB where the estimator was shown to reach it,
    the measured RMSE (outliers included) below that."""
    crb = np.array([bounds.ssb_crb(10 ** (x / 10)) for x in np.atleast_1d(snr)])
    below = np.atleast_1d(snr) < crb_from
    toa = np.where(below, _interp_log(snr, snrs, acc["rmse_toa_s"]), crb[:, 0])
    foa = np.where(below, _interp_log(snr, snrs, acc["rmse_foa_hz"]), crb[:, 1])
    return toa, foa, below


def along_passes(snrs, acc, pd, crb_from):
    out, curves = {}, {}
    for key, payload in lb.PAYLOAD_ORBIT.items():
        tr = orbit.representative_passes(key)
        out[key] = {"payload_orbit": payload, "passes": {}}
        for tag, (p, t, lk) in tr.items():
            rp = {"sat": p.name, "max_el_deg": round(p.max_el_deg, 1), "sets": {}}
            for s in lb.SETS:
                snr = lb.link(s, payload, "S", "DL").cnr_db(lk.range_m, lk.el_deg)
                toa, foa, below = sigma_along(snr, snrs, acc, crb_from)
                pd1 = np.interp(snr, snrs, pd["cold, 1 burst"])
                rp["sets"][s] = dict(
                    snr_re_db=[round(float(snr.min()), 2), round(float(snr.max()), 2)],
                    pd_cold_1_burst_min=round(float(pd1.min()), 3),
                    share_below_crb_region=round(float(below.mean()), 3),
                    sigma_toa_ns=[round(float(np.nanmin(toa) * 1e9), 2), round(float(np.nanmax(toa) * 1e9), 2)],
                    sigma_dl_toa_times_c_m=[round(float(np.nanmin(toa) * C), 2), round(float(np.nanmax(toa) * C), 2)],
                    sigma_foa_hz=[round(float(np.nanmin(foa)), 1), round(float(np.nanmax(foa)), 1)],
                    sigma_range_rate_m_s=[
                        round(float(np.nanmin(foa) * C / FC), 2),
                        round(float(np.nanmax(foa) * C / FC), 2),
                    ],
                )
                if tag == "overhead":
                    curves[(key, s)] = ((t - p.tca_s) / 60.0, lk, toa, foa, below, p)
            out[key]["passes"][tag] = rp
    return out, curves


def _plot_detection(snrs, pd, bands, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    styles = {
        "cold, 1 burst": (SERIES[0], "-"),
        f"cold, {K_COMBINE} bursts, drift compensated": (SERIES[0], "--"),
        f"cold, {K_COMBINE} bursts, no drift compensation": (SERIES[0], ":"),
        "warm, 1 burst": (SERIES[1], "-"),
        f"warm, {K_COMBINE} bursts": (SERIES[1], "--"),
    }
    for (lo, hi), name in bands:
        ax.axvspan(lo, hi, color=GRID, alpha=0.7, linewidth=0)
        ax.text(lo + 0.3, 0.5, name, color=INK2, fontsize=8, va="center")
    for label, (c, ls) in styles.items():
        x = np.append(snrs, max(hi for (_, hi), _ in bands))
        ax.plot(x, np.append(pd[label], 1.0), color=c, linestyle=ls, label=label)
    ax.set_xlabel("SNR per resource element (dB)")
    ax.set_ylabel(f"Detection probability (Pfa {PFA:g} per search)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("SSB acquisition, PSS matched filter, band n256 (15 kHz)", loc="left")
    fig.text(
        0.01,
        0.005,
        "Shaded: SNR over the overhead passes (thermal noise only, beam steered at the UE).",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, path)


def _plot_accuracy(snrs, acc, path):
    crb = np.array([bounds.ssb_crb(10 ** (s / 10)) for s in snrs])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, c, unit, scale, title in (
        (axes[0], acc["rmse_toa_s"], crb[:, 0], "ns", 1e9, "Time of arrival"),
        (axes[1], acc["rmse_foa_hz"], crb[:, 1], "Hz", 1.0, "Frequency of arrival"),
    ):
        ax.semilogy(
            snrs, y * scale, color=SERIES[0], marker="o", markersize=3, label="cold acquisition + fine estimator"
        )
        ax.semilogy(snrs, c * scale, color=INK2, linestyle="--", linewidth=1.2, label="Cramér-Rao bound")
        ax.set_xlabel("SNR per resource element (dB)")
        ax.set_ylabel(f"RMS error ({unit})")
        ax.set_title(f"{title}, one SSB (PSS + SSS)", loc="left")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, path)


def _plot_pass(curves, crb_from, path):
    keys = list(lb.PAYLOAD_ORBIT)
    fig, axes = plt.subplots(3, len(keys), figsize=(5 * len(keys), 8), sharex="col", squeeze=False)
    for j, key in enumerate(keys):
        for k, s in enumerate(lb.SETS):
            m, lk, toa, foa, below, p = curves[(key, s)]
            axes[1, j].semilogy(m, toa * C, color=SERIES[k], label=s.replace("set", "Set-"))
            axes[2, j].semilogy(m, foa * C / FC, color=SERIES[k])
            if below.any():
                axes[2, j].semilogy(
                    m[below],
                    foa[below] * C / FC,
                    ".",
                    color=SERIES[k],
                    markersize=3,
                    label="below the CRB region (measured, outliers included)",
                )
        axes[0, j].plot(m, lk.doppler_hz(FC) / 1e3, color=INK2)
        axes[0, j].set_title(
            f"{tle.CONSTELLATIONS[key].name}, max elevation {p.max_el_deg:.0f} deg, S-band DL handheld", loc="left"
        )
        axes[2, j].set_xlabel("Minutes from closest approach")
    axes[0, 0].set_ylabel("Doppler at 2 GHz (kHz)")
    axes[1, 0].set_ylabel("sigma DL TOA x c, one SSB (m)")
    axes[2, 0].set_ylabel("sigma range rate, one SSB (m/s)")
    axes[1, 0].legend(frameon=False)
    axes[2, 0].legend(frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.005,
        f"Sigma = CRB, which the estimator reaches from {crb_from:g} dB per RE. Excludes the satellite's "
        f"own frequency error\n({SAN_PPM} ppm = {SAN_PPM * 1e-6 * C:.0f} m/s), a bias common to every burst.",
        color=INK2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, path)


def run(trials_pd=150, trials_acc=200, seed=4, **kw):
    style()
    out = Output("acquisition", **kw)
    rng = np.random.default_rng(seed)
    snr_pd = np.arange(-18.0, -1.0, 1.0)
    snr_acc = np.arange(-12.0, 21.0, 2.0)

    pd = pd_curves(snr_pd, trials_pd, rng)
    acc = accuracy(snr_acc, trials_acc, rng)
    crb_from = crb_region_start(snr_acc, acc)
    passes, curves = along_passes(snr_acc, acc, {k: np.interp(snr_acc, snr_pd, v) for k, v in pd.items()}, crb_from)

    thr = thresholds()
    res = {
        "fc_hz": FC,
        "ssb": {
            "scs_hz": nr.SCS,
            "fs_hz": nr.FS,
            "cp_samples": nr.NCP,
            "period_ms": nr.BURST_PERIOD_S * 1e3,
            "pss_rms_bandwidth_khz": round(bounds.ssb_rms_bandwidth_hz() / 1e3, 1),
        },
        "pfa_per_search": PFA,
        "searches": {
            s: {
                "freq_hypotheses": len(c["freqs"]),
                "span_hz": round(float(np.max(np.abs(c["freqs"]))), 1),
                "timing_lags": c["window"],
                "cells": D.n_cells(c["window"], len(c["freqs"])),
            }
            for s, c in SEARCHES.items()
        },
        "threshold": {k: round(v, 2) for k, v in thr.items()},
        "snr_re_db_for_pd_0.9": {k: snr_at(snr_pd, v) for k, v in pd.items()},
        "snr_re_db_for_pd_0.9_analytic_aligned": {
            k: round(D.pd_aligned_snr_db(0.9, thr[k], MODES[k][1]), 2) for k in MODES if MODES[k][0] == "warm"
        },
        "pd_curve": {"snr_re_db": snr_pd.tolist(), **{k: v.round(3).tolist() for k, v in pd.items()}},
        "accuracy": {
            "snr_re_db": snr_acc.tolist(),
            "pd_cold_1_burst": acc["pd"].round(3).tolist(),
            "n_detected": acc["n_detected"].tolist(),
            "rmse_toa_ns": (acc["rmse_toa_s"] * 1e9).round(3).tolist(),
            "crb_toa_ns": [round(float(bounds.ssb_crb(10 ** (s / 10))[0] * 1e9), 3) for s in snr_acc],
            "rmse_foa_hz": acc["rmse_foa_hz"].round(2).tolist(),
            "crb_foa_hz": [round(float(bounds.ssb_crb(10 ** (s / 10))[1]), 2) for s in snr_acc],
            "outlier_rate": acc["outlier_rate"].round(3).tolist(),
            "crb_region_from_snr_re_db": crb_from,
            "toa_crb_region_from_snr_re_db": crb_region_start(snr_acc, acc, which=("toa",)),
        },
        "san_frequency_error": {
            "ppm": SAN_PPM,
            "hz": SAN_PPM * 1e-6 * FC,
            "range_rate_m_s": round(SAN_PPM * 1e-6 * C, 2),
        },
        "passes": passes,
    }
    bands = []
    for s in lb.SETS:
        r = [passes[k]["passes"]["overhead"]["sets"][s]["snr_re_db"] for k in lb.PAYLOAD_ORBIT]
        bands.append(((min(x[0] for x in r), max(x[1] for x in r)), s.replace("set", "Set-")))
    _plot_detection(snr_pd, pd, bands, out.figure("detection"))
    _plot_accuracy(snr_acc, acc, out.figure("accuracy"))
    _plot_pass(curves, crb_from, out.figure("pass"))
    out.write(res)

    print("SNR per RE for Pd 0.9:")
    for k, v in res["snr_re_db_for_pd_0.9"].items():
        print(f"  {k:45s} {v}")
    print(
        "CRB region from",
        res["accuracy"]["crb_region_from_snr_re_db"],
        "dB (TOA alone from",
        res["accuracy"]["toa_crb_region_from_snr_re_db"],
        "dB)",
    )
    print("aligned analytic SNR for Pd 0.9:", res["snr_re_db_for_pd_0.9_analytic_aligned"])
    for key, r in passes.items():
        for tag, rp in r["passes"].items():
            for s, v in rp["sets"].items():
                print(
                    f"{key:8s} {tag:8s} {s}: SNR {v['snr_re_db']} dB, Pd_min {v['pd_cold_1_burst_min']}, "
                    f"sigma DL TOA x c {v['sigma_dl_toa_times_c_m']} m, "
                    f"sigma range rate {v['sigma_range_rate_m_s']} m/s"
                )
    return res

"""Acquiring the NR SSB under Doppler, and measuring its time and frequency of arrival.

Coarse search: PSS matched filter over a timing window, for each N_ID2 and each frequency hypothesis on a
grid. With the noise variance known, the per-cell statistic |corr|^2 / (sigma^2 ||p||^2) is Exp(1) under
noise alone, and the sum over K bursts is Gamma(K, 1). The threshold spreads the false-alarm probability
over every searched cell (Bonferroni), which is conservative because neighbouring cells are correlated.

Combining K bursts (20 ms apart) non-coherently: a frequency hypothesis f also fixes the delay drift
tau' = -f / fc, because both come from the same range rate. The correlation of burst k is shifted back by
tau' k T before summing. Without that, a LEO satellite's ~22 us/s drift moves the peak by ~1.7 samples per
burst at 3.84 MHz and smears it.

Fine estimation (cell identity known, so both PSS and SSS are reference signals): FFT both symbols at the
coarse timing, Z_{s,k} = Y_{s,k} X_{s,k}^*, and maximise
    | sum_s exp(-j 2 pi f t_s) sum_k Z_{s,k} exp(j 2 pi k eps / N) |
over the timing offset eps (samples) and the residual frequency f, re-derotating twice so the
inter-carrier interference of the residual offset is removed. The timing offset of the SSS window is
eps + tau' (t_SSS - t_PSS) fs with tau' = -f / fc: the same Doppler that shifts the carrier compresses
the envelope (22 ppm on Starlink), and leaving it out biases the estimates by ~0.7 ns and ~0.4 Hz.
This is the ML estimator for the frequency-domain model of `bounds.ssb_crb`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from . import nr


def threshold(pfa, n_cells, k_bursts=1):
    """Per-cell threshold on the normalised statistic for a total false-alarm probability pfa."""
    return float(stats.gamma.isf(pfa / n_cells, k_bursts))


def freq_grid(span_hz, step_hz):
    """Hypotheses covering +-span_hz with step step_hz, centred on 0."""
    n = int(np.ceil(span_hz / step_hz))
    return np.arange(-n, n + 1) * step_hz


def n_cells(window_samples, n_freq, n_id2=3):
    return window_samples * n_freq * n_id2


def correlate(rx, replica, f_hz, t0_s=0.0):
    """Unnormalised PSS matched filter after derotating by f_hz: c[n] = sum_m r[n+m] e^{-j2pi f t} p*[m].

    Returns len(rx) - len(replica) + 1 lags."""
    n = np.arange(len(rx))
    r = rx * np.exp(-2j * np.pi * f_hz * (t0_s + n / nr.FS))
    L = len(r)
    nfft = 1 << int(np.ceil(np.log2(L + len(replica))))
    c = np.fft.ifft(np.fft.fft(r, nfft) * np.conj(np.fft.fft(replica, nfft)))
    return c[: L - len(replica) + 1]


def _shift(x, shift_samples):
    """Delay a real sequence by a fractional number of samples (frequency-domain, circular)."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.fftfreq(nfft)
    return np.real(np.fft.ifft(np.fft.fft(x, nfft) * np.exp(-2j * np.pi * f * shift_samples)))[:n]


def search(bursts, sigma2, freqs, fc, n_id2_set=(0, 1, 2), drift_comp=True, t0s=None):
    """Statistic over (n_id2, freq, lag), non-coherently combined over bursts.

    bursts: list of K equal-length windows, opened BURST_PERIOD_S apart on the receiver clock.
    Returns T with shape (len(n_id2_set), len(freqs), n_lags).
    """
    K = len(bursts)
    t0s = np.zeros(K) if t0s is None else t0s
    reps = [nr.pss_replica(i) for i in n_id2_set]
    norm = sigma2 * 127.0
    out = None
    for a, p in enumerate(reps):
        for b, f in enumerate(freqs):
            acc = 0.0
            for k, rx in enumerate(bursts):
                c = np.abs(correlate(rx, p, f, t0s[k])) ** 2 / norm
                if drift_comp and k:
                    c = _shift(c, -(-f / fc) * k * nr.BURST_PERIOD_S * nr.FS)
                acc = acc + c
            if out is None:
                out = np.empty((len(reps), len(freqs), len(acc)))
            out[a, b] = acc
    return out


BACKOFF = 4  # FFT window starts this many samples into the CP, to tolerate coarse timing error
EPS_GRID = np.arange(-nr.NCP, nr.NCP + 1e-9, 0.125)  # timing offsets searched by `fine` (samples)


def _z(rx, n0, f_hz, ref, t0_s=0.0):
    """Z_{s,k} for the PSS (s=0) and SSS (s=2) windows starting at n0 - BACKOFF (+ 2 symbols)."""
    n = np.arange(len(rx))
    r = rx * np.exp(-2j * np.pi * f_hz * (t0_s + n / nr.FS))
    z, ts = [], []
    for s, X in ((0, ref[0]), (2, ref[1])):
        w = n0 - BACKOFF + s * nr.NSYM
        Y = np.fft.fft(r[w : w + nr.NFFT]) / np.sqrt(nr.NFFT)
        z.append(Y[nr.TONES % nr.NFFT] * X)
        ts.append(t0_s + w / nr.FS)
    return np.array(z), np.array(ts)


def _cost(p, z, ts, tau_dot):
    eps, f = p
    eps_s = eps + tau_dot * (ts - ts[0]) * nr.FS
    a = np.sum(z * np.exp(2j * np.pi * np.outer(eps_s, nr.TONES) / nr.NFFT), axis=1)
    return -np.abs(np.sum(np.exp(-2j * np.pi * f * (ts - ts[0])) * a))


@dataclass
class Estimate:
    toa_s: float  # arrival (receiver clock) of the start of the PSS useful part
    t_toa_s: float  # instant the delay toa_s - t_emit refers to: the centre of the PSS FFT window
    foa_hz: float  # frequency of the received carrier (baseband)
    t_foa_s: float  # instant the frequency refers to: midway between the PSS and SSS windows


def _start(rx, n0, f, ref, t0_s):
    """Global search over the timing offset with the frequency step in closed form.

    For a given eps the best residual frequency is the PSS->SSS phase step and the cost is |a_0| + |a_1|.
    (A local search from a phase-slope guess lands on sidelobes of the delay cost at low SNR.)
    Returns (z, ts, eps0, f0, peak)."""
    z, ts = _z(rx, n0, f, ref, t0_s)
    a = z @ np.exp(2j * np.pi * np.outer(nr.TONES, EPS_GRID) / nr.NFFT)  # (2, n_grid)
    m = np.abs(a[0]) + np.abs(a[1])
    i = int(np.argmax(m))
    f0 = np.angle(a[1, i] * np.conj(a[0, i])) / (2 * np.pi * (ts[1] - ts[0]))
    return z, ts, EPS_GRID[i], f0, m[i]


def fine(rx, n0, f_hz, n_id1, n_id2, fc, t0_s=0.0, iterations=3):
    ref = (nr.pss(n_id2), nr.sss(n_id1, n_id2))
    # The phase step is ambiguous by 1 / (2 T_sym) = 7 kHz, and at low SNR the coarse search can pick the
    # neighbouring 5 kHz bin. Try the three wraps: the wrong ones leave ~half a subcarrier of residual
    # offset, whose inter-carrier loss shows up as a lower peak after re-derotation.
    wrap = nr.FS / (2 * nr.NSYM)
    best = None
    for m in (-1, 0, 1):
        _, _, _, f0, _ = _start(rx, n0, f_hz + m * wrap, ref, t0_s)
        peak = _start(rx, n0, f_hz + m * wrap + f0, ref, t0_s)[4]
        if best is None or peak > best[1]:
            best = (f_hz + m * wrap + f0, peak)
    f = best[0]
    eps = 0.0
    for _ in range(iterations):
        z, ts, e0, f0, _ = _start(rx, n0, f, ref, t0_s)
        res = optimize.minimize(
            _cost,
            [e0, f0],
            args=(z, ts, -f / fc),
            method="Nelder-Mead",
            options=dict(xatol=1e-6, fatol=1e-12, maxiter=400),
        )
        eps, df = res.x
        f += df
    toa = t0_s + (n0 - BACKOFF + eps) / nr.FS
    w0 = t0_s + (n0 - BACKOFF) / nr.FS
    return Estimate(toa, w0 + nr.NFFT / 2 / nr.FS, f, w0 + (nr.NSYM + nr.NFFT / 2) / nr.FS)


def pd_aligned(snr_db, thr, k_bursts):
    """Pd at the true lag with no frequency or timing offset: 2T ~ ncx2(2K, 254 K SNR) (127 tones)."""
    return stats.ncx2.sf(2 * thr, 2 * k_bursts, 254 * k_bursts * 10 ** (np.asarray(snr_db) / 10))


def pd_aligned_snr_db(level, thr, k_bursts):
    """Per-RE SNR (dB) at which pd_aligned reaches `level`."""
    return float(optimize.brentq(lambda x: pd_aligned(x, thr, k_bursts) - level, -30.0, 10.0))

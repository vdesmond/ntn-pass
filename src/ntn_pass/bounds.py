"""Cramer-Rao bounds on time and frequency of arrival (TOA, FOA), for the NR SSB and the 406 MHz beacons.

Both use the same Fisher information. For a known signal s(t; tau, f) in complex white noise,
J_ij = 2 SNR Re <ds/dtheta_i, ds/dtheta_j>, with the unknown amplitude and phase as nuisances. Only the
derivative matrix differs:
- ``waveform_crb``: any sampled waveform at a given C/N0 (the 406 MHz FGB and SGB bursts);
- ``ssb_crb``: the frequency-domain model of the SSB fine estimator, Z_{s,k} = A exp(-j 2 pi k df tau)
  exp(j 2 pi f t_s) + noise, with 127 tones on two symbols 2 T_sym apart and a per-RE SNR. The phase
  nuisance is projected out by removing the mean phase derivative. The amplitude decouples.
The closed forms at the end are the textbook limits both reduce to. The tests check them against each other.
"""

import numpy as np

from . import nr


def waveform_crb(s, fs, cn0_dbhz):
    """(sigma TOA s, sigma FOA Hz) for a known unit-power waveform received at C/N0, with unknown amplitude
    and phase. Time is measured from the middle of the burst."""
    n = len(s)
    t = (np.arange(n) - (n - 1) / 2) / fs
    f = np.fft.fftfreq(n, 1 / fs)
    ds_dt = np.fft.ifft(2j * np.pi * f * np.fft.fft(s))
    d = np.stack([-ds_dt, 2j * np.pi * t * s, 1j * s, s], axis=1)  # tau, f, phase, amplitude
    J = 2.0 * 10.0 ** (np.asarray(cn0_dbhz) / 10.0) * np.real(d.conj().T @ d) / fs
    P = np.linalg.inv(J)
    return np.sqrt(P[0, 0]), np.sqrt(P[1, 1])


def ssb_crb(snr_re_lin):
    """(sigma TOA s, sigma FOA Hz) of one SSB (PSS + SSS) at a per-resource-element SNR (linear)."""
    k = nr.TONES.astype(float)
    t = np.array([0.0, 2 * nr.T_SYM])
    # derivative of the phase w.r.t. (tau, f) per observation. The unknown phase removes the mean
    d_tau = np.concatenate([-2 * np.pi * k * nr.SCS] * 2)
    d_f = np.repeat(2 * np.pi * t, len(k))
    D = np.stack([d_tau, d_f], axis=1)
    D = D - D.mean(axis=0)
    P = np.linalg.inv(2.0 * snr_re_lin * D.T @ D)
    return np.sqrt(P[0, 0]), np.sqrt(P[1, 1])


def rms_bandwidth_hz(f, power=None):
    """RMS bandwidth of a spectrum: the spread of `f` weighted by `power` (uniform when None)."""
    f = np.asarray(f, dtype=float)
    w = np.ones_like(f) if power is None else np.asarray(power, dtype=float)
    m = np.sum(f * w) / np.sum(w)
    return float(np.sqrt(np.sum((f - m) ** 2 * w) / np.sum(w)))


def waveform_rms_bandwidth_hz(s, fs):
    return rms_bandwidth_hz(np.fft.fftfreq(len(s), 1 / fs), np.abs(np.fft.fft(s)) ** 2)


def ssb_rms_bandwidth_hz():
    """The 127 equal-power PSS / SSS tones."""
    return rms_bandwidth_hz(nr.TONES * nr.SCS)


def toa_sigma(ebn0_lin, rms_bw_hz):
    """TOA bound from the RMS bandwidth: 1 / (2 pi B_rms sqrt(2 E/N0))."""
    return 1.0 / (2.0 * np.pi * rms_bw_hz * np.sqrt(2.0 * ebn0_lin))


def foa_sigma(ebn0_lin, duration_s):
    """Frequency of a tone observed for T seconds, time from the middle: sqrt(3 / (2 pi^2 T^2 E/N0))."""
    return np.sqrt(3.0 / (2.0 * np.pi**2 * duration_s**2 * ebn0_lin))


def ebn0(cn0_dbhz, duration_s):
    """Signal energy over noise density for an observation of T seconds at a given C/N0."""
    return 10.0 ** (cn0_dbhz / 10.0) * duration_s

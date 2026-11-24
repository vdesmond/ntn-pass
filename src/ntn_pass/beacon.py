"""406 MHz distress beacons: waveforms, burst schedules and carrier stability.

Sources (all March 2021 issues, the ones available):
- C/S T.001 Issue 4 Rev.7, first-generation beacons (FGB): 35-39 dBm into 50 ohm (2.3.2), antenna gain
  -3 to +4 dBi between 5 and 60 deg elevation (2.3.3), 160 ms unmodulated carrier then 144 bits at 400 bps
  for a long message, 520 ms in all (2.2.2-2.2.4), biphase-L (2.3.5), +-1.1 rad phase modulation with
  150 +- 100 us rise/fall times (2.3.6), 15 bit-sync ones then frame sync 000101111 (2.2.8), bursts every
  47.5-52.5 s (2.2.1), frequency residual variation 3e-9 and mean slope 1e-9/min (2.3.1).
- C/S T.018 Issue 1 Rev.7, second-generation beacons (SGB): 1 s bursts (2.2.2), DSSS-OQPSK at 38,400
  chips/s per component, 300 bps (2.2.3, 2.3.3), EIRP 34-45 dBm by elevation (Table 2.5), output filters
  include half-sine (2.3.2), bursts every 5 s for the first 30 s, then every 30 s (Table 2.1).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import bounds
from .link import PASSBAND_HZ

FS = {"fgb": 200e3, "sgb": 307.2e3}  # sample rates used for the waveforms (well above their bandwidths)

# T.001 2.3.1 medium-term stability (non-ELT(DT)) and 2.2.1 repetition period
FGB_RESIDUAL_FREQ = 3e-9
FGB_MEAN_SLOPE_PER_MIN = 1e-9
FGB_REPETITION_S = 50.0


@dataclass(frozen=True)
class Beacon:
    name: str
    kind: str  # "fgb" or "sgb"
    eirp_dbw: object  # callable el_deg -> dBW


def fgb(power_dbm=37.0, gain_dbi=2.0, name="FGB, T.019 nominal"):
    """T.019 Table A.2 nominal: 7 dBW and 2 dBi. Worst compliant: 35 dBm and -3 dBi."""
    return Beacon(name, "fgb", lambda el: np.full(np.shape(el), power_dbm - 30.0 + gain_dbi))


FGB_WORST = fgb(35.0, -3.0, "FGB, worst compliant (35 dBm, -3 dBi)")

_SGB_EL = np.arange(10.0, 86.0, 5.0)
_SGB_MIN_DBM = np.array([34] * 12 + [33] * 4, dtype=float)  # T.018 Table 2.5, minimum EIRP


def sgb_min():
    """Worst compliant SGB: the T.018 Table 2.5 minimum EIRP (held at 34 dBm below 10 deg)."""
    return Beacon("SGB, minimum EIRP", "sgb", lambda el: np.interp(el, _SGB_EL, _SGB_MIN_DBM) - 30.0)


def worst(kind):
    """The weakest compliant beacon of a kind."""
    return FGB_WORST if kind == "fgb" else sgb_min()


def burst_times(kind, horizon_s):
    """Burst emission times after activation: FGB every ~50 s (T.001 2.2.1); SGB every 5 s for 30 s, then
    every 30 s (T.018 Table 2.1)."""
    if kind == "fgb":
        return np.arange(0.0, horizon_s + 1e-9, FGB_REPETITION_S)
    early = np.arange(0.0, 30.0, 5.0)
    return np.concatenate([early, np.arange(30.0, horizon_s + 1e-9, 30.0)])


def fgb_bits(rng):
    """144-bit long message: 15 ones (bit sync), 000101111 (frame sync), then 120 message bits (fixed
    pseudo-random: the CRB depends on the number of transitions, not on what the bits mean)."""
    head = [1] * 15 + [0, 0, 0, 1, 0, 1, 1, 1, 1]
    return np.array(head + list(rng.integers(0, 2, 120)), dtype=int)


def fgb_waveform(fs, rise_s=150e-6, rng=None):
    """Complex baseband (unit power) of a long-message FGB burst: 160 ms carrier, then 144 biphase-L bits
    at 400 bps phase-modulating the carrier by +-1.1 rad, with linear phase ramps of rise_s (10-90 % of a
    linear ramp is 80 % of it, so the full ramp is rise_s / 0.8)."""
    rng = np.random.default_rng(0) if rng is None else rng
    bits = fgb_bits(rng)
    tb = 1.0 / 400.0
    halves = np.repeat(np.where(bits == 1, 1.0, -1.0), 2) * np.tile([1.0, -1.0], len(bits))  # biphase-L
    n_pre = round(0.160 * fs)
    n_half = round(tb / 2 * fs)
    m = np.concatenate([np.zeros(n_pre), np.repeat(halves, n_half)])
    ramp = rise_s / 0.8
    k = max(1, round(ramp * fs))
    m_smooth = np.convolve(m, np.ones(k) / k, mode="same")  # linear ramps between levels
    m_smooth[:n_pre] = 0.0
    return np.exp(1j * 1.1 * m_smooth)


def sgb_waveform(fs, rng=None, duration_s=1.0, chip_rate=38_400.0, passband_hz=PASSBAND_HZ):
    """Complex baseband (unit power) of an SGB burst: OQPSK with half-sine chips at 38,400 chips/s per
    component, Q delayed by half a chip, band-limited to the repeater passband. Chips are pseudo-random
    (T.018's PRN segments matter for acquisition, not for the bound)."""
    rng = np.random.default_rng(1) if rng is None else rng
    n_chip = round(duration_s * chip_rate)
    n = round(duration_s * fs)
    t = np.arange(n) / fs

    def comp(offset):
        c = 2 * rng.integers(0, 2, n_chip + 1) - 1.0
        idx = np.floor(t * chip_rate - offset).astype(int)
        ph = t * chip_rate - offset - idx
        return np.where(idx >= 0, c[np.clip(idx, 0, n_chip)], 0.0) * np.sin(np.pi * ph)

    s = comp(0.0) + 1j * comp(0.5)
    S = np.fft.fft(s)
    S[np.abs(np.fft.fftfreq(n, 1 / fs)) > passband_hz / 2] = 0.0
    s = np.fft.ifft(S)
    return s / np.sqrt(np.mean(np.abs(s) ** 2))


WAVEFORMS = {  # name -> (kind, waveform options)
    "fgb_150us": ("fgb", {"rise_s": 150e-6}),
    "fgb_250us": ("fgb", {"rise_s": 250e-6}),
    "sgb": ("sgb", {}),
}


def waveform(name):
    kind, kw = WAVEFORMS[name]
    return fgb_waveform(FS[kind], **kw) if kind == "fgb" else sgb_waveform(FS[kind])


@cache
def unit_crb():
    """Per waveform: (sigma TOA s, sigma FOA Hz) at 0 dB-Hz and the RMS bandwidth. The sigmas scale as
    10^(-C/N0 / 20)."""
    out = {}
    for name, (kind, _) in WAVEFORMS.items():
        s = waveform(name)
        out[name] = (*bounds.waveform_crb(s, FS[kind], 0.0), bounds.waveform_rms_bandwidth_hz(s, FS[kind]))
    return out


def crb_at(kind, cn0_dbhz):
    """(sigma TOA s, sigma FOA Hz) of one burst of the kind at C/N0 (FGB with the fastest 150 us edges)."""
    toa, foa, _ = unit_crb()["fgb_150us" if kind == "fgb" else "sgb"]
    g = 10 ** (-np.asarray(cn0_dbhz) / 20)
    return toa * g, foa * g

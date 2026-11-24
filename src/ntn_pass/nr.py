"""NR NTN: numerology, the timing fields the network broadcasts, and the SS/PBCH block the UE acquires.

Numerology (TS 38.211 V17.11.0 4.1): T_c = 1 / (df_max N_f), df_max = 480 kHz, N_f = 4096; kappa = T_s / T_c = 64.
TS 38.211 5.3.1: normal cyclic prefix 144 kappa 2^-mu T_c (16 kappa T_c longer on the first symbol of
each half subframe, not used here).

SS/PBCH block, one satellite, vacuum, single path. Clause references cite the TS 38.211 version each clause
was read in. The numerology and SSB clauses quoted here are unchanged between them. Sequences (TS 38.211
V17.6.0 7.4.2.2 PSS, 7.4.2.3 SSS)
are taken from the floating-point reference of nr-cell-search (github.com/vdesmond/nr-cell-search @ 10f8e1c,
python/nr_sync.py).

SSB numerology: band n256 (S-band NTN) carries the SSB at 15 kHz, Case A (TS 38.101-5 V17.5.0 Table 5.4.3.3-1).
nfft = 256 gives fs = 3.84 MHz, enough for the 240-subcarrier (3.6 MHz) block. The normal cyclic prefix
is 144 kappa 2^-mu T_c = 18 samples. The longer first-symbol CP of each half subframe does not fall on
SSB symbols 2-5 or 8-11, so it is not modelled.

Placement: the PSS and SSS occupy subcarriers 56..182 of the 240 (TS 38.211 Table 7.4.3.1-1) and the
SS raster points at subcarrier 120 (TS 38.101-1 V17.11.0 Table 5.4.3.2-1), so relative to the raster
frequency the 127 tones are k = -64..62. nr-cell-search arrived at the same -64 from a real capture.

Scaling: unitary FFT, so every resource element carries unit energy and complex noise of variance
sigma^2 per sample is sigma^2 per resource element after the FFT. The per-RE SNR is then 1 / sigma^2,
which equals the link-budget CNR when the carrier power is spread evenly over the carrier bandwidth.

Channel: the received baseband is
    r(t) = s(t - tau(t)) exp(-j 2 pi fc tau(t)) exp(j 2 pi df t) exp(j theta) + w(t),
with tau(t) the one-way delay along the pass (quadratic around the burst), df the residual frequency
offset left after any UE pre-compensation plus the satellite access node's own error, and theta a random
phase. The envelope delay and the carrier Doppler come from the same tau(t), so the Doppler-driven drift of
the arrival time between bursts is included.
"""

from dataclasses import dataclass

import numpy as np

from .constants import C

TC = 1.0 / (480e3 * 4096)
KAPPA = 64


def slot_s(mu):
    return 1e-3 / 2**mu


def cp_normal_s(mu):
    return 144 * KAPPA * 2.0**-mu * TC


# TS 38.133 V17.23.0 Table 7.1C.2-1, FR1, keyed by (SSB SCS, UL SCS) in kHz.
TE_NTN_S = {
    (15, 15): 29 * 64 * TC,
    (15, 30): 24 * 64 * TC,
    (30, 15): 24 * 64 * TC,
    (30, 30): 22 * 64 * TC,
}

# TS 38.331 V17.18.0 NTN-Config / TA-Info / EphemerisInfo (ranges and granularities).
TA_COMMON_LSB_S = 4.072e-3 * 1e-6
TA_COMMON_RANGE = (0, 66485757)
TA_DRIFT_LSB = 0.2e-3 * 1e-6  # s/s
TA_DRIFT_RANGE = (-257303, 257303)
TA_DRIFT_VAR_LSB = 0.2e-4 * 1e-6  # s/s^2
TA_DRIFT_VAR_RANGE = (0, 28949)
POS_LSB_M, POS_RANGE = 1.3, (-33554432, 33554431)
VEL_LSB_MS, VEL_RANGE = 0.06, (-131072, 131071)
K_OFFSET_RANGE = (1, 1023)  # cellSpecificKoffset, slots of 15 kHz (1 ms)
K_MAC_RANGE = (1, 512)  # kmac, slots of 15 kHz in FR1
UL_SYNC_VALIDITY_S = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 900)
HARQ_MAX = {"legacy": 16, "rel17": 32}  # nrofHARQ-ProcessesForPDSCH {..n16} and -v1700 {n32}


SCS = 15e3
NFFT = 256
FS = NFFT * SCS
NCP = round(cp_normal_s(0) * FS)  # 18
NSYM = NFFT + NCP
T_SYM = NSYM / FS
K_MIN = -64
TONES = np.arange(K_MIN, K_MIN + 127)  # PSS / SSS subcarriers relative to the raster
SSB_K = np.arange(-120, 120)  # the 240 SSB subcarriers relative to the raster
BURST_PERIOD_S = 20e-3  # TS 38.213 V17.7.0 4.1: initial cell selection assumes a 2-frame SSB periodicity
SSB_SYMBOLS = 4


def pss(n_id2: int) -> np.ndarray:
    """Length-127 PSS in {+1, -1}."""
    if n_id2 not in (0, 1, 2):
        raise ValueError("N_ID2 must be 0, 1 or 2")
    x = np.zeros(127, dtype=np.uint8)
    x[:7] = [0, 1, 1, 0, 1, 1, 1]
    for n in range(120):
        x[n + 7] = x[n] ^ x[n + 4]
    d = np.roll(x, -(43 * n_id2))
    return (1 - 2 * d.astype(np.int8)).astype(float)


def sss(n_id1: int, n_id2: int) -> np.ndarray:
    """Length-127 SSS in {+1, -1}. N_ID2 enters through m0, so SSS detection needs the PSS result."""
    if n_id2 not in (0, 1, 2):
        raise ValueError("N_ID2 must be 0, 1 or 2")
    if not 0 <= n_id1 <= 335:
        raise ValueError("N_ID1 must be 0..335")
    c0 = np.zeros(127, dtype=np.uint8)
    c1 = np.zeros(127, dtype=np.uint8)
    c0[0] = c1[0] = 1
    for n in range(120):
        c0[n + 7] = c0[n] ^ c0[n + 4]
        c1[n + 7] = c1[n] ^ c1[n + 1]
    m0 = 15 * (n_id1 // 112) + 5 * n_id2
    m1 = n_id1 % 112
    d0 = np.roll(c0, -m0).astype(np.int8)
    d1 = np.roll(c1, -m1).astype(np.int8)
    return ((1 - 2 * d0) * (1 - 2 * d1)).astype(float)


def ssb_grid(n_id1: int, n_id2: int, rng) -> np.ndarray:
    """(4, 240) resource grid of one SSB: PSS, PBCH, SSS with PBCH on its edges, PBCH.

    PBCH resource elements carry random QPSK (data and DM-RS are not distinguished: the receiver here
    uses only the PSS and SSS, and the PBCH matters only as energy next to them).
    """
    g = np.zeros((SSB_SYMBOLS, 240), dtype=complex)
    k = TONES + 120
    g[0, k] = pss(n_id2)
    g[2, k] = sss(n_id1, n_id2)

    def qpsk(n):
        return ((2 * rng.integers(0, 2, n) - 1) + 1j * (2 * rng.integers(0, 2, n) - 1)) / np.sqrt(2)

    g[1] = qpsk(240)
    g[3] = qpsk(240)
    edge = np.r_[0:48, 192:240]  # TS 38.211 Table 7.4.3.1-1, PBCH in symbol 2
    g[2, edge] = qpsk(len(edge))
    return g


def pss_replica(n_id2: int) -> np.ndarray:
    """Time-domain PSS (useful part, no CP), unit energy per tone: ||p||^2 = 127."""
    X = np.zeros(NFFT, dtype=complex)
    X[TONES % NFFT] = pss(n_id2)
    return np.fft.ifft(X) * np.sqrt(NFFT)


@dataclass(frozen=True)
class Delay:
    """One-way delay around a reference time t_ref (receiver clock): tau(t) = d0 + d1 dt + d2 dt^2 / 2."""

    d0_s: float
    d1: float = 0.0  # s/s (= range rate / c)
    d2: float = 0.0  # 1/s
    t_ref_s: float = 0.0

    def __call__(self, t):
        dt = np.asarray(t) - self.t_ref_s
        return self.d0_s + self.d1 * dt + 0.5 * self.d2 * dt**2

    def rate(self, t):
        return self.d1 + self.d2 * (np.asarray(t) - self.t_ref_s)

    @classmethod
    def from_look(cls, range_m, range_rate, range_accel, t_ref_s=0.0):
        return cls(range_m / C, range_rate / C, range_accel / C, t_ref_s)


def received(grid, t_rx, t_tx0, delay: Delay, fc, df_hz=0.0, theta=0.0):
    """Noise-free received samples at receiver times t_rx for one SSB transmitted from t_tx0.

    Each sample is the transmitted OFDM signal evaluated at its emission time u = t - tau(t), so a
    fractional, time-varying delay is exact (no interpolation, no circular shift).
    """
    t_rx = np.asarray(t_rx, dtype=float)
    u = t_rx - delay(t_rx) - t_tx0
    s = np.floor(u / T_SYM).astype(int)
    v = u - s * T_SYM - NCP / FS  # time from the start of the useful part (negative inside the CP)
    inside = (s >= 0) & (s < SSB_SYMBOLS)
    out = np.zeros(len(t_rx), dtype=complex)
    if inside.any():
        ph = np.exp(2j * np.pi * np.outer(v[inside], SSB_K * SCS))  # (n, 240)
        out[inside] = np.sum(ph * grid[s[inside]], axis=1) / np.sqrt(NFFT)
    carrier = np.exp(-2j * np.pi * fc * (delay(t_rx) - delay.d0_s) + 2j * np.pi * df_hz * t_rx + 1j * theta)
    return out * carrier


def awgn(n, snr_re_db, rng):
    """Complex white noise with per-RE SNR snr_re_db for unit-energy resource elements."""
    sigma2 = 10.0 ** (-snr_re_db / 10.0)
    return np.sqrt(sigma2 / 2.0) * (rng.standard_normal(n) + 1j * rng.standard_normal(n)), sigma2

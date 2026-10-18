"""SSB acquisition: layout against TS 38.211 / 38.101, exact noise-free recovery of a fractional, drifting
delay and a Doppler offset, the CFO loss of the matched filter, the noise-only statistic behind the
threshold, drift-compensated combining, efficiency of the fine estimator against its CRB, and Monte Carlo
detection against the noncentral chi-square."""

import numpy as np
import pytest
from scipy import stats

from ntn_pass import acquire as D
from ntn_pass import bounds, nr
from ntn_pass.studies import acquisition

FC = 2e9
N_ID1, N_ID2 = 17, 1


def _arrival(dl, t_tx0=0.0):
    """Receiver time at which the PSS useful part starts arriving: t - tau(t) = t_tx0 + CP."""
    t = dl.d0_s
    for _ in range(30):
        t = t_tx0 + nr.NCP / nr.FS + dl(t)
    return t


def _burst(dl, df, frac, rng, t_tx0=0.0, n=1600, snr_db=None, theta=0.7):
    """Window opened `frac` samples before the CP of the PSS; noise-free unless snr_db is given."""
    g = nr.ssb_grid(N_ID1, N_ID2, rng)
    t0 = _arrival(dl, t_tx0) - (frac + nr.NCP) / nr.FS
    t = t0 + np.arange(n) / nr.FS
    rx = nr.received(g, t, t_tx0, dl, FC, df, theta)
    sigma2 = 1e-3
    if snr_db is not None:
        w, sigma2 = nr.awgn(n, snr_db, rng)
        rx = rx + w
    return rx, t0, sigma2


def _truth(dl, df, e, t_tx0=0.0):
    tau_hat = e.toa_s - (t_tx0 + nr.NCP / nr.FS)
    return tau_hat - dl(e.t_toa_s), e.foa_hz - (-FC * dl.rate(e.t_foa_s) + df)


def test_numerology():
    assert nr.NCP == 18 and nr.cp_normal_s(0) * nr.FS == pytest.approx(18.0)
    # PSS/SSS on SSB subcarriers 56..182, SS raster at subcarrier 120
    assert (nr.TONES + 120)[0] == 56 and (nr.TONES + 120)[-1] == 182
    assert bounds.ssb_rms_bandwidth_hz() == pytest.approx(15e3 * np.sqrt((127**2 - 1) / 12))


def test_sequences():
    p = [nr.pss(i) for i in range(3)]
    for i in range(3):
        assert p[i] @ p[i] == 127
        for j in range(i):
            assert abs(p[i] @ p[j]) <= 9  # m-sequence shifts by 43: cross-correlation -1 (cyclic) / small
    cells = {nr.sss(a, b).tobytes() for a in range(336) for b in range(3)}
    assert len(cells) == 1008


@pytest.mark.parametrize(
    "rate,accel,df,frac",
    [(0.0, 0.0, 1234.0, 37.3), (-6700.0, 50.0, 0.0, 37.3), (6700.0, 50.0, 80.0, 12.71), (-7200.0, 0.0, -100.0, 3.5)],
)
def test_noise_free_recovery(rate, accel, df, frac):
    rng = np.random.default_rng(1)
    dl = nr.Delay.from_look(600e3, rate, accel)
    rx, t0, s2 = _burst(dl, df, frac, rng)
    freqs = D.freq_grid(50e3, 5e3)
    T = D.search([rx], s2, freqs, FC, t0s=[t0])
    a, b, lag = np.unravel_index(np.argmax(T), T.shape)
    assert a == N_ID2
    assert abs(lag - (frac + nr.NCP)) <= 1
    e = D.fine(rx, lag, freqs[b], N_ID1, N_ID2, FC, t0_s=t0)
    dtau, dfoa = _truth(dl, df, e)
    assert abs(dtau) < 0.2e-9
    assert abs(dfoa) < 0.2


def test_cfo_loss():
    """Matched-filter magnitude against CFO is the energy-weighted phasor sum over the replica, and the
    loss at half a subcarrier is ~3.9 dB (|sinc(1/2)|)."""
    p = nr.pss_replica(N_ID2)
    m = np.arange(nr.NFFT)
    for d in (0.0, 2e3, 7.5e3, 12e3):
        x = p * np.exp(2j * np.pi * d * m / nr.FS)
        got = abs(np.vdot(p, x)) / 127.0
        want = abs(np.sum(np.abs(p) ** 2 * np.exp(2j * np.pi * d * m / nr.FS))) / 127.0
        assert got == pytest.approx(want, rel=1e-9)
    x = p * np.exp(2j * np.pi * 7.5e3 * m / nr.FS)
    loss_db = -20 * np.log10(abs(np.vdot(p, x)) / 127.0)
    assert loss_db == pytest.approx(-20 * np.log10(np.sinc(0.5)), abs=0.5)


def test_noise_only_statistic(rng):
    """Normalised statistic is Exp(1) per cell: mean 1 and the tail matches at the 1 % point."""
    n = 1 << 16
    w, s2 = nr.awgn(n, 0.0, rng)
    T = D.search([w], s2, [0.0], FC, n_id2_set=(0,))[0, 0][:: nr.NFFT]  # ~independent lags
    assert T.mean() == pytest.approx(1.0, abs=4 / np.sqrt(len(T)))
    thr = D.threshold(0.01, 1)
    hits = int(np.sum(T > thr))
    lo, hi = stats.binom.ppf([0.0005, 0.9995], len(T), 0.01)
    assert lo <= hits <= hi
    assert D.threshold(0.01, 1, 4) == pytest.approx(stats.gamma.isf(0.01, 4))


def test_drift_compensated_combining():
    """Four bursts 20 ms apart at 22 ppm drift (1.7 samples per burst): with the drift implied by the
    frequency hypothesis removed, the combined peak is ~4x one burst; without, it smears."""
    rng = np.random.default_rng(2)
    dl = nr.Delay.from_look(600e3, -6700.0, 0.0)
    f_true = -FC * dl.d1
    bursts, t0s = [], []
    for k in range(4):
        t_tx0 = k * nr.BURST_PERIOD_S
        g = nr.ssb_grid(N_ID1, N_ID2, rng)
        t0 = t_tx0 + dl.d0_s  # receiver opens each window at the nominal (first-burst) delay
        t = t0 + np.arange(1600) / nr.FS
        bursts.append(nr.received(g, t, t_tx0, dl, FC))
        t0s.append(t0)
    freqs = np.array([f_true])
    one = D.search(bursts[:1], 1.0, freqs, FC, (N_ID2,), t0s=t0s[:1]).max()
    on = D.search(bursts, 1.0, freqs, FC, (N_ID2,), drift_comp=True, t0s=t0s).max()
    off = D.search(bursts, 1.0, freqs, FC, (N_ID2,), drift_comp=False, t0s=t0s).max()
    assert on > 0.95 * 4 * one
    assert off < 0.85 * 4 * one


def test_fine_estimator_reaches_crb(rng):
    """At 10 dB per RE the (TOA, FOA) error is efficient: NEES over N trials is chi-square(2N)."""
    snr_db, n = 10.0, 400
    Pinv = np.diag(1 / np.square(bounds.ssb_crb(10 ** (snr_db / 10))))  # TOA and FOA decouple
    nees = 0.0
    for _ in range(n):
        dl = nr.Delay.from_look(600e3 + rng.uniform(0, 3e3), rng.uniform(-7e3, 7e3), 40.0)
        df = rng.uniform(-100, 100)
        rx, t0, _ = _burst(dl, df, rng.uniform(10, 30), rng, snr_db=snr_db)
        n0 = round((_arrival(dl) - t0) * nr.FS)
        f_hyp = np.round((-FC * dl.d1 + df) / 5e3) * 5e3
        e = D.fine(rx, n0, f_hyp, N_ID1, N_ID2, FC, t0_s=t0)
        err = np.array(_truth(dl, df, e))
        nees += err @ Pinv @ err
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], 2 * n)
    assert lo < nees < hi


@pytest.mark.parametrize("snr_db,k", [(-9.0, 1), (-8.0, 1), (-13.0, 4)])
def test_detection_matches_noncentral_chi2(snr_db, k):
    """At the true lag with no frequency or timing offset, 2T ~ ncx2(2K, 254 K SNR): Monte Carlo Pd against
    the analytic value at the warm-search threshold. This ties the detection tables to the per-RE SNR."""
    rng = np.random.default_rng(3)
    thr = D.threshold(acquisition.PFA, D.n_cells(2 * nr.NSYM, 1), k)
    g = nr.ssb_grid(N_ID1, N_ID2, rng)
    n = 700
    dl = nr.Delay(100 / nr.FS)  # integer-sample delay: the PSS useful part starts at sample 118
    clean = nr.received(g, np.arange(n) / nr.FS, 0.0, dl, FC)
    p = nr.pss_replica(N_ID2)
    trials, hits = 3000, 0
    for _ in range(trials):
        t = 0.0
        for _ in range(k):
            w, s2 = nr.awgn(n, snr_db, rng)
            t += abs(np.vdot(p, (clean + w)[118 : 118 + nr.NFFT])) ** 2 / (s2 * 127.0)
        hits += t > thr
    want = D.pd_aligned(snr_db, thr, k)
    lo, hi = stats.binom.ppf([0.0005, 0.9995], trials, want)
    assert lo <= hits <= hi

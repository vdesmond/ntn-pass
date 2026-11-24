"""Cramer-Rao bounds on TOA / FOA: the SSB and the 406 MHz waveforms against the closed forms they reduce to."""

import numpy as np
import pytest

from ntn_pass import beacon, bounds, nr


def test_closed_forms():
    # 1 MHz rms bandwidth, E/N0 = 1000 (30 dB): 1 / (2 pi 1e6 sqrt(2000))
    assert bounds.toa_sigma(1000.0, 1e6) == pytest.approx(3.5588e-9, rel=1e-4)
    # 1 s tone at E/N0 = 1000: sqrt(3 / (2 pi^2 1000))
    assert bounds.foa_sigma(1000.0, 1.0) == pytest.approx(0.012328, rel=1e-4)
    assert bounds.ebn0(30.0, 2.0) == pytest.approx(2000.0)


def test_ssb_crb_closed_form():
    """Common delay over 2 x 127 tones and a frequency from two symbols 2 T_sym apart decouple: each matches
    its one-parameter bound exactly."""
    snr = 2.0
    toa, foa = bounds.ssb_crb(snr)
    assert toa == pytest.approx(bounds.toa_sigma(254 * snr, bounds.ssb_rms_bandwidth_hz()), rel=1e-9)
    assert foa == pytest.approx(1 / (2 * np.pi * nr.T_SYM * np.sqrt(2 * 254 * snr)), rel=1e-9)


def test_waveform_crb_closed_forms():
    """FOA of a constant-envelope burst matches the tone bound with E/N0 = C/N0 T (phase unknown, time from
    the middle). TOA of the SGB matches the RMS-bandwidth bound. Within a few percent: the coupling to the
    other parameters is small."""
    cn0 = 35.0
    for name in ("fgb_150us", "sgb"):
        kind = beacon.WAVEFORMS[name][0]
        s, fs = beacon.waveform(name), beacon.FS[kind]
        T = len(s) / fs
        _, foa = bounds.waveform_crb(s, fs, cn0)
        assert foa == pytest.approx(bounds.foa_sigma(bounds.ebn0(cn0, T), T), rel=0.05)
    s, fs = beacon.waveform("sgb"), beacon.FS["sgb"]
    toa, _ = bounds.waveform_crb(s, fs, cn0)
    assert toa == pytest.approx(
        bounds.toa_sigma(bounds.ebn0(cn0, 1.0), bounds.waveform_rms_bandwidth_hz(s, fs)), rel=0.05
    )


def test_crb_scales_with_cn0():
    s, fs = beacon.waveform("sgb"), beacon.FS["sgb"]
    a, b = bounds.waveform_crb(s, fs, 30.0), bounds.waveform_crb(s, fs, 40.0)
    assert np.allclose(np.array(a) / np.array(b), 10**0.5)

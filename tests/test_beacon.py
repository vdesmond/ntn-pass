"""406 MHz beacons: the FGB waveform layout from T.001, the burst schedules, and where the timing information
of a burst comes from."""

import numpy as np
import pytest

from ntn_pass import beacon, bounds


def test_fgb_waveform_layout():
    fs = 200e3
    s = beacon.fgb_waveform(fs)
    assert len(s) / fs == pytest.approx(0.160 + 144 / 400, abs=1e-4)
    assert np.allclose(np.abs(s), 1.0)
    assert np.allclose(np.angle(s[: int(0.150 * fs)]), 0.0)  # unmodulated carrier
    assert np.max(np.abs(np.angle(s))) == pytest.approx(1.1, abs=1e-9)
    bits = beacon.fgb_bits(np.random.default_rng(0))
    assert list(bits[:24]) == [1] * 15 + [0, 0, 0, 1, 0, 1, 1, 1, 1]


def test_fgb_toa_grows_with_edge_time():
    """TOA information comes from the phase transitions: slower edges, larger bound."""
    a = bounds.waveform_crb(beacon.fgb_waveform(200e3, 150e-6), 200e3, 35.0)[0]
    b = bounds.waveform_crb(beacon.fgb_waveform(200e3, 250e-6), 200e3, 35.0)[0]
    assert b > a


def test_burst_schedules():
    assert list(beacon.burst_times("fgb", 600)) == list(np.arange(0, 601, 50.0))  # T.001: ~50 s
    sgb = beacon.burst_times("sgb", 300)
    assert list(sgb[:7]) == [0, 5, 10, 15, 20, 25, 30]  # T.018 Table 2.1: 5 s for 30 s, then 30 s
    assert np.all(np.diff(sgb[6:]) == 30.0)
    assert len(beacon.burst_times("sgb", 30)) == 7 and len(sgb) == 16

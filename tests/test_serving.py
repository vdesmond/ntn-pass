import numpy as np
import pytest

from ntn_pass import serving


def _toy():
    """Three satellites rising one after another: triangular elevation profiles."""
    t = np.arange(0.0, 3600.0, 10.0)
    el = np.stack([60.0 - np.abs(t - c) / 15.0 for c in (600.0, 1500.0, 2400.0)])
    return t, el


def test_max_elevation_switches_at_crossovers():
    t, el = _toy()
    tl = serving.timeline(t, el, mask_deg=10.0, policy="max-elevation")
    assert np.allclose(tl.handover_s, [1060.0, 1960.0])  # first sample past each midpoint
    assert set(tl.serving[tl.serving >= 0]) == {0, 1, 2}


def test_until_mask_holds_satellite():
    t, el = _toy()
    tl = serving.timeline(t, el, mask_deg=10.0, policy="until-mask")
    # satellite 0 drops below 10 deg at t = 600 + 750 = 1350 s
    assert tl.handover_s[0] == pytest.approx(1350.0, abs=10.0)
    assert len(tl.handover_s) == 2


def test_hysteresis_reduces_handovers():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 3600.0, 10.0)
    el = 40.0 + rng.normal(0.0, 3.0, (2, len(t)))  # two satellites at similar elevation, jittering
    n0 = len(serving.timeline(t, el, policy="max-elevation").handover_s)
    n5 = len(serving.timeline(t, el, policy="max-elevation", hysteresis_deg=5.0).handover_s)
    assert n5 < n0


def test_outage_when_nothing_above_mask():
    t = np.arange(0.0, 100.0, 10.0)
    tl = serving.timeline(t, np.full((2, len(t)), 5.0), mask_deg=10.0)
    assert tl.outage_fraction == 1.0 and len(tl.handover_s) == 0

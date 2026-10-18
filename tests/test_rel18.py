"""Rel-18 location verification inputs: the RTT noise read from TS 38.133 and the measurement layout."""

import numpy as np
import pytest

from ntn_pass import nr, rel18


def test_ts38133_sigma():
    # (59 + 80) Tc as a 90 % two-sided bound, two legs
    assert rel18.SIGMA_RTT_S == pytest.approx(np.sqrt(2) * 139 * nr.TC / 1.6448536, rel=1e-6)


def test_measurement_layout():
    meas, st = rel18.measurements(45.0, 60.0)
    assert len(st.t_s) == 7 and len(meas) == 14  # one RTT + Doppler pair every 10 s
    assert np.all(meas.sigma[0::2] == rel18.SIGMA_RTT_S)
    floor, _ = rel18.measurements(45.0, 60.0, floor_m=10.0)
    assert np.all(floor.sigma[0::2] > meas.sigma[0::2])

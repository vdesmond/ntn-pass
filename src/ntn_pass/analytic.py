"""Closed-form circular-orbit geometry, the model behind the 3GPP NTN reference values.

Spherical Earth of radius R_EARTH_TR, UE fixed in the orbital plane, Earth rotation neglected
(TR 38.811 sections 5.3.4.3-5.3.4.4). Used to check the TR tables and to bound the SGP4 results.
"""

import numpy as np

from .constants import MU_EARTH, R_EARTH_TR, C


def slant_range(h_m, el_deg, r_e=R_EARTH_TR):
    """Distance UE-satellite at elevation el for altitude h (law of cosines on the Earth-centre triangle)."""
    s = np.sin(np.deg2rad(el_deg))
    return np.sqrt((r_e * s) ** 2 + h_m**2 + 2.0 * r_e * h_m) - r_e * s


def orbital_speed(h_m, r_e=R_EARTH_TR):
    return np.sqrt(MU_EARTH / (r_e + h_m))


def angular_rate(h_m, r_e=R_EARTH_TR):
    return orbital_speed(h_m, r_e) / (r_e + h_m)


def doppler_ppm(h_m, el_deg, r_e=R_EARTH_TR):
    """|Doppler| / fc for an in-plane UE: omega * R_E * cos(el) / c."""
    return 1e6 * angular_rate(h_m, r_e) * r_e * np.cos(np.deg2rad(el_deg)) / C


def doppler_rate_ppm_s_zenith(h_m, r_e=R_EARTH_TR):
    """|d(Doppler)/dt| / fc at closest approach of an overhead pass: omega^2 R_E (R_E + h) / (h c).

    This is the maximum of the Doppler rate magnitude, reached where the Doppler shift is zero.
    """
    w = angular_rate(h_m, r_e)
    return 1e6 * w**2 * r_e * (r_e + h_m) / (h_m * C)


def max_rtt_s(h_m, el_min_deg, links=1):
    """Round-trip propagation delay at the minimum elevation, and links=2 adds an equally long feeder link
    (transparent payload, TR 38.821 Scenarios A/C)."""
    return 2.0 * links * slant_range(h_m, el_min_deg) / C

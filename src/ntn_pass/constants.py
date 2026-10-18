"""Physical constants (SI unless noted)."""

C = 299_792_458.0  # speed of light, m/s

# WGS-72 gravity values are what SGP4 is fitted with; WGS-84 is used for the ground site.
MU_EARTH = 3.986004418e14  # m^3/s^2 (WGS-84)
OMEGA_EARTH = 7.292115146706979e-5  # rad/s, Earth rotation rate used with GMST82 (Vallado)

WGS84_A = 6_378_137.0  # m
WGS84_F = 1.0 / 298.257223563

# Spherical Earth radius used by the 3GPP TRs (TR 38.821 Table 4.2-2 values reproduce with this).
R_EARTH_TR = 6_371_000.0  # m

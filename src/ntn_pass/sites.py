"""Ground sites (UE / ground station)."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .frames import enu_matrix, geodetic_to_ecef


@dataclass(frozen=True)
class Site:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float

    @cached_property
    def ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)

    @cached_property
    def enu(self) -> np.ndarray:
        return enu_matrix(self.lat_deg, self.lon_deg)


# Approximate coordinates (EURECOM campus; Toulouse city centre).
SOPHIA = Site("Sophia Antipolis", 43.6145, 7.0715, 130.0)
TOULOUSE = Site("Toulouse", 43.6045, 1.4440, 150.0)

SITES = {"sophia": SOPHIA, "toulouse": TOULOUSE}

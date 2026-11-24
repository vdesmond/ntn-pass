"""Committed TLE snapshots and the filter down to operational satellites."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, SatrecArray

DATA = Path(__file__).resolve().parents[2] / "data" / "tle"

SNAPSHOT = "2026-10-13"
# Analyses start at the snapshot date so SGP4 runs within ~1 day of the element epochs.
T0 = datetime(2026, 10, 13, 0, 0, 0, tzinfo=UTC)

R_EQ_KM = 6378.137
MU_KM = 398600.4418


@dataclass(frozen=True)
class Constellation:
    name: str
    group: str  # Celestrak GROUP
    alt_km: tuple[float, float]  # keep mean altitudes in this band (operational shells)
    max_ecc: float
    max_epoch_age_d: float  # drop elements this much older than the newest in the file (stale)


CONSTELLATIONS = {
    # Starlink: most of the fleet sits at ~466 km mean altitude (some at 530-570 km). 340-370 km is
    # orbit raising / very-low shells and below 300 km is deorbiting (2026-10-13 snapshot).
    "starlink": Constellation("Starlink", "starlink", (440.0, 600.0), 0.005, 3.0),
    "oneweb": Constellation("OneWeb", "oneweb", (1100.0, 1300.0), 0.005, 3.0),
    # Galileo: drop the two satellites in eccentric orbits (GSAT0201/0202). MEO elements are
    # published every few weeks (no drag), so a month-old element set is still current.
    "galileo": Constellation("Galileo", "galileo", (23000.0, 23400.0), 0.01, 30.0),
}


@dataclass
class Fleet:
    constellation: Constellation
    names: list[str]
    sats: list[Satrec]
    lines: list[tuple[str, str]]

    def __len__(self):
        return len(self.sats)

    def array(self) -> SatrecArray:
        return SatrecArray(self.sats)

    def subset(self, idx) -> "Fleet":
        idx = list(idx)
        return Fleet(
            self.constellation,
            [self.names[i] for i in idx],
            [self.sats[i] for i in idx],
            [self.lines[i] for i in idx],
        )


def read_tle(path: Path):
    rows = [r.rstrip() for r in Path(path).read_text().splitlines() if r.strip()]
    out = []
    for i in range(0, len(rows) - 2, 3):
        name, l1, l2 = rows[i].strip(), rows[i + 1], rows[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            raise ValueError(f"{path}: bad TLE at line {i + 1}")
        out.append((name, l1, l2))
    return out


def mean_altitude_km(sat: Satrec) -> float:
    n = sat.no_kozai / 60.0  # rad/s
    return (MU_KM / n**2) ** (1.0 / 3.0) - R_EQ_KM


def epoch_jd(sat: Satrec) -> float:
    return sat.jdsatepoch + sat.jdsatepochF


def load(key: str, snapshot: str = SNAPSHOT, operational: bool = True) -> Fleet:
    c = CONSTELLATIONS[key]
    recs = read_tle(DATA / f"{c.group}-{snapshot}.tle")
    sats = [Satrec.twoline2rv(l1, l2) for _, l1, l2 in recs]
    keep = range(len(sats))
    if operational:
        newest = max(epoch_jd(s) for s in sats)
        keep = [
            i
            for i, s in enumerate(sats)
            if c.alt_km[0] <= mean_altitude_km(s) <= c.alt_km[1]
            and s.ecco <= c.max_ecc
            and newest - epoch_jd(s) <= c.max_epoch_age_d
        ]
    return Fleet(
        c,
        [recs[i][0] for i in keep],
        [sats[i] for i in keep],
        [(recs[i][1], recs[i][2]) for i in keep],
    )


def time_grid(duration_s: float, step_s: float, t0: datetime = T0):
    """Seconds from t0 and the matching (jd, fr) split that SGP4 wants."""
    t = np.arange(0.0, duration_s + 1e-9, step_s)
    return t, *jd_split(t, t0)


def jd_split(t_s, t0: datetime = T0):
    midnight = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    jd0 = 2440587.5 + midnight.timestamp() / 86400.0  # Unix epoch -> JD
    fr0 = (t0 - midnight).total_seconds() / 86400.0
    t_s = np.asarray(t_s, dtype=float)
    return np.full(t_s.shape, jd0), fr0 + t_s / 86400.0


def utc(t_s: float, t0: datetime = T0) -> datetime:
    return t0 + timedelta(seconds=float(t_s))

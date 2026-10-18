"""Serving-satellite timeline and handover counts.

The handover rate depends entirely on the selection policy, so the policy is explicit:

- ``max-elevation``: serve the highest satellite; switch when another one is higher than the current
  one by more than ``hysteresis_deg`` (0 = always the highest).
- ``until-mask``: keep the current satellite until it drops below the mask, then take the highest.
"""

from dataclasses import dataclass

import numpy as np

POLICIES = ("max-elevation", "until-mask")


@dataclass
class Timeline:
    policy: str
    mask_deg: float
    hysteresis_deg: float
    t_s: np.ndarray
    serving: np.ndarray  # satellite index per sample, -1 = outage
    handover_s: np.ndarray  # times of satellite-to-satellite switches

    @property
    def hours(self):
        return (self.t_s[-1] - self.t_s[0]) / 3600.0

    @property
    def handovers_per_hour(self):
        return len(self.handover_s) / self.hours

    @property
    def outage_fraction(self):
        return float(np.mean(self.serving < 0))

    def dwell_s(self):
        """Time spent on each serving satellite (complete dwells only)."""
        h = self.handover_s
        return np.diff(h) if len(h) > 1 else np.array([])

    def describe(self):
        pol = self.policy + (f", hysteresis {self.hysteresis_deg:g} deg" if self.policy == "max-elevation" else "")
        return f"{pol}, mask {self.mask_deg:g} deg"


def timeline(t_s, el, mask_deg=10.0, policy="max-elevation", hysteresis_deg=0.0) -> Timeline:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    el = np.where(el >= mask_deg, el, -np.inf)
    best = np.argmax(el, axis=0)
    best_el = el[best, np.arange(el.shape[1])]
    serving = np.full(el.shape[1], -1)
    handovers = []
    cur = -1
    for k in range(el.shape[1]):
        if not np.isfinite(best_el[k]):
            cur = -1
        elif cur < 0 or not np.isfinite(el[cur, k]):
            if cur >= 0:
                handovers.append(t_s[k])
            cur = best[k]
        elif policy == "max-elevation" and best[k] != cur and best_el[k] > el[cur, k] + hysteresis_deg:
            handovers.append(t_s[k])
            cur = best[k]
        serving[k] = cur
    return Timeline(policy, mask_deg, hysteresis_deg, np.asarray(t_s), serving, np.asarray(handovers))


def in_view(el, mask_deg=10.0):
    """Number of satellites above the mask at each sample."""
    return np.sum(el >= mask_deg, axis=0)

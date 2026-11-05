"""Mass-conserving ballistic water slices; no Isaac imports or fluid solver."""

from dataclasses import dataclass
from math import isfinite, pi, sqrt

from config import (
    CLOSED_ANGLE, CUP_CAPACITY_G, CUP_FLOOR_Z, CUP_RADIUS,
    DENSITY_G_M3, EXIT_SPEED, GRAVITY, MAX_ANGLE, MAX_FLOW_G_S, NOZZLE,
    RESERVOIR_FLOOR_Z, RESERVOIR_INITIAL_G, RESERVOIR_RADIUS,
    SCALE_RESOLUTION_G, SETTLE_S, TARGET_G, TOLERANCE_G,
)


def flow_g_s(angle):
    if not isfinite(angle):
        raise ValueError("Lever angle must be finite")
    opening = max(0.0, min(1.0, (abs(angle) - CLOSED_ANGLE) / (MAX_ANGLE - CLOSED_ANGLE)))
    return MAX_FLOW_G_S * opening


def flight_time(surface_z):
    return (sqrt(EXIT_SPEED**2 + 2 * GRAVITY * (NOZZLE[2] - surface_z)) - EXIT_SPEED) / GRAVITY


@dataclass
class Slice:
    """A constant-flow emission interval; its leading part can land before its tail."""

    start: float
    end: float
    rate: float

    @property
    def mass_g(self):
        return (self.end - self.start) * self.rate


class WaterTask:
    """Fixed, upright cup under the nozzle. Call step with the measured PhysX angle."""

    def __init__(self, target_g=TARGET_G):
        self.reset(target_g)

    def reset(self, target_g):
        if not isfinite(target_g) or not 0 < target_g <= CUP_CAPACITY_G:
            raise ValueError("Target must fit in the cup")
        self.target_g = target_g
        self.time = self.cup_g = self.spilled_g = self.emitted_g = 0.0
        self.flow = self.quiet_s = 0.0
        self.slices = []

    @property
    def surface_z(self):
        return CUP_FLOOR_Z + self.cup_g / (DENSITY_G_M3 * pi * CUP_RADIUS**2)

    @property
    def reservoir_g(self):
        return max(0.0, RESERVOIR_INITIAL_G - self.emitted_g)

    @property
    def reservoir_surface_z(self):
        return RESERVOIR_FLOOR_Z + self.reservoir_g / (DENSITY_G_M3 * pi * RESERVOIR_RADIUS**2)

    @property
    def airborne_g(self):
        return sum(s.mass_g for s in self.slices)

    @property
    def reading_g(self):
        return round(self.cup_g / SCALE_RESOLUTION_G) * SCALE_RESOLUTION_G

    @property
    def success(self):
        return (self.quiet_s >= SETTLE_S and self.spilled_g < 1e-9
                and abs(self.cup_g - self.target_g) <= TOLERANCE_G)

    def step(self, angle, dt):
        if not isfinite(dt) or dt <= 0:
            raise ValueError("Time step must be finite and positive")
        self.flow = min(flow_g_s(angle), self.reservoir_g / dt)
        end = self.time + dt
        if self.flow > 0:
            self.slices.append(Slice(self.time, end, self.flow))
            self.emitted_g += self.flow * dt
        self.time = end
        # Substep collision with the current water surface. A slice is credited
        # progressively as its emission-time interval crosses this surface.
        cutoff = self.time - flight_time(self.surface_z)
        landed = 0.0
        remaining = []
        for s in self.slices:
            crossed = min(s.end, max(s.start, cutoff))
            landed += (crossed - s.start) * s.rate
            s.start = crossed
            if s.end > s.start:
                remaining.append(s)
        self.slices = remaining
        accepted = min(landed, CUP_CAPACITY_G - self.cup_g)
        self.cup_g += accepted
        self.spilled_g += landed - accepted
        self.quiet_s = self.quiet_s + dt if self.flow == 0 and not self.slices else 0.0

    def cylinders(self):
        """Return (z_center, radius, length) with volume equal to each airborne mass."""
        result = []
        for s in self.slices:
            ages = (self.time - s.start, self.time - s.end)
            low, high = [NOZZLE[2] - EXIT_SPEED * age - 0.5 * GRAVITY * age**2 for age in ages]
            length = high - low
            radius = sqrt(s.mass_g / (DENSITY_G_M3 * pi * length))
            result.append(((low + high) / 2, radius, length))
        return result

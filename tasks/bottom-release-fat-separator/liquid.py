"""Phase-conserving reduced model of a gravity-fed bottom outlet."""

from collections import deque
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
DT = 1 / CONFIG["simulation"]["physics_hz"]
HZ = CONFIG["simulation"]["control_hz"]
LIMIT = CONFIG["episode"]["max_control_steps"]
FILLS = CONFIG["episode"]["fills_ml"]
CAMERAS = CONFIG["observation"]["camera_names"]
RADIUS = .049
AREA = math.pi * RADIUS**2
SUMP_RADIUS = .020
SUMP_HEIGHT = .030
SLOPE = (RADIUS - SUMP_RADIUS) / SUMP_HEIGHT
SUMP_ML = 1e6 * math.pi * (RADIUS**3 - SUMP_RADIUS**3) / (3 * SLOPE)
FLOOR = .164
OUTLET = .135
BOWL_FLOOR = .010
BOWL_RADIUS = .071
BOWL_AREA = math.pi * BOWL_RADIUS**2
BOWL_LIQUID_RADIUS = BOWL_RADIUS - .00035
VISIBLE_OIL_AREA_MM2 = 25.
STROKE = .016
SEAT = .0015
WATER_DENSITY_G_ML = 1.
OIL_DENSITY_G_ML = .91
FULL_RECOVERY = .90
QUIET = .35
INSTRUCTION = (
    "Drain the gravy into the bowl using the separator's green squeeze paddle. "
    "Recover at least 90% of the lower gravy layer with no visible oil on the bowl's surface. "
    "Release the paddle before the upper fat layer reaches the outlet and leave the valve shut."
)


def reservoir_radius(height):
    return SUMP_RADIUS + SLOPE * min(SUMP_HEIGHT, max(0., height))


def level(volume_ml):
    if volume_ml >= SUMP_ML:
        return SUMP_HEIGHT + (volume_ml - SUMP_ML) * 1e-6 / AREA
    radius = math.cbrt(SUMP_RADIUS**3 + 3 * SLOPE * volume_ml * 1e-6 / math.pi)
    return (radius - SUMP_RADIUS) / SLOPE


def bowl_oil_slick(fat_ml):
    """Return height and radius scale for the projected floating oil patch."""
    depth = fat_ml * 1e-6 / BOWL_AREA
    height = max(.0007, depth)
    return (height if depth else 0.), math.sqrt(depth / height)


class Drain:
    def __init__(self, seed):
        self.initial_gravy, self.initial_fat = FILLS[seed]
        self.gravy, self.fat = self.initial_gravy, self.initial_fat
        self.bowl_gravy = self.bowl_fat = 0.
        self.tail = deque()
        self.flow = self.quiet = self.stroke = 0.
        self.ticks = 0

    def step(self, stroke):
        self.ticks += 1
        self.stroke = max(0., stroke)
        aperture = min(1., max(0., (stroke - SEAT) / (STROKE - SEAT)))
        # Effective head includes the nozzle drop and the lighter upper layer.
        depth = level(self.gravy)
        head = FLOOR - OUTLET + depth + .91 * (level(self.gravy + self.fat) - depth)
        discharge = .64 * math.pi * .0038**2 * math.sqrt(2 * 9.81 * head) * 1e6
        self.flow = aperture * discharge
        # Selective withdrawal starts entraining oil as the interface reaches
        # the capture region of the inlet. Throttling reduces that region.
        capture = .0015 + .00053 * math.sqrt(self.flow)
        fraction_fat = max(0., 1. - depth / capture)**2
        amount = self.flow * DT
        fat = min(self.fat, amount * fraction_fat)
        gravy = min(self.gravy, amount - fat)
        fat = min(self.fat, amount - gravy)
        self.gravy -= gravy
        self.fat -= fat
        self.flow = (gravy + fat) / DT
        if self.flow > 0:
            surface = BOWL_FLOOR + (self.bowl_gravy + self.bowl_fat) * 1e-6 / BOWL_AREA
            v0 = math.sqrt(2 * 9.81 * head)
            self.tail.append((self.ticks, gravy, fat, surface, v0))
        while self.tail:
            tick, g, f, surface, v0 = self.tail[0]
            elapsed = (self.ticks - tick) * DT
            if v0 * elapsed + .5 * 9.81 * elapsed**2 < OUTLET - surface:
                break
            self.bowl_gravy += g
            self.bowl_fat += f
            self.tail.popleft()
        self.quiet = self.quiet + DT if stroke <= SEAT and not self.tail and self.flow == 0 else 0.

    @property
    def oil_area_mm2(self):
        _, scale = bowl_oil_slick(self.bowl_fat)
        return math.pi * (BOWL_LIQUID_RADIUS * scale)**2 * 1e6

    def score(self):
        if self.quiet < QUIET or self.oil_area_mm2 >= VISIBLE_OIL_AREA_MM2:
            return 0.
        water_g = self.bowl_gravy * WATER_DENSITY_G_ML
        minimum_g = self.initial_gravy * WATER_DENSITY_G_ML * FULL_RECOVERY
        if water_g >= minimum_g:
            return 1.
        return .95 * water_g / minimum_g

    def metrics(self):
        tail_g = sum(packet[1] for packet in self.tail)
        tail_f = sum(packet[2] for packet in self.tail)
        error = (self.gravy + self.bowl_gravy + tail_g - self.initial_gravy,
                 self.fat + self.bowl_fat + tail_f - self.initial_fat)
        assert max(map(abs, error)) < 1e-8, error
        return dict(time_s=self.ticks * DT, initial_gravy_ml=self.initial_gravy,
                    initial_fat_ml=self.initial_fat, gravy_in_bowl_ml=self.bowl_gravy,
                    initial_water_g=self.initial_gravy * WATER_DENSITY_G_ML,
                    initial_oil_g=self.initial_fat * OIL_DENSITY_G_ML,
                    minimum_water_g=self.initial_gravy * WATER_DENSITY_G_ML * FULL_RECOVERY,
                    water_in_bowl_g=self.bowl_gravy * WATER_DENSITY_G_ML,
                    oil_in_bowl_g=self.bowl_fat * OIL_DENSITY_G_ML,
                    oil_area_mm2=self.oil_area_mm2,
                    fat_in_bowl_ml=self.bowl_fat, gravy_left_ml=self.gravy,
                    fat_released_ml=self.initial_fat - self.fat,
                    gravy_airborne_ml=tail_g, fat_airborne_ml=tail_f,
                    closure_m=self.stroke, quiet_s=self.quiet, flow_ml_s=self.flow,
                    interface_height_mm=level(self.gravy)*1000,
                    recovery=self.bowl_gravy / self.initial_gravy,
                    phase_balance_error_ml=list(error), score=self.score())

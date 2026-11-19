"""Low-order cola/foam model. Volumes are mL, lengths are m, times are s."""
from collections import deque
from dataclasses import dataclass
from math import exp, pi

RADIUS = .035
HEIGHT = .105
AREA = pi * RADIUS**2 * 1e6
CAPACITY = AREA * HEIGHT
SERVICE_FULL = AREA * (HEIGHT - .010)
TARGET = .90 * SERVICE_FULL
DENSITY = 1.04  # g/mL, nominal cola. Real scoring uses tared mass.
OUTLET_ABOVE_FLOOR = .158
JET_RADIUS = .0045
# Frozen cross-section quadrature: a central streamlet and six around it.
JET_OFFSETS = ((0., 0.), (.0034, 0.), (-.0034, 0.), (.0017, .00294),
               (-.0017, .00294), (.0017, -.00294), (-.0017, -.00294))
TAIL_ML = 3.8
TAIL_TAU = .080
BUBBLE_RISE_TAU = .35
FOAM_LIQUID_PER_GAS = .13
# A volume tolerance for floating-point residue, far below one modeled streamlet.
LOSS_EPS_ML = 1e-8
OPEN_AT = .005
CLOSE_AT = .003


@dataclass(frozen=True)
class Condition:
    flow_ml_s: float
    entrained_gas_per_ml: float = .78
    drainage_tau_s: float = 2.65


CONDITIONS = (Condition(52.), Condition(74.))


class Pour:
    def __init__(self, condition):
        self.condition = condition
        self.time = self.travel = self.nozzle = self.landed = 0.
        self.open = False
        self.openings = 0
        self.tail = self.pending = self.liquid = self.gas = self.bubbles = 0.
        self.emitted = self.spill_liquid = self.spill_foam = self.intercepted = 0.
        self.blocked_now = 0.
        self.obstructions = ()
        self.crest = self.settled = 0.
        self.air = deque()
        self.exit_speed = condition.flow_ml_s / (pi * JET_RADIUS**2 * 1e6)

    @property
    def bulk(self):
        return max(0., self.liquid - FOAM_LIQUID_PER_GAS * self.gas)

    @property
    def top(self):
        return self.liquid + self.gas

    def falling_depth(self, age):
        return self.exit_speed * age + .5 * 9.81 * age**2

    def step(self, travel, dt, obstructions=()):
        self.time += dt
        self.travel = travel
        opened = travel >= (CLOSE_AT if self.open else OPEN_AT)
        if opened and not self.open:
            self.openings += 1
        self.open = opened
        if opened:
            self.nozzle = self.condition.flow_ml_s * dt
            self.tail = TAIL_ML
        else:
            self.nozzle = self.tail * (1 - exp(-dt / TAIL_TAU))
            if self.tail < .015:
                self.nozzle = self.tail
            self.tail -= self.nozzle
        self.emitted += self.nozzle

        self.obstructions = obstructions
        self.landed = self.blocked_now = 0.
        gap = OUTLET_ABOVE_FLOOR - self.top / AREA
        in_flight = deque()
        for birth, lane_ml, mask in self.air:
            age = self.time - birth
            start, end = self.falling_depth(max(0., age-dt)), self.falling_depth(age)
            # Interception precedes cup entry only where the swept jet reaches a collider.
            if start < gap:
                for lane, near, far, _ in obstructions:
                    bit = 1 << lane
                    if mask & bit and start <= far and near <= min(end, gap):
                        mask ^= bit
                        self.blocked_now += lane_ml
            if end >= gap:
                self.landed += lane_ml * mask.bit_count()
            elif mask:
                in_flight.append((birth, lane_ml, mask))
        self.air = in_flight
        if self.nozzle:
            lanes = len(JET_OFFSETS)
            self.air.append((self.time, self.nozzle/lanes, (1 << lanes)-1))
        self.pending += self.nozzle - self.landed - self.blocked_now
        self.intercepted += self.blocked_now

        self.liquid += self.landed
        self.bubbles += self.landed * self.condition.entrained_gas_per_ml
        released = self.bubbles * (1 - exp(-dt / BUBBLE_RISE_TAU))
        self.bubbles -= released
        drained = self.gas * (1 - exp(-dt / self.condition.drainage_tau_s))
        self.gas = min(self.liquid / FOAM_LIQUID_PER_GAS, self.gas - drained + released)

        self.crest = self.top
        excess = max(0., self.crest - CAPACITY)
        spilled_gas = min(self.gas, excess / (1 + FOAM_LIQUID_PER_GAS))
        self.gas -= spilled_gas
        self.spill_foam += spilled_gas * (1 + FOAM_LIQUID_PER_GAS)
        self.spill_liquid += excess - spilled_gas
        self.liquid -= excess - spilled_gas
        quiet = not opened and not self.tail and not self.air and released <= drained
        self.settled = self.settled + dt if quiet else 0.

    @property
    def score(self):
        if (self.openings > 3 or self.spill_liquid > LOSS_EPS_ML
                or self.intercepted > LOSS_EPS_ML or self.settled < .25):
            return 0.
        return min(1., self.liquid / TARGET)

    def measure(self):
        return dict(
            time_s=self.time, valve_travel_mm=1000 * self.travel, valve_open=self.open,
            openings=self.openings, nozzle_ml=self.nozzle, landed_ml=self.landed,
            flow_ml_s=self.condition.flow_ml_s, intercepted_ml=self.intercepted,
            intercepted_this_tick_ml=self.blocked_now, jet_obstructions=self.obstructions,
            emitted_ml=self.emitted, in_flight_ml=max(0., self.pending),
            liquid_ml=self.liquid, liquid_g=self.liquid * DENSITY,
            bulk_height_mm=1000 * self.bulk / AREA, foam_ml=self.top - self.bulk,
            foam_front_mm=1000 * self.top / AREA, crest_before_spill_mm=1000 * self.crest / AREA,
            spill_liquid_ml=self.spill_liquid, spill_foam_ml=self.spill_foam,
            settled_s=self.settled, score=self.score,
        )


def next_press(pour):
    """Expert hysteresis on headroom, settled foam and retained amount."""
    if pour.bulk >= TARGET - 2.0:
        return False
    if pour.open:
        return pour.top < CAPACITY - 26.
    return (not pour.openings or
            (pour.openings < 3 and not pour.tail and not pour.air and
             pour.top - pour.bulk < 19. and pour.top < CAPACITY - 38.))

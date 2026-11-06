"""Ideal series circuit and a voltage-only, history-aware task score."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
HZ = CONFIG["simulation"]["control_hz"]
PHYSICS_HZ = CONFIG["simulation"]["physics_hz"]
MAX_STEPS = CONFIG["episode"]["max_control_steps"]
TRAVEL = .26
SUPPLY_V = 15.
LAMP_OHM = 10.
RHEOSTAT_OHM = 280.
INSTRUCTION = (
    "Light the 5 V bulb safely using the sliding rheostat. Inspect the wiring and "
    "digital voltmeter to choose the sliding direction. Use the left hand to steady "
    "the rheostat base and the right hand to move its slider. Bring the bulb voltage "
    "to 5.0 V (4.8 to 5.2 V) and hold it there for at least one second. Do not exceed "
    "7 V at any time. Keep the base flat on the table within 3 cm of its starting "
    "position. Complete within 120 seconds."
)


def voltage(displacement):
    """Slider displacement in metres; negative is rightward, toward the wired end."""
    if not math.isfinite(displacement):
        raise ValueError("Non-finite slider position")
    fraction = min(1., max(0., .5 + displacement / TRAVEL))
    return SUPPLY_V * LAMP_OHM / (LAMP_OHM + RHEOSTAT_OHM * fraction)


def brightness(volts):
    if not math.isfinite(volts):
        raise ValueError("Non-finite voltage")
    return min(1., max(0., (volts - 3.) / 4.))


def intact(root, initial_root):
    """The wired base must remain on the table, upright and within lead slack."""
    if not all(math.isfinite(v) for v in root):
        raise ValueError("Non-finite base pose")
    x, y, z, qx, qy, qz, qw = root
    return (math.hypot(x-initial_root[0],y-initial_root[1]) <= .03
            and abs(z-initial_root[2]) <= .01
            and 1-2*(qx*qx+qy*qy) >= math.cos(math.radians(15)))


class Score:
    def __init__(self):
        self.ticks = self.held = 0
        self.peak = 0.
        self.volts = 1.

    def update(self, displacement, circuit_intact=True):
        self.ticks += 1
        self.volts = voltage(displacement)
        self.peak = max(self.peak, self.volts)
        self.held = self.held + 1 if circuit_intact and 4.8 <= self.volts <= 5.2 else 0
        return self.result()

    def result(self):
        success = (self.held >= PHYSICS_HZ and self.peak <= 7.
                   and self.ticks <= MAX_STEPS * PHYSICS_HZ // HZ)
        return dict(score=float(success), success=success, voltage_V=self.volts,
                    peak_voltage_V=self.peak, held_s=self.held / PHYSICS_HZ,
                    time_s=self.ticks / PHYSICS_HZ)

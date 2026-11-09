"""Task geometry, execution settings, and instruction in SI units."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'dream-defaults.json').read_text())
HZ = CONFIG['simulation']['control_hz']
PHYSICS_HZ = CONFIG['simulation']['physics_hz']
DT = 1 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // HZ
MAX_STEPS = CONFIG['episode']['max_control_steps']
PAN_XY = (.40, .055)
HEATER_HEIGHT = .028
PAN_FLOOR = .009
PAN_RADIUS = .135
STEAK_HEIGHT = .018
TOOL_XY = (.245, -.215)
ROOT = '/World/envs/env_0'
INSTRUCTION = ('Use the spatula to flip the steak in the pan. Keep its entire initial bottom face '
    'in uninterrupted contact with the pan for at least 50 seconds, but do not let any part '
    'of that face contact the pan for more than 70 seconds. Leave the flipped steak resting '
    'in the pan, cooked side up. The timer counts up from zero.')


def outline():
    a = np.linspace(0, 2*np.pi, 48, endpoint=False)
    radius = 1 + .055*np.cos(3*a) + .035*np.sin(2*a)
    return np.c_[.070*radius*np.cos(a), .049*radius*np.sin(a)]

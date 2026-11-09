"""Shared geometry, mass, timing and instruction (SI units)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'dream-defaults.json').read_text())
PHYSICS_HZ = CONFIG['simulation']['physics_hz']
HZ = CONFIG['simulation']['control_hz']
DT = 1 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // HZ
MAX_STEPS = CONFIG['episode']['max_control_steps']
BOTTLE_XY = (.33, -.19)
PLATE_XY = (.34, .055)
GRAIN_RADIUS = .00125  # Circumscribed radius of each convex grain.
GRAIN_MASS_G = .0125
GRAIN_COUNT = 800
BOTTLE_HEIGHT = .11
HINGE_Z = .112
STEAK_AXES = (.087, .060)
STEAK_HEIGHT = .019
INSTRUCTION = ('Season the steak with the pepper salt. Inspect the left flip-top opening first; '
    'if it is the wide pouring opening, close it, put the bottle down, turn it 180 degrees, '
    'and open the other, perforated side. Pick up the bottle and sprinkle twice over the steak. '
    'Leave seasoning on the steak without exceeding 2 grams.')


def grain_positions():
    """A separated deterministic packing inside the closed bottle."""
    rng = np.random.default_rng(CONFIG['episode']['seed'])
    points = np.array([(x, y, z) for z in np.arange(.007, .08, .0028)
                       for y in np.arange(-.024, .025, .0028)
                       for x in np.arange(-.016, .017, .0028)])[:GRAIN_COUNT]
    assert len(points) == GRAIN_COUNT
    return points + rng.uniform(-.00005, .00005, points.shape)

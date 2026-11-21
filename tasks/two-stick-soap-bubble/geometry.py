"""Wand, rack and dipping basin geometry in metres."""
from pathlib import Path

import numpy as np
from yam_arena.station import TABLE_TOP

HERE = Path(__file__).resolve().parent
GRIP_X, GRIP_Z, HALF_SPAN = .205, TABLE_TOP + .220, .103
TIP = np.array([.185, 0., 0.])
REST_PITCH = 0.
CORD_RADIUS = .002
STRANDS = ((.246, 14), (.396, 22))
TRAY_X, TRAY_Y = (.43, .674), (-.165, .165)
FLOOR, SURFACE, RIM = TABLE_TOP + .009, TABLE_TOP + .103, TABLE_TOP + .129
ROOT = "/World/envs/env_0/bubble_kit"
AIR = np.array([1.48, 0., 0.])

WANDS = ["wand_left", "wand_right"]
LINKS = [[f"cord_{i}_{j:02d}" for j in range(n)] for i, (_, n) in enumerate(STRANDS)]
BODIES = WANDS + LINKS[0] + LINKS[1]
CRADLE = tuple((x, y, z) for x in (.150, .288)
               for y in (HALF_SPAN, -HALF_SPAN)
               for z in (GRIP_Z,))

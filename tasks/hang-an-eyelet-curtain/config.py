"""Shared task geometry and execution settings (SI units, metres)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / 'dream-defaults.json').read_text())
PHYSICS_HZ = CONFIG['simulation']['physics_hz']
HZ = CONFIG['simulation']['control_hz']
DT = 1 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // HZ
MAX_STEPS = CONFIG['episode']['max_control_steps']

# Stand and rod. The rod lies along world y; both U cradles hold it at ROD_HEIGHT above the table.
STAND_XY = (.36, .2595)          # right tip stays at y=-.131; left tip reaches table edge y=.650
SUPPORT_HALF_SPAN = .3395        # right cradle stays at y=-.080; left cradle moves to y=.599
ROD_HEIGHT = .105                # rod axis above the table top
ROD_R = .006
CRADLE_T = .010                  # cradle plate thickness along the rod
CRADLE_SLOT = .015               # slot width across the rod
PRONG_T = .008                   # prong thickness across the rod
PRONG_TOP = .016                 # prong tops above the rod axis
KNOB_R = .010
KNOB_T = .008
KNOB_GAP = .003                  # knob inner face to cradle outer face
TIP = .035                       # rod length beyond each knob
ROD_HALF = SUPPORT_HALF_SPAN + CRADLE_T/2 + KNOB_GAP + KNOB_T + TIP

# Curtain. Built flat in its own frame: top edge along +y, height along -x, face normal +z.
RINGS = 2
RING_RI, RING_RO, RING_T = .015, .018, .003
RING_MASS = .001
RING_PITCH, RING_DROP, HOLE_R = .102, .020, .0165
FABRIC_W, FABRIC_H, FABRIC_T = .189, .200, .0005
BEND_STIFFNESS, BEND_DAMPING = 200000., .1
CURTAIN_XY = (.270, 0.)         # body extends towards the robot, between the two bases
CURTAIN_YAW = 0.

INSTRUCTION = ('Hang the eyelet curtain on the rod: both metal rings must end up on the rod between its two '
    'supports, with the rod resting in both cradles.')



"""Shared task geometry, timing, and latch parameters (SI units)."""
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
BOOK_XY = (.27, .025)
BLOCK_XY = (.49, .045)
CARD_XY = (.49, .025)
HOLES_X = np.array([-.0605, -.0465, -.0325, .0325, .0465, .0605])
HOLE_Y = .064
HOLE_R = .0045
CARD_T = .0016
WIRE_R = .0012
OPEN_ANGLE = np.deg2rad(35)
PIVOT_Y = .010
PIVOT_Z = .011
BUTTON_FORCE = 3.
CLOSE_FORCE = 2.
INSTRUCTION = ('Insert the red six-hole tabbed divider into the open notebook. Hold the notebook down '
    'with the open right gripper across its spine and press the near-end lever with the closed left gripper '
    'to open all six rings. Pick up the divider from its overhanging right edge on the wooden block. '
    'Tilt the card and gripper down-left to match the right ring ends, thread all six holes, release, and withdraw right/down. '
    'Squeeze the two sides of the far three rings together to close the binder.')


def ring_points(side, count=33):
    """One half-ring centerline in its hinge frame, from rail to mating tip."""
    p = np.array([[0,side*.010,.011],[0,side*.038,.012],
                  [0,side*.030,.049],[0,side*.0013,.049]])
    u = np.linspace(0,1,count)[:,None]
    curve = (1-u)**3*p[0]+3*(1-u)**2*u*p[1]+3*(1-u)*u**2*p[2]+u**3*p[3]
    return curve - [0,side*PIVOT_Y,PIVOT_Z]

"""Shared task geometry and execution configuration (metres, seconds)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
PHYSICS_HZ = CONFIG["simulation"]["physics_hz"]
CONTROL_HZ = CONFIG["simulation"]["control_hz"]
MAX_STEPS = CONFIG["episode"]["max_control_steps"]
assert PHYSICS_HZ % CONTROL_HZ == 0
DT = 1 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
HEIGHT = .7112 / 16
BASE_WIDTH = .358775 / 16
BASE_THICKNESS = .002
TIP_HEIGHT = .005
BODY_RADIUS = .254 / 32
TIP_RADIUS = .003
WALL = .0008
STACK_PITCH = .016
STACK_XY = (.29, -.37)
BOARD_CENTER = (.36, .035)
BOARD_SIZE = (.64, .70, .002)
ROAD_WIDTH = .24
RADIUS = .25
ARC_ANGLE = np.pi / 3
LENGTH = 2 * RADIUS * ARC_ANGLE
ROAD_START = np.array([.23, .23])
HEADING = np.arctan2(-.9, .44) - ARC_ANGLE / 2
INSTRUCTION = (
    "Take the six stacked traffic cones from the right of the white board, gripping "
    "only the top 5 mm of each cone. Place them upright along the road's dashed centre "
    "line, one at each end and four evenly between. Each base must rest directly on "
    "the board. Release all cones and move the grippers clear. Finish within 120 seconds."
)


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def curve(s):
    """Two tangent circular arcs parameterised by exact arc length."""
    s = np.asarray(s, dtype=float)
    a = np.minimum(s / RADIUS, ARC_ANGLE)
    first = np.stack((RADIUS * np.sin(a), RADIUS * (1 - np.cos(a))), axis=-1)
    b = np.maximum(s / RADIUS - ARC_ANGLE, 0)
    second = np.stack((RADIUS * np.sin(b), -RADIUS * (1 - np.cos(b))), axis=-1)
    return ROAD_START + (first + second @ rotation(ARC_ANGLE).T) @ rotation(HEADING).T


def tangent(s):
    angle = HEADING + ARC_ANGLE - np.abs(np.asarray(s) / RADIUS - ARC_ANGLE)
    return np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def nearest(points):
    """Exact closest points on the two finite circular arcs; return distance and arc length."""
    points = np.asarray(points, dtype=float)
    local = (points - ROAD_START) @ rotation(HEADING)
    a = np.clip(np.arctan2(local[..., 0], RADIUS - local[..., 1]), 0, ARC_ANGLE)
    join = np.array([RADIUS * np.sin(ARC_ANGLE), RADIUS * (1 - np.cos(ARC_ANGLE))])
    other = (local - join) @ rotation(ARC_ANGLE)
    b = np.clip(np.arctan2(other[..., 0], RADIUS + other[..., 1]), 0, ARC_ANGLE)
    candidates = np.stack((a * RADIUS, (ARC_ANGLE + b) * RADIUS,
                           np.zeros_like(a), np.full_like(a, LENGTH / 2),
                           np.full_like(a, LENGTH)), axis=-1)
    distance = np.linalg.norm(curve(candidates) - points[..., None, :], axis=-1)
    index = np.argmin(distance, axis=-1)
    return np.take_along_axis(distance, index[..., None], -1)[..., 0], np.take_along_axis(candidates, index[..., None], -1)[..., 0]

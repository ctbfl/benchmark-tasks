"""Dimensions and involute flank geometry. All lengths are in metres."""

import math

import numpy as np

MODULE = .0022
RING_TEETH, WHEEL_TEETH = 54, 36
RING = MODULE * RING_TEETH / 2
WHEEL = MODULE * WHEEL_TEETH / 2
ORBIT = RING - WHEEL
OUTSIDE = .071
THICKNESS = .0048
HOLE = .0275
BORE = .00335
SHANK = .00240
TIP = .00085
HEIGHT = .090
GRASP = .063
PAPER = .0105
BOARD = (.254, .230)
CENTER = (.350, -.100)
PARK = (-.066, -.088)
RATIO = ORBIT / WHEEL
PRESSURE_ANGLE = math.radians(25)
ADDENDUM = .77 * MODULE
DEDENDUM = 1.04 * MODULE
FLANK_CLEARANCE = .000105


def polar(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def flank(radius, teeth, internal):
    pitch = MODULE * teeth / 2
    base = pitch * math.cos(PRESSURE_ANGLE)
    alpha = math.acos(min(1., base / radius))
    phase = math.tan(PRESSURE_ANGLE) - PRESSURE_ANGLE - (math.tan(alpha) - alpha)
    return math.pi / (2 * teeth) + (-phase if internal else phase) - FLANK_CLEARANCE / pitch


def tooth(angle, teeth, internal):
    pitch = MODULE * teeth / 2
    root = pitch + DEDENDUM if internal else pitch - DEDENDUM
    crest = pitch - ADDENDUM if internal else pitch + ADDENDUM
    # Slightly overlap the tooth with the backing at its root.
    radii = np.linspace(root + (.00016 if internal else -.00016), crest, 6)
    halves = [flank(radius, teeth, internal) for radius in radii]
    low = [radius * polar(angle - half) for radius, half in zip(radii, halves)]
    high = [radius * polar(angle + half) for radius, half in zip(radii, halves)]
    return low + high[::-1]

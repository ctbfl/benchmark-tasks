"""Measure one continuous, tooth-registered ink stroke from physical samples."""

import math

import numpy as np

from geometry import BORE, CENTER, HOLE, ORBIT, PAPER, RATIO, THICKNESS, TIP, WHEEL_TEETH
from scene import DT

TAU = 2 * math.pi
INK_FORCE = .030
INK_WIDTH = .0008
NIB_LIFT = .00005
TOUCH_FORCE = .012
INK_GAP = .12
MESH_BAND = .00105
MIN_TRAVEL = 2*TAU - .10
PHASE_LIMIT = .48 * TAU / WHEEL_TEETH
CLOSURE = .0020
STRAY_LIMIT = .008
BOARD_TOP = .7475 + .0002 + PAPER


def wrapped(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def local_axes(pose):
    x, y, z, w = pose[3:]
    return (
        np.array([1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)]),
        np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]),
    )


class RosetteScore:
    def __init__(self):
        self.tick = 0
        self.last_mesh = self.last_bore = -math.inf
        self.last_eligible = -math.inf
        self.start = self.previous = self.last_ink = None
        self.unmarked = False
        self.ink_segments = 0
        self.travel = self.spin = self.best = self.stray = 0.
        self.good_ticks = self.ink_ticks = self.mesh_ticks = self.bore_ticks = 0
        self.generation = 0
        self.closure = self.phase_error = None
        self.closed = False
        self.stroke = []

    def update(self, wheel, pen, forces):
        self.tick += 1
        vx, vz = local_axes(wheel)
        _, pz = local_axes(pen)
        mid = wheel[:3] + HOLE*vx + THICKNESS/2*vz
        # The lowest point of the spherical nib is directly below its rotated center.
        tip = pen[:3] + TIP*(pz - (0., 0., 1.))
        delta = wheel[:2] - CENTER
        radius = math.hypot(*delta)
        orbit = math.atan2(delta[1], delta[0])
        spin = math.atan2(vx[1], vx[0])
        in_mesh = abs(radius-ORBIT) <= MESH_BAND and abs(wheel[2]-BOARD_TOP) < .0010 and vz[2] > .998
        shank = pen[:2] + pz[:2]*(mid[2]-pen[2])/max(.2, pz[2])
        in_bore = pz[2] > .974 and np.linalg.norm(shank-mid[:2]) < BORE-.0019
        on_page = (abs(tip[0]-CENTER[0]-.012) <= .099 and abs(tip[1]-CENTER[1]-.018) <= .086)
        at_page = on_page and -.00075 <= tip[2]-BOARD_TOP <= NIB_LIFT
        ink = at_page and forces[0] >= INK_FORCE
        xy = tip[:2]
        # Only connect marks that overlap across missing pressure samples. Test each
        # intervening position so a pen that leaves the footprint and returns still breaks.
        if not at_page or (self.last_ink is not None and (self.unmarked or not ink)
                           and np.linalg.norm(xy-self.last_ink) > INK_WIDTH):
            self.last_ink = self.start = None
        self.unmarked = not ink
        if in_mesh and forces[1] >= TOUCH_FORCE:
            self.last_mesh = self.tick
            self.mesh_ticks += 1
        if in_bore and forces[2] >= TOUCH_FORCE:
            self.last_bore = self.tick
            self.bore_ticks += 1
        touched = (self.tick-min(self.last_mesh, self.last_bore))*DT <= INK_GAP
        eligible = ink and in_mesh and in_bore and touched
        if ink:
            self.ink_ticks += 1
            ink_continuous = self.last_ink is not None
            self.ink_segments += not ink_continuous
            self.stroke.append((self.tick, *xy, int(ink_continuous)))
            if not eligible and ink_continuous:
                self.stray += np.linalg.norm(xy-self.last_ink)
            self.last_ink = xy
        if not eligible:
            if not in_mesh or not in_bore or (self.tick-self.last_eligible)*DT > INK_GAP:
                self.start = None
            return
        self.good_ticks += 1
        if self.start is not None:
            d_orbit = wrapped(orbit-self.previous[0])
            d_spin = wrapped(spin-self.previous[1])
            self.travel += d_orbit
            self.spin += d_spin
            self.phase_error = self.spin + RATIO*self.travel
            if abs(self.phase_error) >= PHASE_LIMIT or abs(d_orbit) > .12 or np.linalg.norm(xy-self.previous[2]) > .0025:
                self.start = None
        if self.start is None:
            self.generation += 1
            self.travel = self.spin = 0.
            self.phase_error = 0.
            self.start = xy
        self.previous = orbit, spin, xy
        self.last_eligible = self.tick
        self.best = max(self.best, abs(self.travel))
        self.closure = float(np.linalg.norm(xy-self.start))
        if abs(self.travel) >= MIN_TRAVEL and self.closure <= CLOSURE:
            self.closed = True

    def result(self):
        fraction = min(abs(self.best)/(2*TAU), 1.)
        score = 1. if self.closed else min(.95, fraction)
        if self.stray > STRAY_LIMIT:
            score = 0.
        return dict(score=float(score), orbit_rad=float(self.best), current_orbit_rad=float(self.travel),
            current_spin_rad=float(self.spin), phase_error_rad=self.phase_error, closure_m=self.closure,
            stray_ink_m=float(self.stray), ink_samples=self.ink_ticks, ink_segments=self.ink_segments, valid_samples=self.good_ticks,
            ring_contact_samples=self.mesh_ticks, bore_contact_samples=self.bore_ticks, segments=self.generation)


def score_trace(trace):
    scorer = RosetteScore()
    for row in trace[1:]:
        scorer.update(row[16:23], row[23:30], row[30:33])
    return scorer


if __name__ == "__main__":
    import json
    import sys

    raw = np.fromfile(sys.argv[1], dtype="<f4").reshape(-1, 33)
    print(json.dumps(score_trace(raw).result(), indent=2))

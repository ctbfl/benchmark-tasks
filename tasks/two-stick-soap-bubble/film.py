"""Reduced soap-film and air model driven by the live, jointed cord boundary."""
from dataclasses import dataclass

import numpy as np

from geometry import AIR, CORD_RADIUS, FLOOR, RIM, STRANDS, SURFACE, TABLE_TOP, TRAY_X, TRAY_Y

SIGMA = .027
RHO = 1.20


def sphere_radius(volume):
    return np.cbrt(3 * volume / (4 * np.pi))


def film_aperture(boundary):
    center = boundary.mean(axis=0)
    local = boundary - center
    cross = .5 * np.cross(local, np.roll(local, -1, axis=0)).sum(axis=0)
    area = np.linalg.norm(cross)
    normal = cross / max(area, 1e-9)
    return center, area, normal if normal[0] >= 0 else -normal


def in_solution(points):
    return ((points[:, 0] > TRAY_X[0]) & (points[:, 0] < TRAY_X[1])
            & (points[:, 1] > TRAY_Y[0]) & (points[:, 1] < TRAY_Y[1])
            & (points[:, 2] >= FLOOR) & (points[:, 2] < SURFACE))


def obstacle_clearance(center, radius):
    # Sphere versus the table plane and the outside of the basin below its rim.
    nearest_tray = np.clip(center, [TRAY_X[0]-.008, TRAY_Y[0]-.008, TABLE_TOP],
                          [TRAY_X[1]+.008, TRAY_Y[1]+.008, RIM])
    return min(center[2]-TABLE_TOP, np.linalg.norm(center-nearest_tray)) - radius


@dataclass
class Bubble:
    center: np.ndarray
    velocity: np.ndarray
    radius: float
    age: float = 0.


class Film:
    def __init__(self, dt):
        self.dt = dt
        self.soak = np.zeros(sum(n for _, n in STRANDS))
        self.wet_until = 0.
        self.time = -dt
        self.previous_center = None
        self.speed = np.zeros(3)
        self.volume = 0.
        self.bubble = None
        self.boundary = None
        self.peak_neck = 0.
        self.score = 0.
        self.events = []

    def event(self, what, **metrics):
        self.events.append({"time_s": self.time, "event": what, **metrics})

    def clear_film(self):
        self.volume = self.peak_neck = self.wet_until = 0.
        self.soak.fill(0.)

    def update(self, upper, lower, cord_centers, wand_centers):
        dt = self.dt
        self.time += dt
        self.boundary = np.concatenate((upper, lower[-2:0:-1]))
        center, area, normal = film_aperture(self.boundary)
        if self.previous_center is None:
            self.previous_center = center
        self.speed += min(1., dt / .038) * ((center-self.previous_center)/dt-self.speed)
        self.previous_center = center
        tips = (upper[[0, -1]] + lower[[0, -1]]) / 2
        span = np.linalg.norm(tips[0]-tips[1])
        disk_r = np.sqrt(area/np.pi)
        u = max(0., np.dot(AIR-self.speed, normal))
        self.sample = {"t": self.time, "area_m2": area, "span_m": span, "air_m_s": u,
                       "mouth_xyz": center.tolist(), "tips_xyz": tips.tolist()}

        if self.bubble is not None:
            b = self.bubble
            b.age += dt
            b.velocity += dt * (.9*(.64*AIR-b.velocity) + np.array([0., 0., -.025]))
            b.center += b.velocity * dt
            cord_gap = np.linalg.norm(self.boundary-b.center, axis=1).min()-CORD_RADIUS
            wand_gap = np.linalg.norm(wand_centers-b.center, axis=1).min()-.012
            gap = min(cord_gap-b.radius, wand_gap-b.radius, obstacle_clearance(b.center, b.radius))
            if b.age > 4.5 or gap < -.012:
                self.event("detached_film_burst", age_s=b.age)
                self.bubble = None
            elif self.score == .5:
                self.score = 1.
                self.event("bubble_floating")

        dipped = in_solution(cord_centers)
        self.soak[dipped] = np.minimum(.10, self.soak[dipped] + dt)
        entirely_dipped = np.all(in_solution(tips)) and np.min(self.soak) >= .09
        if entirely_dipped:
            if self.wet_until <= self.time:
                self.event("loop_wetted")
            self.wet_until = self.time+12.
            self.volume = self.peak_neck = 0.
        wet = self.wet_until > self.time
        clear = self.boundary[:, 2].min() > SURFACE + .007
        if not wet or not clear:
            self.volume = 0.
            return
        if u > 3.35 or np.linalg.norm(self.speed) > .90:
            self.event("film_ruptured", air_m_s=u)
            self.clear_film()
            return

        radius = sphere_radius(self.volume)
        curvature_radius = max(disk_r, radius)
        pressure = .5 * RHO * u**2
        capillary = 4 * SIGMA / max(curvature_radius, .008)
        influx = .40 * area * np.sqrt(max(0., 2*(pressure-capillary)/RHO))
        self.volume = np.clip(self.volume+dt*(influx-.18*self.volume), 0., .019)
        radius = float(sphere_radius(self.volume))
        lobe_center = center + normal * (radius + .028)
        if radius > .028 and obstacle_clearance(lobe_center, radius) < 0.:
            self.event("attached_film_hit_surface", diameter_m=2*radius)
            self.clear_film()
            return
        if radius > .025:
            self.peak_neck = max(self.peak_neck, area)
        if area > 0. and self.score == 0.:
            self.score = .5
            self.event("film_formed")
        closing = self.peak_neck > .012 and area < .33*self.peak_neck and span < .064
        if closing and self.bubble is None and radius > .03 and u > .36:
            self.bubble = Bubble(lobe_center, normal*(.35*u), radius)
            self.event("bubble_released")
            self.clear_film()

    def result(self):
        return {"score": self.score, "film_formed": self.score >= .5,
                "bubble_floating": self.score == 1., "events": self.events}

    def visual(self):
        center, area, normal = film_aperture(self.boundary)
        b = self.bubble
        wet = self.wet_until > self.time and self.boundary[:, 2].min() > SURFACE + .007
        return [*center, *normal, self.volume, area, float(wet),
                *(np.r_[b.center, b.radius] if b is not None else np.zeros(4))]

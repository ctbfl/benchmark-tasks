"""Conservative reduced-order film, pendant and falling-drop model."""
from dataclasses import asdict, dataclass

import numpy as np

from layout import GROOVES, GROOVE_RADIUS, GROOVE_WIDTH, JAR, JAR_INNER, JAR_SURFACE
from layout import PLATE_R, PLATE_TOP, TABLE_TOP, TOAST, TOAST_OUTLINE, TOAST_TOP

DENSITY = 1420.0
VISCOSITY = 2.2
SURFACE_TENSION = .071
GRAVITY = 9.81
CELLS = 32
ANGLE = np.arange(CELLS) * (2*np.pi/CELLS)
EDGE = ANGLE + np.pi/CELLS
DX = 2*np.pi*GROOVE_RADIUS/CELLS
CELL_MASS = 1000*DENSITY*GROOVE_WIDTH*DX
LOAD_G = 4.0
NECK_RADIUS = .0011
BREAK_G = 1000*2*np.pi*NECK_RADIUS*SURFACE_TENSION/GRAVITY
BOTTOM = np.cos(ANGLE) > .65
BREAD_POLY = np.array(TOAST_OUTLINE) * .95
SURFACES = (
    ("jar", JAR_SURFACE, lambda xy: np.linalg.norm(xy-JAR) <= JAR_INNER),
    ("toast", TOAST_TOP, lambda xy: inside_toast(xy)),
    ("plate", PLATE_TOP, lambda xy: np.linalg.norm(xy-TOAST) <= PLATE_R),
    ("table", TABLE_TOP, lambda xy: True),
)


def inside_toast(xy):
    x, y = xy - TOAST
    p = BREAD_POLY
    q = np.roll(p, 1, axis=0)
    crossings = sum((ay > y) != (by > y) and x < ax + (bx-ax)*(y-ay)/(by-ay)
                    for (ax, ay), (bx, by) in zip(p, q))
    return bool(crossings % 2)


@dataclass
class Landing:
    time_s: float
    p: list
    mass_g: float
    surface: str
    release_time_s: float
    release_roll_rad_s: float


class Honey:
    def __init__(self):
        self.film = np.full((len(GROOVES), CELLS), LOAD_G / (len(GROOVES)*CELLS*CELL_MASS))
        self.pendant = np.zeros(len(GROOVES))
        self.flights = []
        self.landings = []
        self.t = 0.0
        self.wet_added = 0.0
        self.rot = 0.
        self.motion = np.zeros(6)
        self.axis = np.array([1., 0., 0.])
        self.down = np.array([0., 0., -1.])
        self.bases = np.zeros((len(GROOVES), 3))
        self.release_count = 0

    @property
    def retained_g(self):
        return float(self.film.sum()*CELL_MASS + self.pendant.sum())

    def pose(self, root, axis):
        self.axis = axis
        g_perp = np.array([0., 0., -1.]) + axis[2]*axis
        horizontal = np.linalg.norm(g_perp)
        self.down = g_perp / max(horizontal, 1e-8) if horizontal > 1e-8 else np.array([1., 0., 0.])
        grooves = root + np.asarray(GROOVES)[:, None]*axis
        self.bases = grooves + self.down*GROOVE_RADIUS
        return horizontal, grooves

    def update(self, dt, root, axis, linear, angular):
        self.t += dt
        # The viscous film responds to tool motion over a finite time. A
        # signed low-pass preserves reversals while rejecting contact chatter.
        self.motion += -np.expm1(-dt/.060)*(np.r_[linear, angular]-self.motion)
        linear, angular = self.motion[:3], self.motion[3:]
        self.rot = float(angular @ axis)
        horizontal, grooves = self.pose(root, axis)

        # Newtonian gravity flux (h^3 law), no-slip roll advection, and a
        # diffusive approximation to circumferential capillary smoothing.
        wall = GROOVE_RADIUS*self.rot
        gravity = -DENSITY*GRAVITY*horizontal*np.sin(EDGE)/(3*VISCOSITY)
        capillary_diffusion = 1.5e-5
        speed_bound = abs(wall) + 3*np.max(np.abs(gravity))*self.film.max()**2 + 2*capillary_diffusion/DX
        n = max(1, int(np.ceil(dt*speed_bound/(.40*DX))))
        for _ in range(n):
            downstream = np.roll(self.film, -1, axis=1)
            flux = wall*(self.film if wall >= 0 else downstream)
            flux += gravity*np.where(gravity >= 0, self.film, downstream)**3
            flux -= capillary_diffusion*(downstream-self.film)/DX
            self.film += dt/(n*DX)*(np.roll(flux, 1, axis=1)-flux)
        assert self.film.min() >= -1e-10, "Film integrator exceeded its stable step"

        # The underside has a finite capillary support depth. Excess supplies
        # a viscous pendant, while roll can advect a thin neck back onto the rib.
        capillary_depth = .00185
        drain = np.maximum(self.film[:, BOTTOM] - capillary_depth, 0) * min(2.8*dt, 1)
        self.film[:, BOTTOM] -= drain
        self.pendant += drain.sum(axis=1)*CELL_MASS
        take_up = self.pendant * min(abs(wall)/(GROOVE_RADIUS*.95)*dt, 1)
        self.pendant -= take_up
        self.film[:, 6 if wall >= 0 else -6] += take_up/CELL_MASS

        # Recharge only while the actual annular groove intersects the bath.
        surface_margin = JAR_SURFACE - self.bases[:, 2]
        immersed = (surface_margin > 0) & (np.linalg.norm(grooves[:, :2]-JAR, axis=1) < JAR_INNER-.007)
        room = np.maximum(LOAD_G/len(GROOVES) - self.film.sum(axis=1)*CELL_MASS - self.pendant, 0)
        recharge = np.where(immersed, room*min(10*dt, 1), 0)
        self.film += recharge[:, None] / (CELLS*CELL_MASS)
        self.wet_added += recharge.sum()

        # Detach on neck force, anywhere in the scene. The receiving geometry
        # is consulted only on an actual impact, never to open the flow.
        for i, mass in enumerate(self.pendant):
            if mass >= BREAK_G:
                position = self.bases[i] + [0., 0., -.010]
                velocity = linear + np.cross(angular, self.bases[i]-root) + [0., 0., -.08]
                self.flights.append((position, velocity, float(mass), self.t, self.rot))
                self.pendant[i] = 0.
                self.release_count += 1
        self.advance_flights(dt)

    def advance_flights(self, dt):
        in_flight = []
        for p, v, mass, started, roll in self.flights:
            v = v + [0., 0., -GRAVITY*dt]
            new = p + v*dt
            for surface, height, covers in SURFACES:
                if new[2] <= height:
                    impact = p + np.clip((height-p[2])/(new[2]-p[2]), 0, 1)*(new-p)
                    if covers(impact[:2]):
                        self.landings.append(Landing(self.t, impact.tolist(), mass, surface, started, roll))
                        break
            else:
                in_flight.append((new, v, mass, started, roll))
        self.flights = in_flight

    def measurements(self):
        landed = [asdict(drop) for drop in self.landings]
        flight = sum(drop[2] for drop in self.flights)
        amounts = {k: sum(drop.mass_g for drop in self.landings if drop.surface == k)
                   for k in ("jar", "toast", "plate", "table")}
        mass_error = LOAD_G + self.wet_added - self.retained_g - flight - sum(amounts.values())
        assert abs(mass_error) < 1e-9, mass_error
        return dict(t_s=self.t, retained_g=self.retained_g, mass_error_g=mass_error,
                    in_flight_g=flight, roll_rad_s=self.rot, landed_g=amounts,
                    releases=self.release_count, landings=landed)

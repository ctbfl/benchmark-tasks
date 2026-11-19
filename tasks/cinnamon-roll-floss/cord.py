"""A tension-only particle cord with contact against the board and the dough."""
import math

from numba import njit
import numpy as np

from geometry import BOARD_X, CORD_LENGTH, CORD_RADIUS, RADIUS, WIDTH

NODES = 53
MASS = .00012
STRETCH_STIFFNESS = 45000.
ITERATIONS = 38
CUT_TOLERANCE = .010
RELEASE_HEIGHT = RADIUS-.003
YIELD_FORCE = .85
TAUT = .25
CONTACT_FLOOR = .025
GROOVE_BIN = .004
GROOVE_BINS = math.ceil((WIDTH-CUT_TOLERANCE)/GROOVE_BIN)
INCISION_DEPTH = .002


@njit(cache=False)
def count_outer_load(load, node, x, force):
    if abs(x) > CUT_TOLERANCE:
        bin_index = min(GROOVE_BINS-1, int((abs(x)-CUT_TOLERANCE)/GROOVE_BIN))
        load[node, int(x > 0.), bin_index] += force


@njit(cache=False)
def material_frame(centers):
    axis = centers[1]-centers[0]
    axis /= np.linalg.norm(axis)
    lateral = np.array([-axis[1], axis[0], 0.])
    lateral /= np.linalg.norm(lateral)
    return np.stack((axis, lateral, np.cross(axis, lateral)))


@njit(cache=False)
def contact(x, y, z, previous_z, half, height, seam):
    """Return penetration and outward normal of the rounded cylinder."""
    rho = math.hypot(y, z)
    radius = RADIUS - .0052 * max(0., -z/max(rho, 1e-8))**6
    if seam and abs(x) < CUT_TOLERANCE:
        band = min(1., (CUT_TOLERANCE-abs(x))/.0015)
        floor = -RADIUS+(RADIUS+height)*band
        # The remaining ligament is a circular cap with a load-bearing underside.
        face = floor-z
        if face >= CORD_RADIUS:
            return 0., 0., 0., 0.
        # A node already supporting the ligament cannot step through its thin underside.
        swept = (previous_z <= floor+.0008 and z >= floor
                 and y*y+floor*floor < (RADIUS-.0005)**2)
        if swept or face > rho-radius:
            return CORD_RADIUS-face, 0., 0., -1.
    d1 = abs(x)-half+.0025
    d2 = rho-radius+.0025
    distance = math.hypot(max(0., d1), max(0., d2))+min(max(d1, d2), 0.)-.0025
    if distance >= CORD_RADIUS:
        return 0., 0., 0., 0.
    ux = 1. if x >= 0. else -1.
    if d1 > 0. and d2 > 0.:
        scale = math.hypot(d1, d2)
        return CORD_RADIUS-distance, ux*d1/scale, (d2/scale)*y/max(rho, 1e-8), (d2/scale)*z/max(rho, 1e-8)
    if d1 > d2:
        return CORD_RADIUS-distance, ux, 0., 0.
    return CORD_RADIUS-distance, 0., y/max(rho, 1e-8), z/max(rho, 1e-8)


@njit(cache=False)
def advance(points, velocity, ends, centers, board, table, height, attached, dt):
    last = points.copy()
    separation = CORD_LENGTH/(NODES-1)
    compliance = MASS/(STRETCH_STIFFNESS*dt*dt)
    for i in range(1, NODES-1):
        velocity[i, 2] -= 9.81*dt
        points[i] += dt*.975*velocity[i]
    points[0] = ends[0]
    points[-1] = ends[1]
    multiplier = np.zeros(NODES-1)
    reactions = np.zeros((2, 3))
    center = (centers[0]+centers[1])*.5
    frame = material_frame(centers)
    normal_load = np.zeros(NODES)
    front_load = np.zeros(NODES)
    x_min = np.zeros(NODES)
    x_max = np.zeros(NODES)
    outer_load = np.zeros((NODES, 2, GROOVE_BINS))
    outer_pressure = np.zeros((2, GROOVE_BINS))
    for iteration in range(ITERATIONS):
        reverse = iteration%2 == 1
        for j in range(NODES-1):
            i = NODES-2-j if reverse else j
            a = points[i+1, 0]-points[i, 0]
            b = points[i+1, 1]-points[i, 1]
            c = points[i+1, 2]-points[i, 2]
            length = math.sqrt(a*a+b*b+c*c)
            movable = 1. if i == 0 or i == NODES-2 else 2.
            delta = max(-multiplier[i], (length-separation-compliance*multiplier[i])/(movable+compliance))
            multiplier[i] += delta
            factor = delta/max(length, 1e-8)
            if i > 0:
                points[i, 0] += factor*a
                points[i, 1] += factor*b
                points[i, 2] += factor*c
            if i+1 < NODES-1:
                points[i+1, 0] -= factor*a
                points[i+1, 1] -= factor*b
                points[i+1, 2] -= factor*c

        for i in range(1, NODES-1):
            x, y, z = points[i]
            floor = board if BOARD_X-.15 < x < BOARD_X+.15 and abs(y) < .213 else table
            if z < floor+CORD_RADIUS:
                points[i, 2] = floor+CORD_RADIUS
                points[i, 0] -= .005*(x-last[i, 0])
                points[i, 1] -= .005*(y-last[i, 1])
            for body in range(1 if attached else 2):
                cc = center if attached else centers[body]
                relative = points[i]-cc
                px, py, pz = np.dot(frame[0], relative), np.dot(frame[1], relative), np.dot(frame[2], relative)
                previous_z = np.dot(frame[2], last[i]-cc)
                penetration, nx, ny, nz = contact(px, py, pz, previous_z,
                                                   WIDTH if attached else WIDTH/2, height, attached)
                if penetration == 0.:
                    continue
                normal = nx*frame[0]+ny*frame[1]+nz*frame[2]
                points[i] += penetration*normal
                f = -penetration*MASS/(dt*dt)
                index = int(px > 0) if attached else body
                reactions[index] += f*normal
                if abs(nx) < .5:
                    axial = px+penetration*nx
                    dough_x = axial if attached else axial+(body-.5)*WIDTH
                    count_outer_load(outer_load, i, dough_x, -f)
                    normal_load[i] -= f
                    if attached:
                        x_min[i] = min(x_min[i], axial)
                        x_max[i] = max(x_max[i], axial)
                        front_load[i] -= f*max(0., -nz)
        if attached and height > -RADIUS:
            # Sample each long link at two interior points as well as at its nodes.
            for i in range(1, NODES-2):
                delta = points[i+1]-points[i]
                if np.dot(delta, delta) <= (1.13*separation)**2:
                    continue
                for t in (1./3., 2./3.):
                    relative = (1.-t)*points[i]+t*points[i+1]-center
                    px = np.dot(frame[0], relative)
                    if abs(px) > CUT_TOLERANCE+CORD_RADIUS:
                        continue
                    py, pz = np.dot(frame[1], relative), np.dot(frame[2], relative)
                    previous_z = np.dot(frame[2], (1.-t)*last[i]+t*last[i+1]-center)
                    penetration, nx, ny, nz = contact(px, py, pz, previous_z, WIDTH, height, True)
                    if penetration <= 0. or abs(nx) >= .5:
                        continue
                    normal = nx*frame[0]+ny*frame[1]+nz*frame[2]
                    correction = penetration/((1.-t)**2+t*t)
                    points[i] += (1.-t)*correction*normal
                    points[i+1] += t*correction*normal
                    f = correction*MASS/(dt*dt)
                    reactions[int(px > 0.)] -= f*normal
                    for j, weight in ((i, 1.-t), (i+1, t)):
                        x_min[j] = min(x_min[j], px+penetration*nx)
                        x_max[j] = max(x_max[j], px+penetration*nx)
                        normal_load[j] += weight*f
                        front_load[j] += weight*f*max(0., -nz)
                        count_outer_load(outer_load, j, px+penetration*nx, weight*f)
    velocity[:] = (points-last)/dt

    pull = np.zeros((2, 3))
    for i in range(2):
        node, neighbor = (0, 1) if i == 0 else (NODES-1, NODES-2)
        tangent = points[neighbor]-points[node]
        pull[i] = (tangent/max(1e-8, np.linalg.norm(tangent))) * multiplier[0 if i == 0 else -1]*MASS/(dt*dt)
    local = (points-center) @ frame.T
    residual_center_z = max(0., (height+RADIUS)*.5)
    wrap = 0.
    for i in range(NODES-1):
        ax, ay = local[i, 1], local[i, 2]-residual_center_z
        bx, by = local[i+1, 1], local[i+1, 2]-residual_center_z
        wrap += math.atan2(ax*by-ay*bx, ax*bx+ay*by)
    bounds = np.array([0., 0.])
    shoulders = np.zeros(2)
    cutting_sides = np.zeros(2)
    extent = math.sqrt(max(0., RADIUS**2-max(0., height)**2))
    for i in range(1, NODES-1):
        local_tension = min(multiplier[i-1], multiplier[i])*MASS/(dt*dt)
        if normal_load[i] < CONTACT_FLOOR or local_tension < .10:
            continue
        # Limit each contact's correction load by the tension in its neighboring links.
        outer_pressure += outer_load[i]*min(1., local_tension/normal_load[i])
        if not attached:
            continue
        bounds[0] = min(bounds[0], x_min[i])
        bounds[1] = max(bounds[1], x_max[i])
        y = local[i, 1]
        side = int(y > 0.)
        if abs(y) > .009:
            shoulders[side] += min(normal_load[i], local_tension)
        if abs(y) > .12*extent:
            cutting_sides[side] += min(front_load[i], local_tension)
    radial = normal_load.sum() if attached else 0.
    return pull, reactions, radial, wrap, local, bounds, shoulders, cutting_sides, outer_pressure


def cut_fraction(height):
    level = np.clip(height/RADIUS, -1., 1.)
    return float((math.asin(level)+level*math.sqrt(1.-level**2)+math.pi/2)/math.pi)


class Cord:
    def __init__(self, ends):
        s = np.linspace(0., 1., NODES)
        self.points = (1-s[:, None])*ends[0]+s[:, None]*ends[1]
        self.points[:, 0] -= .055*np.sin(np.pi*s)
        self.velocity = np.zeros_like(self.points)
        self.height = -RADIUS
        self.aligned_hold_s = 0.
        self.aligned_loop = False
        self.max_yield_offset = 0.
        self.off_plane_loaded_s = 0.
        self.groove_depth = np.zeros((2, GROOVE_BINS))
        self.damage = 0.
        self.filtered_load = 0.
        self.wrap_max = 0.
        self.peak_pull = 0.
        self.peak_contact = 0.
        self.last = {}
        self.released = False
        self.underfed = False
        self.crossed = False
        self.peak_cut_force = 0.

    def step(self, ends, centers, board, table, dt, robot_force):
        subdiv = 2
        for _ in range(subdiv):
            pull, reaction, radial, wrap, local, bounds, shoulders, cutting, outer_pressure = advance(
                self.points, self.velocity, ends, centers, board, table, self.height,
                not self.released, dt/subdiv)
            tension = np.linalg.norm(pull, axis=1)
            # A local gouge needs one flank's yield force. Plastic depth survives recentering.
            local_excess = np.maximum(0., outer_pressure-YIELD_FORCE/2)
            self.groove_depth = np.minimum(
                RADIUS, self.groove_depth+np.minimum(.16, .23*local_excess)*(dt/subdiv))
            self.wrap_max = max(self.wrap_max, abs(wrap))
            self.peak_pull = max(self.peak_pull, *tension)
            self.peak_contact = max(self.peak_contact, radial)
            low = local[np.argmin(local[:, 2])]
            below = ((abs(local[:, 1]) < .008) & (abs(local[:, 0]) <= CUT_TOLERANCE)
                     & (local[:, 2] < -.018))
            if np.any(below) and local[0, 1] > .04 and local[-1, 1] < -.04:
                self.underfed = True
            if ends[0, 1] < ends[1, 1] and min(local[[0, -1], 2]) > .026:
                self.crossed = True
            residual_center_z = max(0., (self.height+RADIUS)*.5)
            enclosed = (self.underfed and self.crossed and abs(wrap) >= 5.85
                        and low[2] < residual_center_z-.002
                        and min(local[[0, -1], 2]) > residual_center_z)
            offset = float(max(abs(bounds)))
            on_plane = offset <= CUT_TOLERANCE
            taut = min(tension) >= TAUT
            if enclosed and taut and not on_plane:
                self.off_plane_loaded_s += dt/subdiv
            aligned = enclosed and taut and on_plane
            self.aligned_hold_s = (self.aligned_hold_s+dt/subdiv
                                   if aligned and min(shoulders) > .1 else 0.)
            self.aligned_loop |= self.aligned_hold_s >= .25
            pressure = min(cutting.sum(), 4.*min(tension))
            resistance = YIELD_FORCE * (1.-max(0., self.height)/RADIUS) ** .75
            if aligned and not self.released:
                self.peak_cut_force = max(self.peak_cut_force, pressure)
                bilateral = min(cutting) > .045 or RADIUS-self.height < .006
                if pressure > resistance and bilateral:
                    travel = min(.16, .23*(pressure-resistance))*dt/subdiv
                    self.max_yield_offset = max(self.max_yield_offset, offset)
                    self.height = min(RELEASE_HEIGHT, self.height+travel)
                    self.released = bool(self.height >= RELEASE_HEIGHT)
            # Uneven side loading and direct pinching can flatten the dough.
            asymmetry = abs(reaction.sum(axis=0)[1])
            load = max(asymmetry - (1.7 if enclosed else .5), robot_force-1.8, 0.)
            self.filtered_load += min(1., 28.*dt/subdiv)*(load-self.filtered_load)
            self.damage = min(.38, self.damage + dt/subdiv*.073*max(self.filtered_load-.5, 0.))
        self.last = dict(wrap_rad=float(wrap), end_tension_N=tension.tolist(),
                         radial_load_N=float(radial), low_x_error_m=float(low[0]),
                         low_z_relative_m=float(low[2]), cut_front_height_m=self.height, bearing_sides_N=cutting.tolist(),
                         cut_load_bounds_m=bounds.tolist(), underfed=self.underfed,
                         crossed=self.crossed, damage=self.damage,
                         off_plane_pressure_N=float(np.max(outer_pressure)))
        return pull, reaction

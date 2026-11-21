"""Finite surface films exchanged at measured contacts, in 2 mm cells."""
import numpy as np
from scipy.spatial.transform import Rotation

from geometry import (AXLE_R, AXLE_W, BLOCK, BLOCK_GRID, BLOCK_MASK, CUT, PALETTE,
                      PALETTE_GRID, PANEL_GRID, PANELS, PITCH, THICKNESS)

CYL_SEGMENTS = 96
TREAD_ACROSS = round(AXLE_W / PITCH)
RIDGE_IDS = np.full(BLOCK_MASK.shape, -1)
for index, (x0, x1, y0, y1) in enumerate(CUT):
    x, y = np.moveaxis(BLOCK_GRID * 1000, -1, 0)
    RIDGE_IDS[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)] = index


def surface_points(grid, origin, table_z):
    z = np.full(grid.shape[:2], origin[2] + table_z)
    return np.dstack((grid + origin[:2], z))


class Film:
    def __init__(self, table_z, dt):
        self.dt = dt
        self.palette = np.zeros(PALETTE_GRID.shape[:2])
        self.block = np.zeros(BLOCK_MASK.shape)
        self.roller = np.zeros((CYL_SEGMENTS, TREAD_ACROSS))
        self.paper = np.zeros((PANELS, *PANEL_GRID.shape[:2]))
        self.paper_relief = np.zeros_like(self.paper)
        self.positions = (surface_points(PALETTE_GRID, PALETTE, table_z),
                          surface_points(BLOCK_GRID, BLOCK, table_z))
        x, y = np.moveaxis(self.positions[0], -1, 0)[:2]
        # A charged bead laid across the slab. All other surfaces are initially clean.
        self.palette[(x >= .252) & (x < .276) & (abs(y - PALETTE[1]) < AXLE_W/2)] = 8.
        self.initial_mass = self.mass()
        self.roll_mm = np.zeros(2)
        self.skid_mm = np.zeros(2)
        self.paper_transfer = np.zeros(PANELS)
        self.paper_slide_mm = 0.
        self.previous = None
        self.previous_paper = None

    def mass(self):
        return float(self.palette.sum() + self.block.sum() + self.roller.sum() + self.paper.sum())

    def step(self, drum_pose, roller_force, paper_poses, paper_force):
        previous = self.previous
        self.previous = drum_pose.copy()
        centre = drum_pose[:3]
        rot = Rotation.from_quat(drum_pose[3:])
        matrix = rot.as_matrix()
        axis = matrix[:, 1]
        fore = np.cross(axis, [0., 0., 1.])
        if previous is not None and np.max(roller_force) > .15 and abs(axis[2]) < .25:
            arc = (rot * Rotation.from_quat(previous[3:]).inv()).as_rotvec() @ axis * AXLE_R
            travel = (centre - previous[:3]) @ fore
            slip = abs(travel - arc)
            roll = min(abs(travel), abs(arc))
            efficiency = max(0., 1. - slip / max(abs(travel), abs(arc), .00002))
            loads = (roller_force[0], roller_force[1:].sum())
            for surface_id, (load, film, coords) in enumerate(zip(
                    loads, (self.palette, self.block), self.positions)):
                if load <= .15:
                    continue
                self.roll_mm[surface_id] += 1000 * roll * efficiency
                self.skid_mm[surface_id] += 1000 * slip
                distance = roll * efficiency
                if distance < .000006:
                    continue
                radial = coords - centre
                axial = radial @ axis
                offset = radial @ fore
                reach = ((abs(axial) < AXLE_W/2) & (abs(offset) < .0035)
                         & (abs(radial[..., 2] + AXLE_R) < .010))
                if surface_id == 1:
                    reach &= BLOCK_MASK
                ids = np.where(reach)
                if not ids[0].size:
                    continue
                a, _, c = (radial[ids] @ matrix).T
                phase = np.arctan2(a, -c)
                row = (np.floor((phase + np.pi) * CYL_SEGMENTS / (2*np.pi)).astype(int)
                       % CYL_SEGMENTS)
                col = np.floor((axial[ids]+AXLE_W/2) / PITCH).astype(int)
                strength = min(.45, 1-np.exp(-distance * 520. * load / (load + .3)))
                exchange = strength * (film[ids] - self.roller[row, col])
                if surface_id == 1:
                    # Wet relief retains more of the thin film than the rubber carries back.
                    exchange *= np.where(exchange > 0., .10, 1.)
                # Map each patch cell to its sleeve address and share the receiving cell's capacity.
                # Do not let repeated indices subtract more ink than that cell contains.
                bins = np.bincount(row*TREAD_ACROSS + col, minlength=self.roller.size)
                exchange /= bins[row*TREAD_ACROSS + col]
                film[ids] -= exchange
                np.add.at(self.roller, (row, col), exchange)
                if surface_id == 0:
                    # Compression of a rolling patch levels neighboring tread cells.
                    active = np.flatnonzero(bins)
                    sleeve = self.roller.ravel()
                    for shift in (-TREAD_ACROSS, TREAD_ACROSS):
                        neighbor = (active + shift) % sleeve.size
                        flow = strength * .22 * (sleeve[active] - sleeve[neighbor])
                        np.add.at(sleeve, active, -flow)
                        np.add.at(sleeve, neighbor, flow)

        self.transfer_to_paper(paper_poses, paper_force)

    def transfer_to_paper(self, paper_poses, paper_force):
        for panel, (pose, force) in enumerate(zip(paper_poses, paper_force, strict=True)):
            if force.sum() <= .018:
                continue
            matrix = Rotation.from_quat(pose[3:]).as_matrix()
            if matrix[2, 2] < .35:
                continue
            in_plane_slip = 0.
            if self.previous_paper is not None and force[1:].sum() > .018:
                displacement = pose[:2] - self.previous_paper[panel, :2]
                in_plane_slip = float(np.linalg.norm(displacement))
                self.paper_slide_mm += 1000 * in_plane_slip
            for index, (film, points) in enumerate(zip((self.palette, self.block), self.positions)):
                if index == 0:
                    load = np.full(self.palette.shape, force[0])
                else:
                    load = force[1:][RIDGE_IDS] * BLOCK_MASK
                wet = (film > .0001) & (load > .012)
                ids = np.where(wet)
                p = (points[ids] - pose[:3]) @ matrix
                cell = np.floor((p[:, :2] - PANEL_GRID[0, 0] + PITCH/2)/PITCH).astype(int)
                near = ((abs(p[:, 2] + THICKNESS/2) <= .0016)
                        & np.all(cell >= 0, axis=1)
                        & np.all(cell < np.array(PANEL_GRID.shape[:2]), axis=1))
                to = tuple(cell[near].T)
                source = tuple(idx[near] for idx in ids)
                pressure = load[source]
                rate = 2.9 * self.dt + 470. * in_plane_slip
                deposited = ((1-np.exp(-rate * pressure/(pressure+.16))) * film[source]
                             * np.clip(1.-self.paper[panel][to]/.80, 0., 1.))
                # Pigment reaches paper coordinates that were physically under each loaded ridge.
                # Sliding while still loaded naturally paints an offset impression.
                film[source] -= deposited
                np.add.at(self.paper[panel], to, deposited)
                if index == 1:
                    np.add.at(self.paper_relief[panel], to, deposited)
                self.paper_transfer[panel] += deposited.sum()
        self.previous_paper = paper_poses.copy()

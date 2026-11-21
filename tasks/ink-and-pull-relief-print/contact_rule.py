"""Latch actual print-face contact with either ink-loading surface."""
import numpy as np
from scipy.spatial.transform import Rotation

from geometry import AXLE_R, AXLE_W, PALETTE, PALETTE_SIZE, PANEL_SIZE, THICKNESS

SURFACES = ("slab", "roller")
INK_PRESENT = .0001
CONTACT_MARGIN = .0016
PAPER_RADIUS = np.linalg.norm([PANEL_SIZE[0]/2, (PANEL_SIZE[1]-.0003)/2, THICKNESS/2])
DRUM_RADIUS = np.hypot(AXLE_R, AXLE_W/2)


class PrintContact:
    def __init__(self):
        self.loaded = np.zeros(2, dtype=bool)
        self.first = None

    def update_loaded(self, film):
        self.loaded |= [film.palette.max() > INK_PRESENT, film.roller.max() > INK_PRESENT]

    def observe(self, view, pose, panel, dt, time):
        """Read actual solver impulses and the loaded face of the contact manifold."""
        if self.first is not None or not self.loaded.any():
            return
        forces, points, normals, gaps, counts, starts = view.get_contact_data(dt)
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        for i, surface in enumerate(SURFACES):
            if not self.loaded[i] or counts[0, i] == 0:
                continue
            index = slice(int(starts[0, i]), int(starts[0, i]+counts[0, i]))
            local_points = (points[index] - pose[:3]) @ rotation
            push_from_print_face = normals[index] @ rotation[:, 2] > .5
            undersurface = ((local_points[:, 2] <= 0.) &
                            (np.abs(local_points[:, :2]) <= np.asarray(PANEL_SIZE)/2 + .0008).all(axis=1))
            touches = (forces[index].ravel() > 1e-5) & (gaps[index].ravel() <= .0002)
            if i == 0:
                touches &= normals[index, 2] > .5  # Working top of the slab.
            hits = np.flatnonzero(push_from_print_face & undersurface & touches)
            if hits.size:
                h = hits[0]
                self.first = {"surface":surface, "panel":panel, "time_s":time,
                              "force_N":float(forces[index].ravel()[h]),
                              "separation_m":float(gaps[index].ravel()[h]),
                              "point_m":points[index][h].tolist()}
                return


def archived_clearance(panels, drum, table_z):
    """Lower bounds from swept world-axis envelopes enclosing every rotation."""
    def swept(centres, extent):
        return (np.minimum(centres[:-1], centres[1:])-extent,
                np.maximum(centres[:-1], centres[1:])+extent)

    paper_lo, paper_hi = swept(panels[..., :3], PAPER_RADIUS)
    drum_lo, drum_hi = swept(drum[:, None, :3], DRUM_RADIUS)
    slab = np.array([*PALETTE[:2], table_z+PALETTE[2]/2])
    slab_half = np.array([*PALETTE_SIZE, PALETTE[2]]) / 2
    evidence = {}
    for name, lo, hi in (("slab", slab-slab_half, slab+slab_half), ("roller", drum_lo, drum_hi)):
        gap = np.maximum(paper_lo-hi, lo-paper_hi).max(axis=-1)
        step, panel = np.unravel_index(gap.argmin(), gap.shape)
        evidence[name] = {"separating_axis_m":float(gap[step,panel]), "interval":int(step),
                          "panel":int(panel)}
    return evidence

"""Score supported closure of each clamp and hands-off retention."""

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import (
    CLAMP_Y, EDGE_X, PAD_LENGTH, PAD_THICKNESS, PAD_WIDTH, PAD_X, PAD_Z,
    PARK_ANGLE, PLATE_THICKNESS, TABLE_THICKNESS, TABLE_Z,
)
from scene import HZ

HOLD_SECONDS = 2
FORCE_MIN = 2.0
MARK_DEPTH = .026
MARK_HALF = .013
TABLE_BOTTOM = TABLE_Z - TABLE_THICKNESS
OPEN_ANGLE, LOADED_ANGLE = .80, .63
CONTACT_TOLERANCE = .0015


def alignment(pose):
    rotation = Rotation.from_quat(pose[3:])
    tilt = np.degrees(np.arccos(np.clip(rotation.apply([0, 0, 1])[2], -1, 1)))
    yaw = rotation.as_euler("zyx", degrees=True)[0]
    error = np.asarray(pose[:3]) - [EDGE_X, 0, TABLE_Z]
    return (abs(error[0]) <= .006 and abs(error[1]) <= .008 and abs(error[2]) <= .003
            and tilt <= 3 and abs(yaw) <= 5), error, float(tilt)


def on_square(points, y):
    return (abs(points[..., 0] - MARK_DEPTH) <= MARK_HALF) & (abs(points[..., 1] - y) <= MARK_HALF)


def under_table(points, mark_y):
    return ((EDGE_X + .011 <= points[..., 0]) & (points[..., 0] <= EDGE_X + .045)
            & (abs(points[..., 1] - mark_y) <= .015))


def pad_face(patch, center, rotation, sign):
    """Identify contact points on the working face of the pad."""
    local = rotation.inv().apply(patch[:, :3] - center)
    return ((abs(local[:, 0]) <= PAD_LENGTH / 2 + CONTACT_TOLERANCE)
            & (abs(local[:, 1]) <= PAD_WIDTH / 2 + CONTACT_TOLERANCE)
            & (abs(local[:, 2] + sign * PAD_THICKNESS / 2) <= CONTACT_TOLERANCE))


def jaw_seating(pose, links, hinge, forces, patches):
    board_rotation = Rotation.from_quat(pose[3:])
    origin = pose[:3]
    board_up = board_rotation.apply([0., 0., 1.])
    centers, faces, normals = [], [], []
    for link, sign in zip(links, (1, -1)):
        rotation = Rotation.from_quat(link[3:])
        center = link[:3] + rotation.apply([PAD_X, 0, sign * PAD_Z])
        pad_rotation = rotation * Rotation.from_euler("y", sign * PARK_ANGLE / 2)
        centers.append(center)
        normals.append(pad_rotation)
        faces.append(center + pad_rotation.apply([0, 0, -sign * PAD_THICKNESS / 2]))
    upper, lower = faces
    upper_face_local = board_rotation.inv().apply(upper - origin)
    footprint = np.array([upper_face_local[:2], lower[:2]])
    top_points = board_rotation.inv().apply(patches[0][:, :3] - origin)
    lower_points = patches[1][:, :3]
    upper_axial = patches[0][:, 3:6] @ board_up
    lower_axial = -patches[1][:, 5]
    top_face = (pad_face(patches[0], centers[0], normals[0], 1) & (upper_axial > .8)
                & (abs(top_points[:, 2] - PLATE_THICKNESS) <= CONTACT_TOLERANCE))
    bottom_face = (pad_face(patches[1], centers[1], normals[1], -1) & (lower_axial > .8)
                   & (abs(lower_points[:, 2] - TABLE_BOTTOM) <= CONTACT_TOLERANCE))
    top_load = top_face * upper_axial * patches[0][:, 6]
    lower_load = bottom_face * lower_axial * patches[1][:, 6]

    loaded = (hinge <= LOADED_ANGLE and forces[0, 0, 2] > FORCE_MIN and forces[1, 1, 2] < -FORCE_MIN
              and top_load.sum() > FORCE_MIN and lower_load.sum() > FORCE_MIN)
    open_site = seated_site = None
    across, pad_contact_sites = [], []
    mark_loads = []
    for site, y in enumerate(CLAMP_Y):
        mark_y = (origin + board_rotation.apply([MARK_DEPTH, y, PLATE_THICKNESS]))[1]
        square_points = on_square(top_points, y)
        underside_points = under_table(lower_points, mark_y)
        upper_n = float(top_load[square_points].sum())
        lower_n = float(lower_load[underside_points].sum())
        if hinge <= LOADED_ANGLE and np.any(top_face & square_points) and np.any(bottom_face & underside_points):
            pad_contact_sites.append(site)
        mark_loads.append([upper_n, lower_n])
        if (under_table(lower, mark_y) and abs(upper_face_local[0] - MARK_DEPTH) <= MARK_HALF + PAD_LENGTH / 2
                and abs(upper_face_local[1] - y) <= MARK_HALF + PAD_WIDTH / 2):
            across.append(site)
            if (hinge >= OPEN_ANGLE and on_square(upper_face_local, y)
                    and upper_face_local[2] >= PLATE_THICKNESS + .004 and lower[2] <= TABLE_BOTTOM - .004):
                open_site = site
        if loaded and upper_n > FORCE_MIN and lower_n > FORCE_MIN:
            seated_site = site
    return open_site, seated_site, loaded, across, pad_contact_sites, footprint, centers, mark_loads


class Score:
    def __init__(self):
        self.last_open = {}
        self.pending = {}
        self.credit = {}
        self.last_work = {}
        self.streaks = [0, 0]
        self.anchor = None

    def update(self, step, pose, velocity, clamp_data, robot_contact, left_tips, closest_hand):
        aligned, error, tilt = alignment(pose)
        left_upward = float(-left_tips[:, 2].sum())
        supported = bool(aligned and left_tips[:, 2].min() < -1. and left_tips[:, 2].max() > 1.
                         and left_upward > .4)
        seated_clamps, credited_sites = set(), set()
        centers, measurements, changes = [], {}, []
        for name, (links, hinge, forces, patches, hand) in clamp_data.items():
            opened, site, loaded, across, pad_sites, footprint, pads, loads = jaw_seating(
                pose, links, hinge, forces, patches)
            centers.append([p.tolist() for p in pads])
            working = bool(np.linalg.norm(hand, axis=-1).max() > 1.)
            if working:
                self.last_work[name] = step

            if name in self.credit:
                credited_site, anchor = self.credit[name]
                # While the clamp stays seated, a temporary load redistribution pauses scoring.
                if credited_site not in pad_sites or np.linalg.norm(footprint - anchor, axis=1).max() > .004:
                    del self.credit[name]
                    changes.append(dict(clamp=name, site=credited_site, event="seat_lost"))

            # Enter the straddle with jaws open. Holding an already seated clamp cannot arm a new event.
            if opened is not None and opened != self.last_open.get(name) and supported and working:
                self.pending[name] = (opened, step)
                changes.append(dict(clamp=name, site=opened, event="supported_open"))
            self.last_open[name] = opened
            if name in self.pending:
                target, start = self.pending[name]
                if not supported or target not in across:
                    del self.pending[name]
                    changes.append(dict(clamp=name, site=target, event="support_or_straddle_lost"))
                elif loaded:
                    del self.pending[name]
                    if site == target and step - self.last_work[name] <= round(.25 * HZ):
                        self.credit[name] = (site, footprint)
                        changes.append(dict(clamp=name, site=site, event="supported_seat", open_step=start))
            if site is not None:
                seated_clamps.add(site)
                if name in self.credit and self.credit[name][0] == site:
                    credited_sites.add(site)
            measurements[name] = dict(hinge_rad=hinge, open_site=opened, seated_site=site,
                                      upper_lower_mark_loads_N=loads, right_tip_forces_N=hand.tolist(),
                                      pad_contact_patches=[p.tolist() for p in patches], opposed_pair_forces_N=forces.tolist())

        contact = float(np.max(np.linalg.norm(robot_contact, axis=-1)))
        clear = contact < .4 and min(closest_hand) > .025
        still = np.linalg.norm(velocity[:3]) < .015 and np.linalg.norm(velocity[3:]) < .22
        qualified = len(credited_sites)
        stable = aligned and still and clear and qualified > 0
        if self.anchor is not None:
            stable &= np.linalg.norm(pose[:3] - self.anchor) <= .003
        if stable:
            if self.anchor is None:
                self.anchor = pose[:3].copy()
        else:
            self.anchor = None
        for n in (1, 2):
            self.streaks[n-1] = self.streaks[n-1] + 1 if stable and qualified >= n else 0
        score = .5 * sum(streak >= HZ * HOLD_SECONDS for streak in self.streaks)
        return dict(score=score, plate_pose=pose.tolist(), velocity=velocity.tolist(),
                    alignment_error_m=error.tolist(), tilt_deg=tilt, aligned=bool(aligned),
                    left_finger_forces_N=left_tips.tolist(), left_net_upward_N=left_upward,
                    left_supporting=supported, held_during_attachment=sorted({site for site, _ in self.credit.values()}),
                    credited_pairs={name: site for name, (site, _) in self.credit.items()},
                    attachment_changes=changes, clamps=measurements,
                    clamp_pad_centers_m=centers, seated_sites=sorted(seated_clamps),
                    max_robot_contact_N=contact, hands_clear=bool(clear),
                    hand_clearance_m=closest_hand, stable_steps=self.streaks.copy())

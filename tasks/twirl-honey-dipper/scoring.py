"""Grade the video-observable motion and the final honey deposits."""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import YAM_URDF
from yam_arena.analysis import link_hulls
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.station import BASES
from yam_arena.urdf import forward_kinematics

from honey import inside_toast
from layout import DIPPER, GROOVES, JAR, JAR_INNER, JAR_RIM, JAR_RISER, JAR_SURFACE, TABLE_TOP, TOAST, TOAST_TOP
from world import DT

STRIDE = 4  # 60 Hz motion audit, as in the bench videos.
VIDEO_DT = STRIDE * DT
CONTACT_TOL = .002
WET_X, TIP_X = .085, .143
TOOL_R = .016
HORIZONTAL = np.sin(np.deg2rad(20))
CARRY_LINES = JAR[1] + np.array([.045, .095, .145, .195])
CARRY_TURN_DEG = 30.
SWEEP_TURN_DEG = 20.
SWEEP_SECONDS = 1.
MIN_DOSE_G = .20
MAX_SPILL_G = .15
LINKS = ("gripper", "tip_left", "tip_right")
JOINTS = ARM_JOINTS + FINGER_JOINTS
REST_X = JAR[0] + np.array([-.079, -.049])
REST_Z = TABLE_TOP + JAR_RISER + .0715 + .0064
HAND_CLEAR_M = .020
PARK_SECONDS = 1.
PARK_ROTATION_COS = np.cos(np.deg2rad(3.)/2)


@lru_cache(maxsize=1)
def hand_points():
    """Native collision hull vertices and the two distal pad sections."""
    fk = forward_kinematics(YAM_URDF, {})
    palm = fk["gripper"]
    samples = {}
    for name in LINKS:
        v = np.unique(np.concatenate([h.vertices for h in link_hulls()[name]]), axis=0)
        z = ((v @ fk[name][:3, :3].T + fk[name][:3, 3] - palm[:3, 3]) @ palm[:3, :3])[:, 2]
        samples[name] = v, v[z >= .124], np.linalg.norm(v, axis=1).max()
    return samples


def touches_toast(points):
    low = points[(points[:, 2] <= TOAST_TOP + CONTACT_TOL)
                 & (np.abs(points[:, 0]-TOAST[0]) < .070)
                 & (np.abs(points[:, 1]-TOAST[1]) < .049)]
    return any(inside_toast(p[:2]) for p in low)


def inspect_hands(q, root, axis, check_clearance=False):
    """Compare native jaw geometry with the marked handle and wet envelope."""
    dry_grip = wet_touch = food_touch = hand_near = False
    margin = HAND_CLEAR_M if check_clearance else CONTACT_TOL
    for side, joints in zip(("left", "right"), q.reshape(2, 8), strict=True):
        fk = forward_kinematics(YAM_URDF, dict(zip(JOINTS, joints, strict=True)))
        dry_pads = 0
        for name, (vertices, pad, bound) in hand_points().items():
            r = fk[name][:3, :3]
            p = fk[name][:3, 3] + BASES[side]
            relative = (p-root) @ axis
            radial = np.linalg.norm(p-root-relative*axis)
            if radial <= bound + TOOL_R + margin and -.012-bound-margin <= relative <= TIP_X+bound+margin:
                local_root, local_axis = (root-p) @ r, axis @ r
                delta = vertices - local_root
                along = np.einsum("ij,j->i", delta, local_axis)
                rad = np.einsum("ij,ij->i", delta, delta) - along**2
                if check_clearance:
                    radius = np.where(along < WET_X, .0075, TOOL_R)
                    end_distance = np.maximum(-.012-along, 0.) + np.maximum(along-TIP_X, 0.)
                    side_distance = np.maximum(np.sqrt(np.maximum(rad, 0.))-radius, 0.)
                    hand_near |= bool(np.any(np.hypot(end_distance, side_distance) <= HAND_CLEAR_M))
                wet_touch |= bool(np.any((along >= WET_X-CONTACT_TOL)
                    & (along <= TIP_X+CONTACT_TOL) & (rad <= (TOOL_R+CONTACT_TOL)**2)))
                if len(pad):
                    delta = pad-local_root
                    along = np.einsum("ij,j->i", delta, local_axis)
                    rad = np.einsum("ij,ij->i", delta, delta)-along**2
                    radius = np.where(along < .041, .0075, .0064)
                    dry_pads += bool(np.any((along >= -.012-CONTACT_TOL)
                        & (along < WET_X-CONTACT_TOL) & (rad <= (radius+CONTACT_TOL)**2)))
            if p[2]-bound <= TOAST_TOP+CONTACT_TOL and np.linalg.norm(p[:2]-TOAST) < bound+.075:
                food_touch |= touches_toast(vertices @ r.T+p)
        jaw_gap = -sum(joints[6:])
        dry_grip |= dry_pads == 2 and jaw_gap <= .020
    return dry_grip and not wet_touch, wet_touch, food_touch, hand_near


def in_rest(root, axis, head):
    if axis[0] < np.cos(np.deg2rad(12.)) or abs(root[0]-DIPPER[0]) > .020:
        return False
    along = (REST_X-root[0])/axis[0]
    supports = root + along[:, None]*axis
    return bool(np.all((.024 <= along) & (along < WET_X))
                and np.all(np.abs(supports[:, 1]-JAR[1]) <= .0045)
                and np.all(np.abs(supports[:, 2]-REST_Z) <= .0035)
                and np.linalg.norm(head[:2]-JAR) <= .020
                and JAR_SURFACE-.004 <= head[2] <= JAR_RIM+.008)


def roll_angle(frame):
    """Angle of the tool's longitudinal reference stripe relative to vertical."""
    axis = frame[:, 0]
    up = np.array([0., 0., 1.])
    marker = frame[:, 2]
    return np.degrees(np.arctan2(marker @ np.cross(axis, up), marker @ up))


def turn_between(a, b):
    return abs((b-a+180.) % 360.-180.)

class Scoring:
    def __init__(self):
        self.source_grasp = self.arrived = self.carry_pass = False
        self.grip_and_pitch = True
        self.crossing_angles = []
        self.wet_first_s = self.toast_touch_first_s = None
        self.sweep_angle = None
        self.windows = []
        self.parked_since = None
        self.parked_at = None
        self.parked_hold_s = 0.

    def advance(self, row):
        t = row[0]
        tick = round(t/DT)
        if tick % STRIDE:
            return
        root = row[17:20]
        frame = Rotation.from_quat(row[20:24]).as_matrix()
        axis = frame[:, 0]
        angle = roll_angle(frame)
        head = root + np.mean(GROOVES)*axis
        seated = in_rest(root, axis, head)
        dry, wet, food, hand_near = inspect_hands(row[1:17], root, axis, seated)
        if seated and not hand_near and self.arrived:
            pose = row[17:24]
            if (self.parked_since is None
                or np.linalg.norm(pose[:3]-self.parked_at[:3]) > .002
                or abs(pose[3:] @ self.parked_at[3:]) < PARK_ROTATION_COS):
                self.parked_since, self.parked_at = tick, pose.copy()
            self.parked_hold_s = (tick-self.parked_since)*DT
        else:
            self.parked_since = None
            self.parked_hold_s = 0.
        if wet and self.wet_first_s is None:
            self.wet_first_s = t

        horizontal = abs(axis[2]) <= HORIZONTAL
        over_jar = np.linalg.norm(head[:2]-JAR) <= JAR_INNER
        over_toast = head[2] > TOAST_TOP+.025 and inside_toast(head[:2])
        low_z = root[2] + min(-.012*axis[2], TIP_X*axis[2]) - TOOL_R
        if low_z <= TOAST_TOP+CONTACT_TOL and np.linalg.norm(head[:2]-TOAST) < .210:
            # Probe the solid dipper's axis against the toast footprint. The
            # cross-section bound includes all collar radii and the dry shaft.
            for x in np.linspace(-.012, TIP_X, 33):
                centre = root + x*axis
                radius = .0075 if x < WET_X else TOOL_R
                if centre[2]-radius > TOAST_TOP+CONTACT_TOL:
                    continue
                points = centre + radius*np.array([[0.,0.,0.], [1.,0.,0.], [-1.,0.,0.],
                                                   [0.,1.,0.], [0.,-1.,0.]])
                food |= any(inside_toast(p[:2]) for p in points)
        if food and self.toast_touch_first_s is None:
            self.toast_touch_first_s = t

        if not self.arrived:
            if over_jar and head[2] <= JAR_SURFACE+.035:
                self.source_grasp = dry
                self.grip_and_pitch = True
                self.crossing_angles.clear()
            if self.source_grasp and not over_jar:
                self.grip_and_pitch &= dry and horizontal
                while len(self.crossing_angles) < len(CARRY_LINES):
                    if head[1] < CARRY_LINES[len(self.crossing_angles)]:
                        break
                    self.crossing_angles.append(angle)
            if self.source_grasp and over_toast:
                self.arrived = True
                turns = [turn_between(a,b) for a,b in zip(self.crossing_angles,self.crossing_angles[1:])]
                self.carry_pass = bool(self.grip_and_pitch and dry and horizontal
                                       and len(turns) == len(CARRY_LINES)-1
                                       and min(turns) >= CARRY_TURN_DEG)

        frame_id = tick//STRIDE
        clear = dry and horizontal and over_toast and low_z > TOAST_TOP+.008 and not food
        if clear:
            if self.sweep_angle is None or turn_between(self.sweep_angle, angle) > SWEEP_TURN_DEG:
                self.sweep_angle = angle
                self.windows.append([frame_id, frame_id])
            else:
                self.windows[-1][1] = frame_id
        else:
            self.sweep_angle = None

    def evaluate(self, landings):
        toast = [d for d in landings if d["surface"] == "toast"]
        windows = [(a,b) for a,b in self.windows if (b-a)*VIDEO_DT >= SWEEP_SECONDS]
        mixed = False
        if toast:
            first = round(toast[0]["time_s"]/DT)//STRIDE
            last = round(toast[-1]["time_s"]/DT)//STRIDE
            mixed = not any(a <= first <= last <= b for a,b in windows)
        dose = sum(d["mass_g"] for d in toast)
        spill = sum(d["mass_g"] for d in landings if d["surface"] in ("plate", "table"))
        span = np.ptp([d["p"][1] for d in toast]).item() if toast else 0.
        valid_deposits = dose >= MIN_DOSE_G and not mixed and self.toast_touch_first_s is None
        clean_carry = self.carry_pass and spill <= MAX_SPILL_G
        parked = self.parked_hold_s >= PARK_SECONDS
        if self.wet_first_s is not None or not valid_deposits:
            score = 0.
        elif clean_carry:
            if parked and dose >= .70 and span >= .030:
                score = 1.
            else:
                score = .75 if dose >= .30 else .50
        else:
            score = .25
        return dict(score=score, carry_pass=self.carry_pass, toast_g=dose, spill_g=spill,
                    mixed_phase_toast=mixed, toast_span_m=span, parked=parked,
                    parked_hold_s=self.parked_hold_s,
                    checkpoint_roll_deg=self.crossing_angles,
                    band_turns_deg=[turn_between(a,b) for a,b in zip(self.crossing_angles,self.crossing_angles[1:])],
                    wet_contact_first_s=self.wet_first_s, toast_contact_first_s=self.toast_touch_first_s,
                    qualifying_sweep_s=[[a*VIDEO_DT,b*VIDEO_DT] for a,b in windows])


def score_record(rows, landings):
    scorer = Scoring()
    for row in rows:
        scorer.advance(row)
    return scorer.evaluate(landings)


def rescore(root):
    measurements = json.loads((root/"measurements.json").read_text())
    return score_record(np.load(root/"trajectory.npy"), measurements["landings"])


if __name__ == "__main__":
    print(json.dumps(rescore(Path(sys.argv[1])), indent=2))

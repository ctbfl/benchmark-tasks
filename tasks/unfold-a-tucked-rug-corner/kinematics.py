"""Pad-space motion planning using the unmodified YAM URDF and joint bounds."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yam_arena import station
from yam_arena.constants import TCP_OFFSET, FINGER_OPEN, ARM_JOINTS
from yam_arena.ik import LOWER, UPPER, tcp_pose
from yam_arena.analysis import link_hulls, fk_link_poses_world

PAD_Z = .140


def attitude(yaw_deg, pitch_deg=35., roll_deg=0.):
    pitch, yaw = np.deg2rad([pitch_deg, yaw_deg])
    z = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), -np.sin(pitch)])
    x = np.array([np.sin(pitch)*np.cos(yaw), np.sin(pitch)*np.sin(yaw), np.cos(pitch)])
    y = np.cross(z, x)
    return np.column_stack((x, y, z)) @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()


def pad_pose(q, side):
    p, r = tcp_pose(q, station.BASES[side])
    return p + r[:, 2] * (PAD_Z - TCP_OFFSET[2]), r


def solve(point, rotation, side, seed):
    point, seed = np.asarray(point), np.asarray(seed)
    def residual(q):
        p, r = pad_pose(q, side)
        return np.r_[20*(p-point), Rotation.from_matrix(rotation @ r.T).as_rotvec(), .002*(q-seed)]
    lower, upper = LOWER + .025, UPPER - .025
    result = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                           max_nfev=100, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    p, r = pad_pose(result.x, side)
    errors = (float(np.linalg.norm(p-point)), float(np.linalg.norm(Rotation.from_matrix(rotation @ r.T).as_rotvec())))
    return result.x, errors


def tool_floor_clearance(q, side, opening):
    joints = dict(zip(ARM_JOINTS, q, strict=True))
    joints.update(joint7=FINGER_OPEN*opening, joint8=FINGER_OPEN*opening)
    poses = fk_link_poses_world(joints, station.BASES[side])
    low = np.inf
    for link in ("gripper", "tip_left", "tip_right"):
        t = poses[link]
        for hull in link_hulls()[link]:
            low = min(low, (hull.vertices @ t[:3, :3].T + t[:3, 3])[:, 2].min())
    return float(low - station.TABLE_TOP)

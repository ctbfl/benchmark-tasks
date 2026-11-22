"""Check full native URDF convex colliders for bimanual motion planning."""
import numpy as np
from scipy.optimize import linprog
from yam_arena.analysis import fk_link_poses_world, world_hulls
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
from yam_arena.station import BASES
from yam_arena.station import TABLE_TOP


def hulls(q, side, opening):
    joints = dict(zip(ARM_JOINTS, q, strict=True))
    joints.update(joint7=FINGER_OPEN*opening, joint8=FINGER_OPEN*opening)
    result = world_hulls(fk_link_poses_world(joints, BASES[side]))
    return result, np.array([h.vertices.min(0) for _,h in result]), np.array([h.vertices.max(0) for _,h in result])


def clear(first, second, margin=.01):
    aa, lo1, hi1 = first
    bb, lo2, hi2 = second
    separation = np.maximum(lo2[None]-hi1[:,None], lo1[:,None]-hi2[None]).max(2)
    for i,j in np.argwhere(separation < margin):
        a,b = aa[i][1].planes, bb[j][1].planes
        result = linprog(np.zeros(3), A_ub=np.concatenate((a[:,:3],b[:,:3])),
            b_ub=np.r_[a[:,3]+margin/2,b[:,3]+margin/2], bounds=[(None,None)]*3, method="highs")
        if result.success:
            return False, (aa[i][0],bb[j][0])
        if result.status != 2:
            raise RuntimeError(f"Collider intersection check failed: {result.message}")
    return True, None

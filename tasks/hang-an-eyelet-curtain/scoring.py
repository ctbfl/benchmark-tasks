"""0.5 per eyelet on the rod between its two supports, only while the rod lies in both stand cradles; otherwise 0."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import (RINGS, RING_RI, ROD_HALF, ROD_HEIGHT, ROD_R, SUPPORT_HALF_SPAN, CRADLE_SLOT,
                    PRONG_TOP, CRADLE_T, KNOB_GAP)


def ring_on_rod(ring_pose, rod_pose):
    """The rod axis crosses the grommet plane inside the hole, within the rod's length."""
    p, n = ring_pose[:3], Rotation.from_quat(ring_pose[3:]).apply([0., 0., 1.])
    c, u = rod_pose[:3], Rotation.from_quat(rod_pose[3:]).apply([0., 1., 0.])
    cos = float(u@n)
    if abs(cos) < .2:
        return False, 1., 0.
    t = float((p-c)@n/cos)
    radial = float(np.linalg.norm(c+t*u-p))
    return bool(radial < RING_RI and abs(t) < ROD_HALF), radial, t


def rod_in_cradles(rod_pose, stand_pose):
    """Both cradle sections of the rod sit inside the U slots, below the prong tops."""
    sr = Rotation.from_quat(stand_pose[3:])
    c = sr.inv().apply(rod_pose[:3]-stand_pose[:3])
    u = sr.inv().apply(Rotation.from_quat(rod_pose[3:]).apply([0., 1., 0.]))
    details = []
    for y in (-SUPPORT_HALF_SPAN, SUPPORT_HALF_SPAN):
        t = (y-c[1])/u[1] if abs(u[1]) > .5 else 1e9
        q = c+t*u
        inside = bool(abs(q[0]) < CRADLE_SLOT/2 and ROD_HEIGHT-.004 < q[2] < ROD_HEIGHT+PRONG_TOP-ROD_R
                      and abs(abs(t)-SUPPORT_HALF_SPAN) < CRADLE_T/2+KNOB_GAP+.002)
        details.append(dict(inside=inside, slot_offset_m=float(q[0]), axis_height_m=float(q[2]),
                            axial_shift_m=float(abs(t)-SUPPORT_HALF_SPAN)))
    return all(d['inside'] for d in details), details


def evaluate(ring_poses, rod_pose, stand_pose):
    rings = [ring_on_rod(p, rod_pose) for p in ring_poses]
    assert len(rings) == RINGS
    cradled, cradles = rod_in_cradles(rod_pose, stand_pose)
    on_rod = [r[0] for r in rings]
    u = Rotation.from_quat(rod_pose[3:]).apply([0., 1., 0.])
    facing = np.array([Rotation.from_quat(p[3:]).apply([0., 0., 1.])@u for p in ring_poses])
    # Each eyelet centre lies between the inner faces of the two cradles (hence also inside both end knobs).
    sr = Rotation.from_quat(stand_pose[3:])
    along = [float(sr.inv().apply(np.asarray(p[:3])-stand_pose[:3])[1]) for p in ring_poses]
    between = [bool(abs(y) < SUPPORT_HALF_SPAN-CRADLE_T/2) for y in along]
    # 0.5 per eyelet that is on the rod between the supports; 0 whenever the rod is not in both cradles.
    hung = sum(o and b for o, b in zip(on_rod, between))
    score = hung/RINGS if cradled else 0.
    success = score == 1.
    return dict(score=float(score), success=success, rings_hung=int(hung), rings_on_rod=int(sum(on_rod)),
                ring_on_rod=on_rod, ring_radial_m=[r[1] for r in rings], ring_axial_m=[r[2] for r in rings],
                ring_facing=facing.tolist(), ring_between_supports=between, ring_stand_y_m=along,
                rod_on_stand=cradled, cradles=cradles)

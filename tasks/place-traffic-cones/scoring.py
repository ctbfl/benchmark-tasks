"""Final-state binary scoring using the user-confirmed geometric rules."""
import numpy as np
from scipy.spatial.transform import Rotation
from config import BASE_WIDTH, BOARD_CENTER, BOARD_SIZE, LENGTH, MAX_STEPS, CONTROL_HZ, nearest


def measure(roots, robot_contact, table_top, time_s):
    roots = np.asarray(roots, dtype=float)
    assert roots.shape == (6, 7) and np.isfinite(roots).all()
    rotations = Rotation.from_quat(roots[:, 3:]).as_matrix()
    tilt = np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1)))
    corners = np.array([[x, y, 0.] for x in (-BASE_WIDTH/2, BASE_WIDTH/2)
                        for y in (-BASE_WIDTH/2, BASE_WIDTH/2)])
    bottom = (np.einsum('nij,kj->nki', rotations, corners) + roots[:, None, :3])[:, :, 2].min(1)
    on_board = (np.abs(roots[:, :2] - BOARD_CENTER) <= np.array(BOARD_SIZE[:2])/2-BASE_WIDTH/2).all(1)
    surface = table_top + on_board * BOARD_SIZE[2]
    ground_error = np.abs(bottom-surface)
    distance, arc = nearest(roots[:, :2])
    ordered = np.sort(arc)
    gaps = np.diff(ordered)
    gap_error = abs(float(gaps.max()) - LENGTH/5)
    endpoints = [float(ordered[0]), float(LENGTH-ordered[-1])]
    # 1 mm is a surface/contact measurement tolerance, not a placement tolerance.
    grounded = ground_error <= .001
    released = np.asarray(robot_contact) < .001
    conditions = dict(upright=bool((tilt <= 10).all()), grounded=bool(grounded.all()),
        released=bool(released.all()), on_curve=bool((distance <= .02).all()),
        endpoints=bool(max(endpoints) <= .02), spacing=bool(gap_error <= .05),
        in_time=bool(time_s <= MAX_STEPS/CONTROL_HZ + 1e-8))
    return dict(time_s=float(time_s), score=float(all(conditions.values())), conditions=conditions,
        tilt_deg=tilt.tolist(), ground_error_m=ground_error.tolist(),
        curve_distance_m=distance.tolist(), arc_m=arc.tolist(), gaps_m=gaps.tolist(),
        expected_gap_m=LENGTH/5, max_gap_error_m=gap_error, endpoint_error_m=endpoints,
        robot_contact_N=np.asarray(robot_contact).tolist())

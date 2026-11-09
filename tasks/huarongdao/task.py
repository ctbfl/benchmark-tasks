"""Puzzle geometry, shortest legal solution and final-state scoring."""

from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

INITIAL = (1, 2, 3, 5, 0, 6, 4, 7, 8)
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
PITCH = .072
CENTER = (.30, -.04)
TABLE_Z = .7475
FLOOR = .006
TILE_HEIGHT = .027
TILE_Z = TABLE_Z + FLOOR
ROTATION = Rotation.from_euler("z", -90, degrees=True).as_quat()


def cell(index):
    row, col = divmod(index, 3)
    return np.array([CENTER[0]+(1-row)*PITCH, CENTER[1]+(1-col)*PITCH, TILE_Z])


def solve(initial=INITIAL):
    """Return tile numbers for a shortest sequence of adjacent blank swaps."""
    if sorted(initial) != list(range(9)):
        raise ValueError("Expected each tile 0..8 exactly once")
    queue = deque([(tuple(initial), [])])
    seen = {tuple(initial)}
    while queue:
        state, moves = queue.popleft()
        if state == GOAL:
            return moves
        blank = state.index(0)
        for index in range(9):
            if abs(index//3-blank//3)+abs(index%3-blank%3) != 1:
                continue
            next_state = list(state)
            next_state[blank], next_state[index] = state[index], 0
            key = tuple(next_state)
            if key not in seen:
                seen.add(key)
                queue.append((key, moves+[state[index]]))
    raise ValueError("Unsolvable puzzle")


def score(poses, robot_contact_n, time_s):
    """Evaluate the final arrangement; never retain an intermediate best score."""
    poses = np.asarray(poses, dtype=float)
    contacts = np.asarray(robot_contact_n, dtype=float)
    if poses.shape != (8, 7) or contacts.shape != (8,):
        raise ValueError("Expected eight tile poses and eight contact magnitudes")
    if not np.isfinite(poses).all() or not np.isfinite(contacts).all() or not np.isfinite(time_s):
        raise ValueError("Non-finite score input")
    targets = np.array([cell(i) for i in range(8)])
    errors = np.linalg.norm(poses[:, :3]-targets, axis=1)
    angles = (Rotation.from_quat(poses[:, 3:])*Rotation.from_quat(ROTATION).inv()).magnitude()
    aligned = bool(np.count_nonzero(errors > .003+1e-8) <= 1 and np.max(errors) <= .010+1e-8)
    seated = bool(np.all(np.abs(poses[:, 2]-TILE_Z) <= .001))
    upright = bool(np.all(angles <= np.deg2rad(10)))
    released = bool(np.max(contacts) < .05)
    success = aligned and seated and upright and released and 0 <= time_s <= 120+1e-8
    return dict(score=float(success), time_s=float(time_s), errors_m=errors.tolist(),
                orientation_error_deg=np.rad2deg(angles).tolist(), aligned=aligned,
                seated=seated, upright=upright, released=released,
                robot_contact_n=contacts.tolist())

"""Cut, shape, and final placement measurements."""
import numpy as np
from cord import INCISION_DEPTH, cut_fraction
from geometry import BOARD_SIZE, CUT_X, FLAT, HANDLE_SIZE, RADIUS

SHAPE_TOLERANCE = .11


def section_scale(damage):
    """Height and width scales of the simulated end faces."""
    return 1.-.75*damage, 1.+.25*damage


def shape_preserved(height, width, start_height, start_width):
    """Check the two dimensions of an end-on outline independently."""
    h, w = height/start_height, width/start_width
    lower, upper = 1.-SHAPE_TOLERANCE, 1.+SHAPE_TOLERANCE
    return bool(lower <= h <= upper and lower <= w <= upper)


def grade(intact, on_board, released, aligned, settled_handoff):
    if not (intact and on_board):
        return 0.
    if not released:
        return .35 if aligned else 0.
    return 1. if settled_handoff else .75


def score(cord, poses, speeds, spin, board, tips, openings):
    centers = poses[2:, :3]
    shift = np.linalg.norm(centers.mean(axis=0)[:2]-[CUT_X, 0.])
    split = np.linalg.norm(centers[1]-centers[0])
    fraction_cut = cut_fraction(cord.height)
    handle_xy = np.abs(poses[:2, :2]-[CUT_X, 0.])
    clearance = np.linalg.norm(np.asarray(tips)[:, None, :]-poses[None, :2, :3], axis=2).min()
    handoff = bool(np.max(poses[:2, 2]) < board+HANDLE_SIZE[2]/2+.007
                   and np.min(poses[:2, 2]) > board+.004
                   and np.all(handle_xy < np.asarray(BOARD_SIZE[:2])/2-.014)
                   and np.max(speeds[:2]) < .015 and min(openings) > .75 and clearance > .04)
    height, width = section_scale(cord.damage)
    groove = float(cord.groove_depth.max())
    intact = shape_preserved(height, width, 1., 1.) and groove < INCISION_DEPTH
    on_board = bool(np.max(np.abs(centers[:, 0]-CUT_X)) < .12
                    and np.max(np.abs(centers[:, 1])) < .17
                    and np.max(np.abs(centers[:, 2]-board-RADIUS+FLAT)) < .012)
    resting = bool(np.max(speeds[2:]) < .015 and np.max(spin[2:]) < .5)
    value = grade(intact, on_board, cord.released, cord.aligned_loop,
                  resting and handoff and shift < .027)
    return dict(score=value, fraction_cut=float(fraction_cut), released=cord.released,
                max_cut_offset_m=cord.max_yield_offset, off_plane_loaded_s=cord.off_plane_loaded_s,
                off_plane_groove_m=groove, intact=intact,
                aligned_loop=bool(cord.aligned_loop), shape_axis_ratio=height/width,
                height_fraction=height, width_fraction=width, center_shift_m=float(shift),
                center_distance_m=float(split), dough_on_board=on_board, dough_resting=resting,
                tool_put_down=handoff, gripper_clearance_m=float(clearance), wrap_peak_rad=cord.wrap_max, peak_pull_N=cord.peak_pull,
                peak_radial_N=cord.peak_contact, peak_cut_load_N=cord.peak_cut_force)

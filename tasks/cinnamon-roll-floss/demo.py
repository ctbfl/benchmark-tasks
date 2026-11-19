"""Bimanual Cartesian waypoints executed with the normal joint actions."""
import numpy as np
from inspect_robots import Action
from yam_arena.constants import READY
from yam_arena.ik import matrix_to_quat_xyzw, solve
from yam_arena.station import BASES
from yam_arena.urdf import axis_angle_matrix
from geometry import CUT_X, HANDLE_X, HANDLE_Y


def script(sim):
    z = sim.board
    down = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    bases = [BASES[key] for key in ("left", "right")]
    q = [np.array(READY), np.array(READY)]
    start = None
    grip = np.ones(2)
    wrist = np.zeros((2, 3))
    crossed_wrist = np.array([[-30., 70., 40.], [120., 25., 0.]])

    def pair(x, y, h):
        return np.array([[x, y, z+h], [x, -y, z+h]])

    def aligned(y, low, high, center=CUT_X):
        return pair(center, y, 0.) + [[-.013, 0., low], [.013, 0., high]]

    def orient(wrist_angles):
        yaw, pitch, roll = np.deg2rad(wrist_angles)
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.])
        lean = axis_angle_matrix(np.cross([0., 0., -1.], direction), pitch)
        spin = axis_angle_matrix(np.array([0., 0., 1.]), roll)
        return matrix_to_quat_xyzw(lean @ down @ spin)

    def move(name, target, seconds, opening=0, angles=((0., 0., 0.), (0., 0., 0.))):
        nonlocal q, start, grip, wrist
        angles = np.asarray(angles)
        opening = np.broadcast_to(opening, (2,))
        print(f"{sim.tick/240:.2f}s {name}", flush=True)
        steps = round(seconds*20)
        if start is None:
            dest = [solve(p, orient(r), a, base_pos=b, iters=450)
                    for p, r, a, b in zip(target, angles, q, bases)]
            if any(e > .003 or a > .04 for _, e, a in dest):
                raise RuntimeError(f"Grips are outside the approach envelope: {dest}")
        for i in range(1, steps+1):
            u = i/steps
            u = u*u*(3-2*u)
            if start is None:
                joints = [(1-u)*v+u*d[0] for v, d in zip(q, dest)]
            else:
                poses = (1-u)*start+u*target
                rotations = (1-u)*wrist+u*angles
                dest = [solve(p, orient(r), a, base_pos=b, iters=40)
                        for p, r, a, b in zip(poses, rotations, q, bases)]
                if any(e > .005 or a > .075 for _, e, a in dest):
                    raise RuntimeError(f"Unreachable {name}: {poses.tolist()} {dest}")
                q = joints = [v[0] for v in dest]
            g = (1-u)*grip+u*opening
            sim.step(Action(data=np.r_[joints[0], g[0], joints[1], g[1]]))
        q = [v[0] for v in dest]
        grip, start, wrist = opening, np.asarray(target), angles
        print(sim.summary(), flush=True)

    pickup = pair(HANDLE_X, HANDLE_Y, .018)
    move("Approach grips", pair(HANDLE_X, HANDLE_Y, .105), 2., 1.)
    move("Lower to grips", pickup, 1., 1.)
    move("Close both grips", pickup, .85)
    move("Straighten slack low on the board", pair(.220, .126, .018), 1.1)
    move("Feed floss beneath the center mark", pair(CUT_X+.008, .1265, .019), 2.)
    move("Lift the ends along both sides", pair(CUT_X, .101, .096), 1.45)
    move("Orient both wrists for clearance", aligned(.065, .0815, .1545), 1.45, angles=crossed_wrist)
    move("Cross the slack ends in the marked plane", aligned(-.021, .0815, .1545), 1.7, angles=crossed_wrist)
    center = float(sim.poses[2:, 0].mean())
    move("Take up slack toward the mark", aligned(-.040, .1035, .1715, center),
         1.1, angles=((-30., 70., 40.), (116., 30., 0.)))
    move("Draw the centered loop upward", aligned(-.085, .1435, .2065, center),
         2.35, angles=((-30., 70., 40.), (105., 41., 0.)))
    move("Lift the floss clear", aligned(-.090, .1805, .2475, center),
         .9, angles=((-30., 70., 40.), (100., 45., 0.)))
    move("Uncross above the dough", pair(.246, .113, .166), 1.7)
    drop = pair(.174, .119, .027)
    move("Return floss to board", drop, 1.45)
    move("Release grips", drop, .55, 1.)
    move("Move hands clear", pair(.160, .139, .141), .9, 1.)
    near = sim.poses[2, :3].copy()
    parked = pair(.160, .139, .141)[1]
    approach = np.vstack((near+[0., .076, .095], parked))
    side = np.vstack((near+[0., .076, .009], parked))
    move("Approach beside the near portion", approach, .9, (0., 1.))
    move("Lower beside the cut portion", side, .7, (0., 1.))
    separate = side + [[-.004, -.065, 0.], [0., 0., 0.]]
    move("Nudge the near portion away from the other", separate, 1.45, (0., 1.))
    move("Lift the hand clear of the dough", separate + [[0., 0., .08], [0., 0., 0.]], .75, (0., 1.))
    move("Withdraw and let both rolls rest", pair(.160, .139, .141), 1.65, 1.)

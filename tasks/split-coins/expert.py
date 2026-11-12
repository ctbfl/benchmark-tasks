"""Scripted planner that generated the demo: pour the jar, then sort coins found by top-camera CV.

The planner sees the coins only through the top camera's RGB image (rendered at 1920x1080 with
the calibrated pose and lens; see coin_vision.py) and reads only the robot's own joint and
gripper state. It never reads coin or jar poses. Every command it sends is a native 14-D
joint/gripper target, and the executed list is saved as runs/expert-<timestamp>/actions.npy;
demo_actions.npy is that list, replayed open loop by demo.py.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from yam_arena import station
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
from yam_arena.ik import smoothstep, solve

HERE = Path(__file__).resolve().parent
TABLE = station.TABLE_TOP
SIDES = ("left", "right")
# Right hand approaching the jar from -Y with fingers closing along world X.
JAR_GRIP = Rotation.from_matrix(np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])))
# Integral correction of steady joint error while planning (per second).
BIAS_GAIN = 3.
# Pushing: the coin sits between the open finger pads and the trailing pad's flat, vertical
# inner face pushes it along the closing axis. The tips' front and back faces are rounded at the
# bottom, so a coin pushed by them can slip underneath; the pad faces are square to the table.
CRADLE_OPENING = .33
CRADLE_GAP = .094*CRADLE_OPENING     # distance between the inner pad faces
FINGER = .004                        # fingertip thickness along the closing axis
FINGER_FACE = .0077                  # fingertip half-width across the push direction
TIP_CLEARANCE = .0002
PUSH_SPEED = .2
CLEARANCE = .0005
OWNER = {"penny": 0, "quarter": 1}
PARK = {0: np.array([.30, .21, TABLE + .11]), 1: np.array([.30, -.21, TABLE + .11])}
POUR = {"x0": .35, "x1": .33, "y": .02, "z": .09, "a0": 88., "a1": 118., "seconds": 6.}
SPREAD = {"r0": .004, "r1": .055, "turns": 2.5, "phase": 0., "height": .001, "seconds": 3., "side": None}
ROI_SAFE = (.21, .37)          # extraction pushes stay in the middle of the table
# Twelve candidate slots per frame for five coins, 30 mm apart and at least 13 mm inside the tape. Rows fill
# from the far side of the frame inward, so later pushes never cross coins already placed.
SLOTS = {kind: [np.array([x, sign*y]) for y in (.275, .245, .215) for x in (.335, .365, .395, .425)]
         for kind, sign in (("penny", 1), ("quarter", -1))}


def robust_solve(side, point, rotation, seed=None, iters=150):
    """Damped least squares from several seeds: the given one, READY and READY with the wrist
    rolled, since a 90 degree yaw change can trap the solver in a local minimum."""
    seeds = ([] if seed is None else [np.asarray(seed)]) + [np.asarray(READY) + [0, 0, 0, 0, 0, a]
                                                            for a in (0., 1.57, -1.57, 3.)]
    best = None
    for start in seeds:
        q, ep, er = solve(point, rotation.as_quat(), start, station.BASES[SIDES[side]],
                          damping=.015, iters=iters, tol=1e-5)
        if best is None or ep + .1*er < best[1] + .1*best[2]:
            best = (q, ep, er)
        if ep < .0015 and er < .02:
            break
    return best


def reach_ok(result):
    """Planning accepts only poses solved with margin, so execution from another seed succeeds."""
    return result[1] < .0008 and result[2] < .015


class Expert:
    def __init__(self, sim):
        self.sim = sim
        self.desired = sim.action.copy()
        self.previous = self.desired.copy()
        self.bias = np.zeros((2, 6))
        self.gripper_ids = [r.body_names.index("gripper") for r in sim.robots]
        self.log = []
        self.planned = {}
        self.history = []        # start points of earlier non-final pushes

    # ---------------------------------------------------------------- motion primitives
    def tcp(self, side):
        pose = self.sim.robots[side].data.body_link_pose_w.torch[0, self.gripper_ids[side]].cpu().numpy()
        rotation = Rotation.from_quat(pose[3:])
        return pose[:3] + rotation.apply([0, 0, .134]), rotation

    def step(self):
        action = self.desired.copy()
        for side in range(2):
            action[7*side:7*side+6] += self.bias[side]
        self.sim.step(action)
        measured = self.sim.measured_action()
        for side in range(2):
            sl = slice(7*side, 7*side+6)
            # Integrate only steady error; tracking lag during fast moves would wind up and overshoot.
            if np.abs(self.desired[sl] - self.previous[sl]).max() < .03:
                self.bias[side] = np.clip(self.bias[side] + BIAS_GAIN*self.sim.control_dt *
                                          (self.desired[sl] - measured[sl]), -.10, .10)
        self.previous = self.desired.copy()

    def wait(self, seconds):
        for _ in range(round(seconds/self.sim.control_dt)):
            self.step()

    def ik(self, side, point, rotation, seed, continuous=False):
        q, ep, er = robust_solve(side, point, rotation, seed)
        assert ep < .002 and er < .025, ("Unreachable pose", SIDES[side], np.round(point, 4), ep, er)
        # Along a straight-line move the joints must not jump to another arm configuration.
        assert not continuous or np.abs(q - seed).max() < .35, ("IK branch jump", SIDES[side], np.round(point, 4))
        return q

    def move_many(self, targets, seconds=2., settle=.2):
        """Cartesian straight-line TCP motion; targets: {side: (tcp point, rotation, gripper or None)}."""
        starts = {side: (self.planned[side] if side in self.planned else self.tcp(side)) for side in targets}
        rotations = {side: Slerp([0, 1], Rotation.concatenate([starts[side][1], target[1]]))
                     for side, target in targets.items()}
        grips = self.desired.copy()
        for fraction in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(fraction)
            for side, (point, _, opening) in targets.items():
                sl = slice(7*side, 7*side+6)
                position = np.asarray(starts[side][0])*(1-t) + np.asarray(point)*t
                self.desired[sl] = self.ik(side, position, rotations[side](t), self.desired[sl], fraction > 0)
                if opening is not None:
                    self.desired[7*side+6] = grips[7*side+6]*(1-t) + opening*t
            self.step()
        for side, (point, rotation, _) in targets.items():
            self.planned[side] = (np.asarray(point, float), rotation)
        self.wait(settle)

    def move(self, side, point, rotation, seconds=2., opening=None, settle=.2):
        self.move_many({side: (point, rotation, opening)}, seconds, settle)

    def transit_many(self, targets, seconds=2., settle=.2):
        """Joint-space interpolation between free-space poses."""
        start = self.desired.copy()
        goals = {side: self.ik(side, point, rotation, READY) for side, (point, rotation, _) in targets.items()}
        for fraction in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(fraction)
            for side, (_, _, opening) in targets.items():
                sl = slice(7*side, 7*side+6)
                self.desired[sl] = start[sl]*(1-t) + goals[side]*t
                if opening is not None:
                    self.desired[7*side+6] = start[7*side+6]*(1-t) + opening*t
            self.step()
        for side, (point, rotation, _) in targets.items():
            self.planned[side] = (np.asarray(point, float), rotation)
        self.wait(settle)

    def transit(self, side, point, rotation, seconds=2., opening=None, settle=.2):
        self.transit_many({side: (point, rotation, opening)}, seconds, settle)

    def joints(self, side, q, seconds=2., opening=None, settle=.2):
        start = self.desired.copy()
        sl = slice(7*side, 7*side+6)
        for fraction in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(fraction)
            self.desired[sl] = start[sl]*(1-t) + np.asarray(q)*t
            if opening is not None:
                self.desired[7*side+6] = start[7*side+6]*(1-t) + opening*t
            self.step()
        self.planned.pop(side, None)
        self.wait(settle)

    def grip(self, side, opening, seconds=.6, settle=.2):
        initial = self.desired[7*side+6]
        for t in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            self.desired[7*side+6] = initial*(1-t) + opening*t
            self.step()
        self.wait(settle)

    # ---------------------------------------------------------------- camera
    def report(self, phase, save=True):
        """Capture the top camera image used for planning and save it with the phase name."""
        from PIL import Image
        self.sim.phase = phase
        images = self.sim.capture(["top_cam"])
        if save:
            (self.sim.out / "screenshots").mkdir(exist_ok=True)
            Image.fromarray(images["top_cam"]).save(self.sim.out/"screenshots"/f"{self.sim.steps:05d}-{phase}.png")
        entry = {"phase": phase, "time_s": round(self.sim.time, 2)}
        self.log.append(entry)
        print("PHASE", json.dumps(entry), flush=True)
        return images["top_cam"]

    # ---------------------------------------------------------------- task phases
    def pour(self):
        from scene import JAR_XY
        jar = np.array([*JAR_XY, TABLE])
        grip_z = .045
        self.transit(1, jar + [0, -.12, grip_z+.03], JAR_GRIP, 2.2, opening=1.)
        self.move(1, jar + [0, -.10, grip_z], JAR_GRIP, .8)
        self.move(1, jar + [0, 0., grip_z], JAR_GRIP, 1.2)
        self.grip(1, 0., .8, .3)
        self.report("jar_grasped")
        held = jar + [0, 0, grip_z]
        self.move(1, held + [0, 0, .12], JAR_GRIP, 1.2)
        # Tilt the open top toward the robot (-X) to just before the coins slide, then drift along
        # a gentle zigzag while the tilt grows. The coins land in front of the jar, near the middle
        # of the table where both arms can reach across.
        p = POUR
        tilt = lambda angle: Rotation.from_euler("y", -angle, degrees=True) * JAR_GRIP
        self.move(1, [p["x0"], 0., TABLE + .14], JAR_GRIP, 1.2)
        self.move(1, [p["x0"], 0., TABLE + p["z"]], tilt(p["a0"]), 1.2, settle=.1)
        self.report("pour_start")
        steps = round(p["seconds"] / .25)
        for i in range(1, steps + 1):
            f = i / steps
            angle = p["a0"] + (p["a1"] - p["a0"])*f
            point = [p["x0"] + (p["x1"] - p["x0"])*f, p["y"]*np.sin(2*np.pi*f), TABLE + p["z"]]
            self.move(1, point, tilt(angle), .25, settle=0.)
        self.move(1, [p["x1"], 0., TABLE + p["z"]], tilt(p["a1"]), .4, settle=.3)
        self.report("pour_done")
        self.wait(.4)
        self.move(1, [POUR["x1"] + .03, -.02, TABLE + .16], JAR_GRIP, 1.2)
        self.move(1, held + [0, 0, .02], JAR_GRIP, 1.5)
        self.move(1, held + [0, 0, .002], JAR_GRIP, .6)
        self.grip(1, 1., .6)
        self.move(1, held + [0, -.10, 0.], JAR_GRIP, 1.)
        self.joints(1, READY, 1.5)
        image = self.report("jar_returned")
        from simulation import write_json
        from coin_vision import Camera
        # Both arms are clear of the tape here, so the tape fit calibrates the camera.
        self.camera = Camera.from_sensor(self.sim.base.scene["top_cam"], TABLE)
        score = self.camera.calibrate(image)
        print("CAMERA SHIFT", self.camera.shift.tolist(), "score", round(score, 3), flush=True)
        write_json(self.sim.out / "camera.json", self.camera.to_dict())

    # ---------------------------------------------------------------- sorting
    def push_rotation(self, side, direction, at):
        """Gripper pointing down with its closing (X) axis along the push direction (either sign)."""
        d = np.array([*direction, 0.]) / np.linalg.norm(direction)
        best = None
        for sign in (1, -1):
            x = sign*d
            z = np.array([0., 0., -1.])
            rotation = Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z)))
            q, ep, er = robust_solve(side, at, rotation, self.desired[7*side:7*side+6])
            cost = ep*50 + er + .2*np.abs(q - self.desired[7*side:7*side+6]).max()
            if best is None or cost < best[0]:
                best = (cost, rotation)
        return best[1]

    def lead(self, kind):
        """TCP offset ahead of the coin centre while the trailing pad pushes it."""
        from build_assets import COINS
        return CRADLE_GAP/2 - COINS[kind]["radius"]

    def segment_moves(self, side, kind, start, end, first, last):
        """Motions for one straight push of a coin centre from start to end."""
        d = (end - start) / np.linalg.norm(end - start)
        rotation = self.push_rotation(side, d, [*start, TABLE + .05])
        tcp_offset = self.pad_offset(rotation, d)
        tcp_z = TABLE + TIP_CLEARANCE + tcp_offset[2]
        start, end = start + tcp_offset[:2], end + tcp_offset[:2]
        moves = []
        # Land with the coin centred between the pads; the slack lets small errors pass.
        # (motion, TCP point, rotation, seconds, settle, gripper opening)
        g = CRADLE_OPENING
        if first:
            moves.append(("transit", np.array([*start, tcp_z + .06]), rotation, .7, .1, g))
        moves.append(("move", np.array([*start, tcp_z + .02]), rotation, .35, .3, g))
        moves.append(("move", np.array([*start, tcp_z]), rotation, .3, .2, g))
        finish = end + d*self.lead(kind)
        moves.append(("move", np.array([*finish, tcp_z]), rotation,
                      max(.4, np.linalg.norm(finish - start)/PUSH_SPEED), .05, g))
        # Back off the trailing pad before lifting so the coin is not dragged.
        moves.append(("move", np.array([*(finish - d*.004), tcp_z]), rotation, .15, .0, g))
        moves.append(("move", np.array([*(finish - d*.004), tcp_z + (.05 if last else .015)]), rotation, .3, .0, g))
        return moves

    @staticmethod
    def pad_offset(rotation, d):
        """TCP offset from the coin centre that puts the trailing pad's bottom inner edge half a
        pad gap behind the coin, with the fingertips at table height."""
        def tip_corners(side_sign):
            xs = side_sign*np.array([CRADLE_GAP/2, CRADLE_GAP/2 + FINGER])
            corners = np.array([[x, y, PAD_TIP_PAST_TCP] for x in xs for y in (-FINGER_FACE, FINGER_FACE)])
            return rotation.apply(corners)
        # The trailing fingertip is the one behind the coin along the push direction.
        trailing = 1 if rotation.apply([1, 0, 0]) @ np.r_[d, 0] < 0 else -1
        back = tip_corners(trailing)
        lowest = back[:, 2].min()
        inner_edge = back[:2].mean(0)
        target = -np.r_[d, 0]*CRADLE_GAP/2
        return np.r_[target[:2] - inner_edge[:2], -lowest]

    def job_moves(self, side, kind, path):
        moves = []
        for i in range(len(path) - 1):
            moves += self.segment_moves(side, kind, path[i], path[i+1], i == 0, i == len(path) - 2)
        return moves

    def run_jobs(self, jobs):
        """Execute per-arm move lists in lockstep; jobs: {side: [(kind, point, rotation, seconds)]}."""
        jobs = {side: list(moves) for side, moves in jobs.items() if moves}
        while jobs:
            batch = {side: moves.pop(0) for side, moves in jobs.items()}
            seconds = max(m[3] for m in batch.values())
            settle = max(m[4] for m in batch.values())
            transit = {side: m for side, m in batch.items() if m[0] == "transit"}
            linear = {side: m for side, m in batch.items() if m[0] == "move"}
            if transit and linear:
                # Joint and Cartesian interpolation run as separate, simultaneous plans.
                self.mixed_move(transit, linear, seconds, settle)
            elif transit:
                self.transit_many({side: (m[1], m[2], m[5]) for side, m in transit.items()}, seconds, settle)
            else:
                self.move_many({side: (m[1], m[2], m[5]) for side, m in linear.items()}, seconds, settle)
            jobs = {side: moves for side, moves in jobs.items() if moves}

    def mixed_move(self, transit, linear, seconds, settle):
        start = self.desired.copy()
        goals = {side: self.ik(side, m[1], m[2], READY) for side, m in transit.items()}
        starts = {side: (self.planned[side] if side in self.planned else self.tcp(side)) for side in linear}
        rotations = {side: Slerp([0, 1], Rotation.concatenate([starts[side][1], m[2]])) for side, m in linear.items()}
        for fraction in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(fraction)
            for side in goals:
                sl = slice(7*side, 7*side+6)
                self.desired[sl] = start[sl]*(1-t) + goals[side]*t
                self.desired[7*side+6] = start[7*side+6]*(1-t) + transit[side][5]*t
            for side, m in linear.items():
                sl = slice(7*side, 7*side+6)
                position = np.asarray(starts[side][0])*(1-t) + m[1]*t
                self.desired[sl] = self.ik(side, position, rotations[side](t), self.desired[sl], fraction > 0)
            self.step()
        for side, m in {**transit, **linear}.items():
            self.planned[side] = (m[1], m[2])
        self.wait(settle)

    def park(self, sides=(0, 1), seconds=.8):
        """Wait beside the own frame, high and out of the camera's view of the pile."""
        self.transit_many({side: (PARK[side], self.push_rotation(side, (1., 0.), PARK[side]), CRADLE_OPENING)
                           for side in sides}, seconds, .1)

    def look(self, phase):
        from coin_vision import find_coins
        image = self.report(phase)
        free, points, _ = find_coins(image, self.camera)
        return free, points

    def corridor_clear(self, path, kind, obstacles, own):
        """No other coin pixel or placed coin near the swept coin and fingertip region."""
        from build_assets import COINS
        r = COINS[kind]["radius"]
        half = max(r, FINGER_FACE) + CLEARANCE
        reach = CRADLE_GAP/2 + FINGER
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            d = (b - a) / np.linalg.norm(b - a)
            # Both fingertips: the trailing one behind the coin and the leading one ahead of it.
            a, b = a - d*reach, b + d*(reach + self.lead(kind))
            if len(obstacles):
                rel = obstacles - a
                t = np.clip(rel @ d, 0, np.linalg.norm(b - a))
                dist = np.linalg.norm(rel - np.outer(t, d), axis=1)
                if np.any((dist < half) & ~own):
                    return False
        return True

    def reachable(self, side, kind, path):
        """Every pose of the job, including midpoints of straight moves, has an IK solution when
        solved with the same seeds as during execution (transits start from READY)."""
        seed = self.desired[7*side:7*side+6].copy()
        previous = None
        for move in self.job_moves(side, kind, path):
            if move[0] == "transit":
                q, ep, er = robust_solve(side, move[1], move[2], READY)
                if not reach_ok((q, ep, er)):
                    return False
                seed, previous = q, move[1]
                continue
            count = max(1, int(np.ceil(np.linalg.norm(move[1] - previous) / .02)))
            for point in [previous + (move[1] - previous)*k/count for k in range(1, count + 1)]:
                q, ep, er = robust_solve(side, point, move[2], seed)
                if not reach_ok((q, ep, er)) or np.abs(q - seed).max() > .35:
                    return False
                seed = q
            previous = move[1]
        return True

    def options(self, coin, points, placed):
        """The preferred feasible job for one coin, as [(side, path, final, slot)] or [].

        1. Direct push into the next slot of its own frame.
        2. A coin on the wrong side of the pile crosses to the correct side through a clear gap,
           pushed by whichever arm reaches; a later look finds it there with a direct route.
        3. Otherwise a short push out of the crowd, preferably away from the other coins.
        """
        from build_assets import COINS
        kind = coin["kind"]
        owner = OWNER[kind]
        sign = 1. if kind == "penny" else -1.
        c = np.asarray(coin["xy"])
        # The coin's own rim pixels, allowing for the detected centre's error and edge blur.
        own = np.linalg.norm(points - c, axis=1) < COINS[kind]["radius"] + .0035
        extra = [np.asarray(p) for k in placed for p in placed[k]]
        obstacles = np.vstack([points, *[e[None] for e in extra]]) if extra else points
        mask = np.r_[own, np.zeros(len(extra), bool)]
        clear = lambda path: self.corridor_clear(path, kind, obstacles, mask)
        near_side = 0 if c[1] > 0 else 1
        used = [tuple(np.round(p, 4)) for p in placed[kind]]
        free_slots = [q for q in SLOTS[kind] if tuple(np.round(q, 4)) not in used]
        # Far rows first (later pushes then never cross placed coins), nearest column first.
        free_slots.sort(key=lambda q: (-abs(q[1]), abs(q[0] - c[0])))
        for slot in free_slots[:6]:
            waypoint = np.array([np.clip(c[0], .26, .42), sign*.135])
            for path in ([c, slot], [c, waypoint, slot]):
                if np.linalg.norm(path[1] - path[0]) < .015 or not clear(path):
                    continue
                if self.reachable(owner, kind, path):
                    return [(owner, path, True, slot)]
        others = points[~own]
        if c[1]*sign < .04:
            targets = [np.array([x, sign*y]) for y in (.07, .095) for x in np.arange(.22, .33, .02)]
            targets.sort(key=lambda t: np.linalg.norm(t - c))
            for t in targets:
                path = [c, t]
                if not clear(path):
                    continue
                for side in (near_side, 1 - near_side):
                    if self.reachable(side, kind, path):
                        return [(side, path, False, None)]
        nearby = others[np.linalg.norm(others - c, axis=1) < .06]
        # Away from neighbouring coins; a coin with no neighbours is pulled toward the middle of
        # the reachable area on its target side.
        away = c - nearby.mean(0) if len(nearby) else np.array([.30, sign*.06]) - c
        goal = away / (np.linalg.norm(away) + 1e-9)
        headings = []
        for angle in np.radians(np.arange(0, 360, 20)):
            d = np.array([np.cos(angle), np.sin(angle)])
            end = c + .045*d
            if d @ goal < .2 or any(np.linalg.norm(end - h) < .015 or np.linalg.norm(c - h) < .03 for h in self.history):
                continue
            if ROI_SAFE[0] < end[0] < ROI_SAFE[1] and abs(end[1]) < .15:
                headings.append((-(d @ goal) - .5*(d[1]*sign), end))
        for _, end in sorted(headings, key=lambda h: h[0]):
            path = [c, end]
            if clear(path):
                for side in (near_side, 1 - near_side):
                    if self.reachable(side, kind, path):
                        return [(side, path, False, None)]
        return []

    @staticmethod
    def separation(a, b):
        """Smallest distance between points sampled along two coin paths."""
        sample = lambda path: np.vstack([np.linspace(path[i], path[i+1], 12) for i in range(len(path)-1)])
        pa, pb = sample(a), sample(b)
        return float(np.min(np.linalg.norm(pa[:, None] - pb[None], axis=2)))

    def stir(self, points, attempt):
        """No single coin can be moved: stir the largest remaining cluster with a spiral."""
        from scipy.cluster.vq import kmeans2
        groups = [points]
        if len(points) > 60:
            centroids, labels = kmeans2(points, 3, seed=attempt, minit="++")
            groups = [points[labels == k] for k in np.argsort(-np.bincount(labels, minlength=3)) if np.any(labels == k)]
        for group in groups:
            for radius in (.045, .03, .02):
                if self.spread(group, phase=1.3*attempt, radius=radius):
                    return
        raise AssertionError("No reachable stir")

    def feasible(self, side, points, rotation):
        return all(reach_ok(robust_solve(side, p, rotation)) for p in points)

    def spread(self, points, phase=0., radius=None):
        """Stir the pile: the closed fingertips enter at the pile centre and spiral outward just
        above the table, knocking coins off stacks and pushing them apart."""
        sp = SPREAD
        centre = np.median(points, axis=0)
        tcp_z = TABLE + sp["height"] + PAD_TIP_PAST_TCP
        turns = np.linspace(0, 1, max(8, round(sp["seconds"]/.15)))
        r1 = sp["r1"] if radius is None else radius
        path = [centre + (sp["r0"] + (r1-sp["r0"])*f) * np.array([np.cos(2*np.pi*sp["turns"]*f + sp["phase"] + phase),
                                                                  np.sin(2*np.pi*sp["turns"]*f + sp["phase"] + phase)])
                for f in turns]
        poses = [[*path[0], tcp_z + .04]] + [[*p, tcp_z] for p in path]
        preferred = 0 if centre[1] > 0 else 1
        for side in (preferred, 1 - preferred):
            rotation = self.push_rotation(side, (1., 0.), [*path[0], tcp_z + .04])
            lift = [np.array(poses[-1]) + [0, 0, h] for h in (.01, .02, .03, .04)]
            if self.feasible(side, poses + lift, rotation):
                break
        else:
            return False
        self.run_jobs({side: [("transit", np.array(poses[0]), rotation, .6, .1, 0.),
                              ("move", np.array(poses[1]), rotation, .3, .05, 0.)]
                             + [("move", np.array(p), rotation, sp["seconds"]/len(path), 0., 0.) for p in poses[2:]]
                             + [("move", np.array(poses[-1]) + [0, 0, .04], rotation, .25, 0., 0.)]})
        self.park((side,), .7)
        return True

    def sort(self):
        from scene import COINS_PER_KIND
        self.park((0, 1), 1.)
        free, points = self.look("before_spread")
        self.spread(points)
        print("SPREAD DONE", round(self.sim.time, 1), flush=True)
        placed = {"penny": [], "quarter": []}
        stirs = 0
        failed = []          # positions of coins that a push left in place
        while len(placed["penny"]) + len(placed["quarter"]) < 2*COINS_PER_KIND:
            free, points = self.look("sort_look")
            for xy in getattr(self, "last_jobs", []):
                if any(np.linalg.norm(c["xy"] - xy) < .004 for c in free):
                    failed.append(xy)
                    print("NO PROGRESS", np.round(xy, 3).tolist(), flush=True)
            self.last_jobs = []
            candidates = []
            print("LOOK", len(points), "coin pixels;", [(c["kind"][0], np.round(c["xy"], 3).tolist()) for c in free], flush=True)
            for coin in free:
                if len(placed[coin["kind"]]) == COINS_PER_KIND:
                    continue
                if any(np.linalg.norm(coin["xy"] - xy) < .005 for xy in failed):
                    continue
                found = self.options(coin, points, placed)
                for side, path, final, slot in found:
                    length = sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))
                    candidates.append((not final, length, side, coin, path, final, slot))
            if not candidates:
                assert stirs < 30, "Too many stirring attempts"
                stirs += 1
                self.stir(points, stirs)
                continue
            candidates.sort(key=lambda c: (c[0], c[1]))
            best = {}
            for cand in candidates:
                best.setdefault(cand[2], cand)
            ranked = sorted(best.values(), key=lambda c: (c[0], c[1]))
            first = ranked[0]
            jobs = {first[2]: first}
            # Open grippers are wide: run two pushes together only on opposite sides, well apart.
            if len(ranked) == 2 and ranked[1][3] is not first[3] and self.separation(first[4], ranked[1][4]) > .16 \
                    and min(p[1] for p in jobs[first[2]][4]) * max(p[1] for p in ranked[1][4]) <= 0 \
                    and all(p[1]*(1 if first[2] == 0 else -1) > .02 for p in first[4]) \
                    and all(p[1]*(1 if ranked[1][2] == 0 else -1) > .02 for p in ranked[1][4]):
                jobs[ranked[1][2]] = ranked[1]
            moves = {side: self.job_moves(side, job[3]["kind"], job[4]) for side, job in jobs.items()}
            self.run_jobs(moves)
            for side, job in jobs.items():
                kind = job[3]["kind"]
                if job[5]:
                    placed[kind].append(job[6])
                else:
                    self.history.append(np.asarray(job[3]["xy"]))
                self.last_jobs = getattr(self, "last_jobs", []) + [job[3]["xy"]]
                print("PUSHED", round(self.sim.time, 1), SIDES[side], kind, np.round(job[3]["xy"], 3).tolist(),
                      "final" if job[5] else "staged", "segments", len(job[4])-1, flush=True)
            # After a final push the hand is already beside its frame, out of the camera's view of
            # the middle; only hands that pushed in the middle move back out.
            staged = tuple(side for side, job in jobs.items() if not job[5])
            if staged:
                self.park(staged, .6)
        self.wait(1.)
        self.report("sorted")
        print("SORT FINISHED", round(self.sim.time, 1), flush=True)

    def execute(self):
        self.pour()
        self.sort()


def main():
    from simulation import Simulation, archive_source, write_json
    out = HERE/"runs"/("expert-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    archive_source(out)
    print("RUN_DIRECTORY", out, flush=True)
    # The top camera renders at 1920x1080 for coin detection; the rendering resolution does not
    # change the physics, and the same camera pose and calibration are used.
    sim = Simulation(out, review=True)
    expert = Expert(sim)
    error = None
    try:
        expert.execute()
        np.save(out/"actions.npy", np.asarray(sim.commands))
    except Exception:
        error = traceback.format_exc()
        traceback.print_exc()
    finally:
        np.save(out/"actions.npy", np.asarray(sim.commands))
        write_json(out/"phases.json", expert.log)
        sim.save("success" if error is None else "failed", error)
        sim.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    try:
        main()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

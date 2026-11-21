"""Native position targets for a measured-pose, measured-load printmaking demo."""
import json
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from yam_arena import station
from yam_arena.constants import PAD_TIP_PAST_TCP, READY
from yam_arena.ik import smoothstep, solve

from geometry import AXLE_R, BLOCK, CUT, HANDLE, PALETTE, PANEL_SIZE, SHEET_HOME, TAB, THICKNESS
from scene import HZ
from simulation import Simulation, as_array

TABLE = station.TABLE_TOP
DOWN = Rotation.from_matrix([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
BASES = [station.BASES[side] for side in ("left", "right")]
TILT = Rotation.from_euler("y", 4, degrees=True)
TAB_TCP = np.asarray(TAB) + [0, 0, .006]


def solve_pose(side, position, rot, seed, free=False):
    starts = [seed, READY, np.asarray(READY)+[0,0,0,0,0,1.5],
             np.asarray(READY)+[0,0,0,0,0,-1.5]] if free else [seed]
    for start in starts:
        q, ep, er = solve(position, rot.as_quat(), start, BASES[side], damping=.017,
                          iters=110 if free else 30, tol=5e-5)
        if ep < .001 and er < .015:
            return q
    raise RuntimeError(f"IK arm {side} at {position}: {ep:.4f} m, {er:.4f} rad")


class Expert:
    def __init__(self, sim):
        self.sim = sim
        self.q = sim.action.copy()
        self.bias = np.zeros((2, 6))
        self.last = self.q.copy()
        self.feedback = [True, True]
        self.goal = [self.tcp(0), self.tcp(1)]

    def tcp(self, side):
        robot = self.sim.robots[side]
        p = as_array(robot.data.body_link_pose_w)[robot.body_names.index("gripper")]
        r = Rotation.from_quat(p[3:])
        return p[:3] + r.apply([0, 0, .134]), r

    def advance(self):
        action = self.q.copy()
        for side in range(2):
            arm = slice(7*side, 7*side+6)
            action[arm] += self.bias[side]
        self.sim.step(action)
        for side in range(2):
            arm = slice(7*side, 7*side+6)
            if self.feedback[side] and np.max(abs(self.q[arm] - self.last[arm])) < .032:
                actual = as_array(self.sim.robots[side].data.joint_pos)[self.sim.ids[side][0]]
                self.bias[side] = np.clip(self.bias[side] + .13*(self.q[arm]-actual), -.10, .10)
        self.last[:] = self.q

    def wait(self, seconds):
        for _ in range(round(seconds*HZ)):
            self.advance()

    def phase(self, name):
        print("PHASE", json.dumps(self.sim.info), "NEXT", name, flush=True)
        self.sim.phase = name

    def move(self, side, p, r, duration, *, joint_path=False, z_feedback=None):
        p = np.asarray(p)
        arm = slice(7*side, 7*side+6)
        start_q = self.q[arm].copy()
        if joint_path:
            target = solve_pose(side, p, r, start_q, free=True)
            for t in np.linspace(0, 1, max(2, round(duration*HZ))):
                s = smoothstep(t)
                self.q[arm] = (1-s)*start_q + s*target
                self.advance()
        else:
            start_p, start_r = self.goal[side]
            orientation = Slerp([0, 1], Rotation.concatenate([start_r, r]))
            for t in np.linspace(0, 1, max(2, round(duration*HZ))):
                s = smoothstep(t)
                target = (1-s)*start_p + s*p
                if z_feedback is not None:
                    target[2] += z_feedback()
                final = target
                self.q[arm] = solve_pose(side, final, orientation(s), self.q[arm])
                self.advance()
            p = final
        self.goal[side] = (p, r)

    def grip(self, side, opening, seconds=.55):
        index = 7*side+6
        for self.q[index] in np.linspace(self.q[index], opening, round(seconds*HZ)):
            self.advance()
        self.wait(.12)

    def roll(self):
        self.phase("Grasp brayer")
        initial = as_array(self.sim.props[0].data.root_pose_w)
        handle = Rotation.from_quat(initial[3:]).apply(HANDLE) + initial[:3]
        self.move(0, handle+[0,0,.078], DOWN, 1.6, joint_path=True)
        self.move(0, handle, DOWN, .9)
        self.grip(0, 0.)
        actual, rot = self.tcp(0)
        offset = rot.inv().apply(actual-as_array(self.sim.props[0].data.root_pose_w)[:3])
        self.move(0, handle+[0,0,.065], TILT*DOWN, .85)

        def position(x, y, z):
            return np.array([x, y, TABLE+z]) + (TILT*DOWN).apply(offset)

        for label, surface, start, finish, strokes in (
            ("Roll ink on palette", PALETTE, .230, .434, 8),
            ("Roll ink on relief", BLOCK, .270, .365, 6),
        ):
            self.phase(label)
            z = surface[2]+AXLE_R
            y = surface[1]
            clearance = .038 if surface == PALETTE else .016
            for i in range(strokes):
                self.move(0, position(start,y,z+clearance), TILT*DOWN, .85 if i else 1.45)
                self.move(0, position(start,y,z-.0015), TILT*DOWN, .55)
                offset_z = 0.
                averaged = 2.

                def z_feedback():
                    nonlocal offset_z, averaged
                    latest = np.array(self.sim.forces[-12:])[:, 8:19]
                    load = latest[:, 0].mean() if surface == PALETTE else latest[:, 1:].sum(axis=1).mean()
                    averaged = .6*averaged + .4*load
                    offset_z = np.clip(offset_z - .00013*(2.0-averaged), -.005, .003)
                    return offset_z

                self.move(0, position(finish,y,z-.0015), TILT*DOWN, 2.0 if i else 2.5,
                          z_feedback=z_feedback)
                self.move(0, position(finish,y,z+clearance), TILT*DOWN, .55)
                print("STROKE", i, json.dumps(self.sim.info), flush=True)
        self.phase("Set down brayer")
        park = np.array([.225, PALETTE[1], TABLE+PALETTE[2]+AXLE_R]) + DOWN.apply(offset)
        self.move(0, park+[0,0,.055], DOWN, 1.45)
        self.move(0, park, DOWN, .65)
        self.grip(0, 1.)
        self.move(0, park+[0,0,.075], DOWN, .65)
        self.move(0, [.25,.24,TABLE+.15], DOWN, .7, joint_path=True)

    def tab_point(self):
        p = as_array(self.sim.props[1].data.body_link_pose_w)[self.sim.panel_ids[0]]
        return p[:3] + Rotation.from_quat(p[3:]).apply(TAB_TCP)

    def press_sheet(self):
        self.phase("Stabilize the paper at the corner tab")
        self.move(1, [.29,-.234,TABLE+.14], DOWN, 1.2)
        tab = self.tab_point()
        hold = Rotation.from_euler("y", -14, degrees=True) * DOWN
        self.move(0, tab+[0,0,.041], hold, 1.7, joint_path=True)
        self.wait(.35)
        self.move(0, tab, hold, 1.)
        self.wait(.45)
        assert np.linalg.norm(self.tcp(0)[0]-tab) < .006
        self.grip(0, 0., .8)
        self.feedback[0] = False
        held = self.tcp(0)[0]

        self.phase("Press the back while holding the corner")
        rot = Rotation.from_euler("y", 10, degrees=True) * DOWN
        self.move(1, [.350,-.195,TABLE+.137], rot, 1.1)
        count = 1+len(CUT)
        for x, y, ridge, degrees in (
            (.340, -.044, 7, 10),
            (.335, -.037, 9, 10),
            (.328, -.028, 6, 10),
            (.334, -.020, 5, 10),
            (.338, -.049, 8, 10),
            (.301, -.055, 1, 17),
        ):
            rot = Rotation.from_euler("y", degrees, degrees=True) * DOWN
            surface = np.array([x,y,TABLE+BLOCK[2]+THICKNESS])
            point = surface - rot.apply([0,0,PAD_TIP_PAST_TCP])
            above = point+[0,0,.044]
            if ridge == 1:
                self.move(1, [.328,-.091,TABLE+.083], rot, .85)
            self.move(1, above, rot, 1.4 if ridge == 7 else .65)
            self.wait(.18)
            assert np.linalg.norm(self.tcp(1)[0]-above) < .009
            assert np.linalg.norm(self.tcp(0)[0]-held) < .004
            self.move(1, point+[0,0,.003], rot, .52)
            for i in range(22):
                self.move(1, point+[0,0,-.00045*i], rot, .05)
                recent = np.asarray(self.sim.forces[-12:])[:,29+count:].reshape(-1,3,count)
                load = recent[:,:,ridge+1].sum(1).mean()
                if load >= .50:
                    break
            self.wait(.50)
            self.move(1, above, rot, .50)
            print("PRESS", ridge, float(load), json.dumps(self.sim.info), flush=True)

        self.phase("Clear both wrists before regrasping")
        self.move(1, [.328,-.091,TABLE+.083], rot, .82)
        self.move(1, [.325,-.164,TABLE+.104], DOWN, .9)
        self.move(1, [.350,-.195,TABLE+.137], DOWN, .55)
        self.grip(0,1.)
        self.feedback[0] = True
        self.move(0, tab+[0,0,.041], hold, .65)
        self.move(0, [.25,.24,TABLE+.15], DOWN, 1.1, joint_path=True)

    def paper(self):
        self.phase("Lay clean paper on the block")
        tab = self.tab_point()
        self.move(1, tab+[0,0,.080], DOWN, 1.7, joint_path=True)
        self.move(1, tab, DOWN, .8)
        self.grip(1, 0., .8)
        high = tab+[0,0,.060]
        self.move(1, high, DOWN, .75)
        laid = np.array([BLOCK[0], BLOCK[1]-PANEL_SIZE[1], TABLE+BLOCK[2]+THICKNESS/2]) + TAB_TCP
        self.move(1, laid+[0,0,.056], DOWN, 1.55)
        self.move(1, laid, DOWN, 1.25)
        self.grip(1, 1.)
        self.move(1, laid+[0,0,.052], DOWN, .6)
        self.grip(1, 0.)
        self.press_sheet()
        self.phase("Lift the corner and separate the print")
        self.grip(1, 1.)
        tab = self.tab_point()
        self.move(1, tab+[0,0,.046], DOWN, 1.2)
        self.move(1, tab, DOWN, .65)
        self.grip(1, 0., .7)
        self.move(1, tab+[0,0,.025], DOWN, .85)
        self.move(1, tab+[0,0,.085], DOWN, .85)
        self.wait(.5)
        self.phase("Show the pulled print")
        self.move(1, tab+[0,-.093,.115], DOWN, .95)
        show = Rotation.from_euler("x", 80, degrees=True)*DOWN
        self.move(1, tab+[0,-.093,.115], show, 1.25)
        self.wait(1.)


def run_demo(out):
    sim = Simulation(out)
    error = None
    try:
        expert = Expert(sim)
        expert.roll()
        expert.paper()
        assert sim.info["score"] == 1., sim.info
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        sim.save("demo", "success" if error is None else "failed", error)
        sim.close()

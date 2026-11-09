"""Development expert; export the executed 14-D actions for open-loop replay."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from yam_arena import station
from yam_arena.ik import LOWER, UPPER, smoothstep, solve

PAD_Z = .140


class Expert:
    def __init__(self, sim):
        self.sim = sim
        self.desired = sim.action.copy()
        self.bias = np.zeros((2, 6))
        self.gripper_ids = [r.body_names.index("gripper") for r in sim.robots]
        self.low, self.high = np.r_[LOWER, 0., LOWER, 0.], np.r_[UPPER, 1., UPPER, 1.]

    def hand(self, side):
        pose = self.sim.robots[side].data.body_link_pose_w.torch[0, self.gripper_ids[side]].cpu().numpy()
        rotation = Rotation.from_quat(pose[3:])
        return pose[:3]+rotation.apply([0, 0, PAD_Z]), rotation

    def step(self):
        action = self.desired.copy()
        for side in range(2):
            sl = slice(7*side, 7*side+6)
            action[sl] += self.bias[side]
        self.sim.step(np.clip(action, self.low, self.high))
        for side, robot in enumerate(self.sim.robots):
            actual = robot.data.joint_pos.torch[0, :6].cpu().numpy()
            error = self.desired[7*side:7*side+6]-actual
            self.bias[side] = np.clip(self.bias[side]+self.sim.control_dt*error, -.10, .10)

    def wait(self, seconds):
        for _ in range(round(seconds/self.sim.control_dt)):
            self.step()

    def target(self, side, point, rotation):
        sl = slice(7*side, 7*side+6)
        tcp = np.asarray(point)+rotation.apply([0, 0, .134-PAD_Z])
        q, ep, er = solve(tcp, rotation.as_quat(), self.desired[sl],
                         station.BASES[("left", "right")[side]], damping=.015, iters=100, tol=1e-5)
        assert ep < .002 and er < .025, ("Unreachable hand pose", side, point, ep, er)
        self.desired[sl] = q

    def move(self, side, point, rotation, seconds=2., opening=None):
        start, r0 = self.hand(side)
        rotations = Slerp([0, 1], Rotation.concatenate([r0, rotation]))
        grip0 = self.desired[7*side+6]
        for s in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(s)
            self.target(side, start*(1-t)+np.asarray(point)*t, rotations(t))
            if opening is not None:
                self.desired[7*side+6] = grip0*(1-t)+opening/.0939*t
            self.step()
        self.wait(.4)

    def grip(self, side, width, seconds=1.):
        index = 7*side+6
        start = self.desired[index]
        for s in np.linspace(0, 1, max(2, round(seconds/self.sim.control_dt))):
            t = smoothstep(s)
            self.desired[index] = start*(1-t)+width/.0939*t
            self.step()
        self.wait(.4)

    def mark(self, name):
        self.sim.phase = name
        print(name, "t", round(self.sim.time, 2), "hand", self.hand(1)[0].round(4).tolist(), flush=True)

    def home(self, side):
        from yam_arena.constants import READY
        sl = slice(7*side, 7*side+6)
        start = self.desired[sl].copy()
        for s in np.linspace(0, 1, round(2./self.sim.control_dt)):
            t = smoothstep(s)
            self.desired[sl] = start*(1-t)+np.asarray(READY)*t
            self.step()
        self.wait(.5)



def run(out):
    from scene import ROOT
    from simulation import Simulation, archive_source
    from task import INITIAL, cell, solve, TILE_Z
    archive_source(out)
    sim = Simulation(out)
    ex = Expert(sim)
    board = list(INITIAL)
    down = Rotation.from_euler("x", 180, degrees=True)
    try:
        for number in solve():
            side = 0 if number in (5, 4) else 1
            down = Rotation.from_euler("xyz", [180, 0, 180 if side == 0 else 0], degrees=True)
            index, blank = board.index(number), board.index(0)
            source, target = cell(index), cell(blank)
            # Fingers straddle the raised face, with tips inside the recessed rim.
            z = TILE_Z+.019
            start = np.array([*source[:2], z])
            end = np.array([*target[:2], z])
            ex.mark(f"approach_{number}")
            ex.move(side, start+(0, 0, .08), down, 2., opening=.065)
            ex.move(side, start, down, 2.)
            ex.mark(f"grasp_{number}")
            ex.grip(side, .039, 1.)
            ex.mark(f"slide_{number}")
            ex.move(side, end, down, 3.)
            ex.mark(f"release_{number}")
            ex.grip(side, .065, 1.)
            ex.move(side, end+(0, 0, .08), down, 1.5)
            board[blank], board[index] = number, 0
            pose = sim.tiles[number-1].data.root_link_pose_w.torch[0].cpu().numpy()
            print("Tile", number, "pose", pose.round(5).tolist(), "target", target.tolist(), flush=True)
            ex.home(side)
        ex.mark("settled")
        sim.wait(3.)
        result = sim.result()
        print("RESULT", result, flush=True)
        sim.save("success")
        assert result["score"] == 1., "Expert did not solve the puzzle"
        phases = np.asarray(sim.phases)[1::sim.substeps]
        np.savez_compressed(ROOT/"demo_actions.npz", actions=np.asarray(sim.commands), phases=phases)
    except Exception as error:
        sim.save("error", str(error))
        raise
    finally:
        sim.close()

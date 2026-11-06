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

    def ruler(self):
        pose = self.sim.ruler.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        return pose[:3], Rotation.from_quat(pose[3:])

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
        position, rotation = self.ruler()
        print(name, "t", round(self.sim.time, 2), "ruler", position.round(5).tolist(),
              "yaw", round(rotation.as_euler("xyz", degrees=True)[2], 2),
              "hand", self.hand(1)[0].round(4).tolist(), flush=True)

    def home(self, side):
        from yam_arena.constants import READY
        sl = slice(7*side, 7*side+6)
        start = self.desired[sl].copy()
        for s in np.linspace(0, 1, round(2./self.sim.control_dt)):
            t = smoothstep(s)
            self.desired[sl] = start*(1-t)+np.asarray(READY)*t
            self.step()
        self.wait(.5)

    def move_ruler(self, center, rotation, seconds):
        point, r0 = self.ruler()
        hand, hr = self.hand(1)
        offset, relative = r0.inv().apply(hand-point), r0.inv()*hr
        rotations = Slerp([0, 1], Rotation.concatenate([r0, rotation]))
        for s in np.linspace(0, 1, round(seconds/self.sim.control_dt)):
            t = smoothstep(s)
            r = rotations(t)
            p = point*(1-t)+np.asarray(center)*t
            self.target(1, p+r.apply(offset), r*relative)
            self.step()
        self.wait(.6)

    def observe(self, side, target, yaw):
        mount = station.rig("calibrated").wrist_cams[("left", "right")[side]]
        eye = np.asarray(target)+[0, 0, .18]
        rotation = Rotation.from_euler("z", yaw, degrees=True)*Rotation.from_quat(mount.rot_xyzw).inv()
        point = eye+rotation.apply(np.array([0, 0, PAD_Z])-mount.pos)
        self.move(side, point, rotation, 3.)
        self.wait(1.)

    def align_ruler(self, center, rotation):
        # Development feedback only; the delivered policy replays recorded actions.
        for _ in range(round(2./self.sim.control_dt)):
            p, r = self.ruler()
            hand, hr = self.hand(1)
            correction = rotation*r.inv()
            self.target(1, center+correction.apply(hand-p), correction*hr)
            self.step()

    def execute(self):
        from model import BOARD_SIZE, RULER_SIZE, TRIANGLE_BASE, TRIANGLE_HEIGHT
        from scene import BOARD_XY
        down = Rotation.from_euler("z", 90, degrees=True)*Rotation.from_euler("x", 180, degrees=True)
        point, _ = self.ruler()
        self.mark("approach_ruler")
        self.move(1, point+[0, 0, .065], down, 3., opening=.060)
        self.move(1, point+[0, 0, .003], down, 2.)
        self.mark("grasp_ruler")
        self.grip(1, .019)
        p, r = self.hand(1)
        self.move(1, p+[0, 0, .065], r, 2.)
        self.mark("lift_ruler")
        assert self.ruler()[0][2] > station.TABLE_TOP+.055, "Ruler was not lifted"
        z = station.TABLE_TOP+BOARD_SIZE[2]
        a = np.array([BOARD_XY[0]-TRIANGLE_HEIGHT/2, BOARD_XY[1]+TRIANGLE_BASE/2, z])
        b = a+[0, -TRIANGLE_BASE, 0]
        c = b+[TRIANGLE_HEIGHT, 0, 0]
        long_center = a+[-RULER_SIZE[1]/2, -.075, RULER_SIZE[2]/2]
        long_rotation = Rotation.from_euler("z", -90, degrees=True)
        self.mark("align_long_edge")
        self.move_ruler(long_center+[0, 0, .060], long_rotation, 3.)
        self.move_ruler(long_center+[0, 0, .001], long_rotation, 2.)
        self.align_ruler(long_center+[0, 0, .0003], long_rotation)
        self.mark("release_long_edge")
        self.grip(1, .060)
        p, r = self.hand(1)
        self.move(1, p+[0, 0, .060], r, 2.)
        self.home(1)
        self.mark("inspect_long_zero")
        self.observe(0, a, 225)
        self.mark("clear_left_after_long_zero")
        self.home(0)
        self.mark("read_long_edge")
        self.observe(1, b, 0)
        self.mark("regrasp_ruler")
        point, rotation = self.ruler()
        down_long = Rotation.from_euler("x", 180, degrees=True)
        self.move(1, point+[0, 0, .060], down_long, 2., opening=.060)
        self.move(1, point+[0, 0, .003], down_long, 2.)
        self.grip(1, .019)
        p, r = self.hand(1)
        self.move(1, p+[0, 0, .065], r, 2.)
        assert self.ruler()[0][2] > z+.05, "Second grasp failed"
        short_center = b+[.075, -RULER_SIZE[1]/2, RULER_SIZE[2]/2]
        self.mark("align_short_edge")
        self.move_ruler(short_center+[0, 0, .060], Rotation.identity(), 3.)
        self.move_ruler(short_center+[0, 0, .001], Rotation.identity(), 2.)
        self.align_ruler(short_center+[0, 0, .0003], Rotation.identity())
        self.mark("release_short_edge")
        self.grip(1, .060)
        p, r = self.hand(1)
        self.move(1, p+[0, 0, .060], r, 2.)
        self.home(1)
        self.mark("inspect_short_zero")
        self.observe(0, b, 180)
        self.mark("clear_left_after_short_zero")
        self.home(0)
        self.mark("read_short_edge")
        self.observe(1, c, 0)
        self.mark("measurement_complete")


def run(out):
    from simulation import Simulation, archive_source
    from keyboard import press_keys
    archive_source(out)
    sim = Simulation(out)
    try:
        expert = Expert(sim)
        expert.execute()
        press_keys(expert, ("5", "0", "equals"))
        assert sim.instruments.result(sim.time)["score"] == 1., "Expert did not submit the correct area"
        sim.save("success")
        np.savez_compressed(out/"demo_actions.npz", actions=sim.commands,
                            phases=np.asarray(sim.phases)[1::sim.substeps])
    except Exception as error:
        sim.save("error", str(error))
        raise
    finally:
        sim.close()


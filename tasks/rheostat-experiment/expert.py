"""Development-only feedback planner; export targets for independent open-loop replay."""
import numpy as np
from scipy.spatial.transform import Rotation

from circuit import HERE, HZ, voltage


class Expert:
    def __init__(self, experiment):
        from yam_arena import ik, station
        from yam_arena.constants import READY
        self.experiment, self.ik, self.station = experiment, ik, station
        self.q = {s: np.array(READY, float) for s in ("left","right")}
        self.grip = dict(left=1., right=1.)
        self.targets = {s: ik.tcp_pose(q, station.BASES[s]) for s,q in self.q.items()}
        self.press = False
        self.press_z = station.TABLE_TOP + .046
        self.phase = "ready"

    def tcp(self, side):
        from yam_arena.constants import TCP_OFFSET
        robot = self.experiment.base.scene[side+"_robot"]
        i = robot.find_bodies("gripper")[0][0]
        pose = robot.data.body_link_pose_w.torch[0,i].cpu().numpy()
        rot = Rotation.from_quat(pose[3:]).as_matrix()
        return pose[:3] + rot @ TCP_OFFSET, rot

    def solve(self, side, pos, rot):
        q, pe, re = self.ik.solve(pos,self.ik.matrix_to_quat_xyzw(rot), self.q[side],
            self.station.BASES[side], iters=300, damping=.02, tol=2e-5)
        assert pe < .002 and re < .04, (self.phase,side,pe,re)
        self.q[side], self.targets[side] = q,(np.array(pos),np.array(rot))

    def tick(self):
        if self.press:
            force = np.asarray(self.experiment.measures[-1]["finger_force_N"])[0,:2].sum()
            self.press_z = np.clip(self.press_z + .002*(force-5.)/HZ,
                                   self.station.TABLE_TOP+.015, self.station.TABLE_TOP+.05)
            p,r = self.targets["left"]
            actual,_ = self.tcp("left")
            p = p.copy()
            p[:2] += np.clip(.2*(np.array([.52,.185])-actual[:2]),-.0005,.0005)
            p[2] = self.press_z
            self.solve("left",p,r)
        self.experiment.step(np.r_[self.q["left"],self.grip["left"],self.q["right"],self.grip["right"]])

    def status(self):
        print(self.phase, self.experiment.score.result(), "forces", np.round(
            self.experiment.measures[-1]["finger_force_N"],2).tolist(),
            "tcp", {s:np.round(self.tcp(s)[0],4).tolist() for s in self.q}, flush=True)

    def hold(self, seconds, phase):
        self.phase = phase
        for _ in range(round(seconds*HZ)):
            self.tick()
        self.status()

    def transit(self, side, pos, rot, seconds, phase, grip=1.):
        self.phase = phase
        q0, g0 = self.q[side].copy(),self.grip[side]
        self.solve(side,pos,rot)
        q1 = self.q[side].copy()
        for i in range(round(seconds*HZ)):
            s = self.ik.smoothstep((i+1)/round(seconds*HZ))
            self.q[side] = q0+s*(q1-q0)
            self.grip[side] = g0+s*(grip-g0)
            self.tick()
        self.status()

    def move(self, side, pos, seconds, phase, grip=None):
        self.phase = phase
        p0,r = self.targets[side]
        p0 = p0.copy()
        g0,g1 = self.grip[side], self.grip[side] if grip is None else grip
        for i in range(round(seconds*HZ)):
            s = self.ik.smoothstep((i+1)/round(seconds*HZ))
            self.solve(side,p0+s*(np.array(pos)-p0),r)
            self.grip[side] = g0+s*(g1-g0)
            self.tick()
        self.status()


def execute(experiment):
    expert = Expert(experiment)
    z = expert.station.TABLE_TOP
    down = Rotation.from_euler("y",-40,degrees=True).as_matrix() @ np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]])
    expert.hold(1.,"initial")
    expert.transit("left",[.52,.185,z+.16],down,3.,"left approach",grip=.3)
    expert.move("left",[.52,.185,z+.044],3.,"left contact")
    expert.press = True
    expert.hold(4.,"left steady")
    expert.transit("right",[.505,0,z+.16],down,3.,"right approach",grip=.7)
    expert.move("right",[.505,0,z+.112],3.,"right descend")
    expert.move("right",[.505,0,z+.112],2.,"grasp slider",grip=.15)
    expert.hold(1.,"grasp settle")
    expert.move("right",[.505,-.10,z+.112],6.,"slide right")
    expert.phase = "adjust voltage"
    for _ in range(12*HZ):
        volts = voltage(experiment.displacement())
        p,r = expert.targets["right"]
        p = p.copy()
        p[1] -= np.clip((5.-volts)*.00012, -.0004,.0004)
        expert.solve("right",p,r)
        expert.tick()
        if experiment.score.result()["held_s"] >= 2. and abs(volts-5.) < .02:
            break
    expert.status()
    expert.press = False
    expert.hold(3.,"settle hands")
    expert.phase = "fine adjustment"
    for _ in range(10*HZ):
        p,r = expert.targets["right"]
        p = p.copy()
        p[1] -= (5.-voltage(experiment.displacement()))*.00012
        expert.solve("right",p,r)
        expert.tick()
    expert.hold(3.,"final hold")
    assert experiment.score.result()["success"], experiment.score.result()
    assert np.asarray([m["finger_force_N"] for m in experiment.measures[-3*HZ:]])[:,0,:2].mean() > .1
    np.save(HERE/"demo_actions.npy", np.asarray(experiment.actions))

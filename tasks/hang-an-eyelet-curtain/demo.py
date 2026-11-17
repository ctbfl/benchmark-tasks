"""Scripted demo. Its plan reads the simulator's ring and rod poses (and the arms' joint angles) at fixed
points: before the right hand takes the first ring, and before and after it grasps the second ring.
Every action goes through the same native 14-D joint-position/gripper interface as the Dream-1 run."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from config import ROD_HALF
from runtime import write_json
import demo_plan


def run(e):
    pending = [0]
    senses = []
    holder = {}

    def sync():
        rows = holder['s'].rows
        for action in rows[pending[0]:]:
            e.step(action)
        pending[0] = len(rows)

    def sense():
        sync()
        rings = e.ring_poses()
        r0, r1 = rings[0], rings[1]
        m0 = Rotation.from_quat(r0[3:]).as_matrix()
        rod = e.pose(e.rod)
        u = Rotation.from_quat(rod[3:]).as_matrix()[:, 1]
        m = dict(c0=r0[:3].copy(), n0=m0[:, 2], t0=m0[:, 0],
                 c1=r1[:3].copy(), n1=Rotation.from_quat(r1[3:]).as_matrix()[:, 2],
                 qr=e.robots[1].data.joint_pos.torch[0, :6].cpu().numpy().astype(float),
                 tip=rod[:3]-u*ROD_HALF, u=u)
        senses.append(dict(time_s=e.steps/20, **{k: np.asarray(v).tolist() for k, v in m.items()}))
        return m

    original = demo_plan.Seq.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        holder['s'] = self
    demo_plan.Seq.__init__ = init
    try:
        s = demo_plan.build(sense=sense)
        sync()
    finally:
        demo_plan.Seq.__init__ = original
        if 's' in holder:
            write_json(e.out/'demo-plan.json', dict(segments=holder['s'].marks, measurements=senses))

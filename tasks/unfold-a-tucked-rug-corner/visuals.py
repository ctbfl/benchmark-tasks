"""Synchronize display-only meshes and cameras with measured native joint states."""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
from yam_arena import YAM_URDF, station
from yam_arena.robot import asset_link_paths
from yam_arena.urdf import forward_kinematics


class Visuals:
    def __init__(self, stage, robots, cameras):
        self.stage, self.robots = stage, robots
        self.items = []
        cache = UsdGeom.XformCache()
        for side in ("left", "right"):
            for name, path in asset_link_paths().items():
                link = stage.GetPrimAtPath("/World/envs/env_0/" + side.title() + "Arm" + path)
                candidates = [link.GetChild(n) for n in (name, "dream_black", "dream_white")]
                if cameras and name == "gripper":
                    candidates.append(link.GetChild(side + "_wrist_cam"))
                for prim in candidates:
                    if not prim:
                        continue
                    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(UsdPhysics.CollisionAPI)
                    offset = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(link).GetInverse()
                    xf = UsdGeom.Xformable(prim)
                    xf.ClearXformOpOrder()
                    xf.SetResetXformStack(True)
                    self.items.append((side, name, offset, xf.AddTransformOp(opSuffix="measured")))

    def update(self):
        poses = {}
        for side, robot in zip(("left", "right"), self.robots, strict=True):
            joints = dict(zip(robot.joint_names, robot.data.joint_pos.torch[0].cpu().numpy(), strict=True))
            for name, matrix in forward_kinematics(YAM_URDF, joints).items():
                matrix = matrix.copy()
                matrix[:3, 3] += station.BASES[side]
                poses[side, name] = Gf.Matrix4d(*matrix.T.ravel())
        for side, name, offset, op in self.items:
            op.Set(offset * poses[side, name])

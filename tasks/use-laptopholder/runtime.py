"""Isaac Sim scene and live PhysX readback, imported after AppLauncher."""

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import get_current_stage_id
import omni.physics.tensors as tensors

import model

PHYSICS_HZ = 960


def simulation(dt=1/PHYSICS_HZ):
    (model.ROOT / "outputs/validation").mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cpu", enable_scene_query_support=True,
        log_dir=str(model.ROOT / "outputs/validation")))
    stage = sim.stage
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    ps = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(sim.cfg.physics_prim_path))
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateEnableStabilizationAttr(False)
    ps.CreateEnableGPUDynamicsAttr(False)
    ground = model.material(stage, "/World/materials/table", (.32, .23, .15), (.65, .5))
    model.shape(stage, "/World/Table", model.block((5, 5, .04), (0, 0, -.02)), ground)
    light = UsdLux.DomeLight.Define(stage, "/World/Dome")
    light.CreateIntensityAttr(700)
    sun = UsdLux.DistantLight.Define(stage, "/World/Key")
    sun.CreateIntensityAttr(2500)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-30, -35, -15))
    return sim


class Mechanism:
    def __init__(self, sim, path):
        self.sim = sim
        self.path = path
        self.view = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.bodies = self.view.create_rigid_body_view(path + "/links/*")
        self.names = [p.rsplit("/", 1)[-1] for p in self.bodies.prim_paths]
        assert set(self.names) == set(model.NAMES), self.names
        self.index = {n: i for i, n in enumerate(self.names)}
        self.ids = np.arange(len(self.names), dtype=np.uint32)
        self.local_points = {}
        self.mass = {}
        self.com = {}
        for name in self.names:
            prim = sim.stage.GetPrimAtPath(path + "/links/" + name)
            meshes = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
            self.local_points[name] = np.concatenate([np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in meshes])
            ma = UsdPhysics.MassAPI(prim)
            self.mass[name] = ma.GetMassAttr().Get()
            self.com[name] = np.asarray(ma.GetCenterOfMassAttr().Get())

    def state(self):
        t = self.bodies.get_transforms().copy()
        v = self.bodies.get_velocities().copy()
        assert np.isfinite(t).all() and np.isfinite(v).all()
        r = Rotation.from_quat(t[:, 3:]).as_matrix()
        return {n: (t[i, :3], r[i], v[i]) for i, n in enumerate(self.names)}

    def wrench(self, forces=None, torques=None):
        f, t = np.zeros((len(self.names), 3), np.float32), np.zeros((len(self.names), 3), np.float32)
        for name, val in (forces or {}).items():
            f[self.index[name]] = val
        for name, val in (torques or {}).items():
            t[self.index[name]] = val
        self.bodies.apply_forces_and_torques_at_position(f, t, None, self.ids, True)

    def measure(self):
        state = self.state()
        result = {"sides": {}}
        for side in ("L", "R"):
            bp, br, _ = state["base_"+side]
            dp, dr, _ = state["deck_"+side]
            pp, pr, _ = state["prop_"+side]
            toe = br.T @ (pp + pr @ np.array([model.PROP_LENGTH, 0, 0]) - bp)
            gear = int(np.argmin(abs(model.SLOTS-toe[0]))) + 1
            error = toe - [model.SLOTS[gear-1], 0, model.SEAT_Z]
            deck_axis = br.T @ dr[:, 0]
            result["sides"][side] = {"gear": gear, "toe_local": toe.tolist(), "seat_error": error.tolist(),
                "tilt_deg": float(np.rad2deg(np.arctan2(deck_axis[2],deck_axis[0]))), "base_z": float(bp[2]),
                "base_tilt_deg": float(np.rad2deg(np.arccos(np.clip(br[2,2],-1,1))))}
        result["width"] = float(np.linalg.norm(state["deck_L"][0]-state["deck_R"][0]) + model.RAIL_WIDTH)
        result["max_speed"] = float(max(np.linalg.norm(v[:3]) for _, _, v in state.values()))
        result["max_spin"] = float(max(np.linalg.norm(v[3:]) for _, _, v in state.values()))
        points = np.concatenate([p + self.local_points[n] @ r.T for n, (p, r, _) in state.items()])
        result["low"] = points.min(axis=0).tolist()
        result["high"] = points.max(axis=0).tolist()
        mass = sum(self.mass.values())
        result["com"] = (sum(self.mass[n] * (p + r @ self.com[n]) for n, (p,r,_) in state.items()) / mass).tolist()
        result["com_velocity"] = (sum(self.mass[n] * v[:3]
                                      for n, (_,r,v) in state.items()) / mass).tolist()
        errors = []
        for prim in self.sim.stage.GetPrimAtPath(self.path+"/joints").GetChildren():
            joint = UsdPhysics.Joint(prim)
            a = str(joint.GetBody0Rel().GetTargets()[0]).rsplit("/",1)[-1]
            b = str(joint.GetBody1Rel().GetTargets()[0]).rsplit("/",1)[-1]
            p0,r0,_ = state[a]
            p1,r1,_ = state[b]
            d = r0.T @ (p1 + r1 @ np.asarray(joint.GetLocalPos1Attr().Get()) - p0) - joint.GetLocalPos0Attr().Get()
            if prim.IsA(UsdPhysics.PrismaticJoint):
                d[0] = 0
            errors.append(np.linalg.norm(d))
        result["joint_error"] = float(max(errors))
        return result


def step(sim, seconds, action=None):
    for i in range(round(seconds / sim.cfg.dt)):
        if action is not None:
            action(i * sim.cfg.dt)
        sim.step(render=False)

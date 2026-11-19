"""Common Isaac Lab environment and actual state recording for both policies."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from honey import Honey
from layout import DIPPER, JAR, JAR_RISER, TABLE_TOP, TOAST

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "dream-defaults.json").read_text())
SIM = CONFIG["simulation"]
HZ = SIM["control_hz"]
DT = 1/SIM["physics_hz"]
SUBSTEPS = SIM["physics_hz"] // HZ
MAX_STEPS = CONFIG["episode"]["max_control_steps"]
CAMERAS = CONFIG["observation"]["camera_names"]


def make_env(cameras, out):
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
    from yam_arena import station
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback, task_room

    rig = station.rig(CONFIG["station"]["top_camera"])
    robot = BimanualYamEmbodiment(enable_cameras=cameras, rig=rig, contact_sensors=False,
                                  robot_contact_offset=.0007)
    if cameras:
        for name in CAMERAS:
            cfg = getattr(robot.camera_config, name)
            lens, cam = cfg.spawn.distortion, cfg.spawn
            assert [cfg.width, cfg.height] == CONFIG["observation"]["render_size_wh"]
            cam.horizontal_aperture = cfg.width * cam.focal_length / lens.fx
            cam.vertical_aperture = cfg.height * cam.focal_length / lens.fy
            cam.horizontal_aperture_offset = (cfg.width/2-lens.cx) * cam.focal_length/lens.fx
            cam.vertical_aperture_offset = (lens.cy-cfg.height/2) * cam.focal_length/lens.fy
            ren = CONFIG["renderer"]
            cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=ren["ambient_light_intensity"], carb_settings={
                    "/rtx/rendermode": ren["mode"], "/rtx/post/aa/op": ren["antialiasing_op"],
                }))

    def object_at(name, pos, dynamic=False):
        return Object(name=name, object_type=ObjectType.RIGID if dynamic else ObjectType.BASE,
                      usd_path=str(HERE/"assets"/f"{name}.usda"), initial_pose=Pose(pos, (0,0,0,1)))

    def configure(cfg):
        cfg = env_cfg_callback(HZ)(cfg)
        assert SIM["backend"] == "physx" and 1/DT == HZ*SUBSTEPS
        cfg.sim.dt = DT
        cfg.decimation = cfg.sim.render_interval = SUBSTEPS
        cfg.sim.device = SIM["device"]
        cfg.sim.gravity = tuple(SIM["gravity_m_s2"])
        cfg.sim.use_fabric = SIM["use_fabric"]
        cfg.sim.physics.solver_type = SIM["solver_type"]
        cfg.sim.log_dir = str(out)
        cfg.log_dir = str(out)
        return cfg

    scene = Scene([*task_room("station", rig.top_cam),
                   object_at("pot_base", (*JAR, TABLE_TOP)),
                   object_at("honey_pot", (*JAR, TABLE_TOP+JAR_RISER)),
                   object_at("plate", (*TOAST, TABLE_TOP)),
                   object_at("toast", (*TOAST, TABLE_TOP)),
                   object_at("dipper", DIPPER, True)])
    spec = IsaacLabArenaEnvironment(name="twirl_honey_dipper", embodiment=robot,
        scene=scene, env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=SIM["num_envs"],
        solve_relations=False, disable_fabric=not SIM["use_fabric"],
        device=SIM["device"])).make_registered()


class World:
    def __init__(self, out, cameras=False):
        import torch
        from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
        from yam_arena.ik import LOWER, UPPER
        from yam_arena.inspect_embodiment import ARMS

        self.out = out
        self.torch = torch
        self.env = make_env(cameras, out)
        self.base = self.env.unwrapped
        self.scene = self.base.scene
        self.dipper = self.scene["dipper"]
        self.robots = [self.scene[name] for name in ARMS]
        self.joint_ids = [r.find_joints(ARM_JOINTS+FINGER_JOINTS, preserve_order=True)[0]
                          for r in self.robots]
        self.low = np.tile(np.r_[LOWER, 0.], 2)
        self.high = np.tile(np.r_[UPPER, 1.], 2)
        assert np.isclose(self.base.physics_dt, DT) and self.base.cfg.decimation == SUBSTEPS

    def state(self):
        from yam_arena.constants import FINGER_OPEN
        q = self.robot_joints().reshape(2, 8)
        return np.c_[q[:, :6], np.clip(q[:, 6:].mean(axis=1)/FINGER_OPEN, 0., 1.)].ravel()

    def robot_joints(self):
        return np.concatenate([r.data.joint_pos.torch[0, ids].cpu().numpy()
                               for r, ids in zip(self.robots, self.joint_ids)])

    def tool(self):
        return (self.dipper.data.root_link_pose_w.torch[0].cpu().numpy().copy(),
                self.dipper.data.root_link_vel_w.torch[0].cpu().numpy().copy())

    def reset(self, seed):
        from scoring import Scoring
        from yam_arena.constants import READY
        self.env.reset(seed=seed)
        self.honey = None
        ready = self.torch.tensor([[*READY, 1., *READY, 1.]], dtype=self.torch.float32)
        for _ in range(HZ):
            self.env.step(ready)
        self.honey = Honey()
        self.scorer = Scoring()
        self.tick = self.steps = 0
        self.actions, self.trajectory = [], []
        self.previous_pose = self.tool()[0]
        axis = Rotation.from_quat(self.previous_pose[3:]).as_matrix()[:, 0]
        self.honey.pose(self.previous_pose[:3], axis)
        self.record(self.previous_pose, np.zeros(6))

    def record(self, pose, speed):
        row = np.r_[self.tick*DT, self.robot_joints(), pose, speed,
                    self.honey.rot, self.honey.retained_g]
        self.trajectory.append(row)
        self.scorer.advance(row)

    def step(self, action):
        assert self.steps < MAX_STEPS
        action = np.asarray(action, dtype=float)
        assert action.shape == (14,) and np.isfinite(action).all()
        target = np.clip(action, self.low, self.high)
        self.actions.append(target)
        self.base.action_manager.process_action(self.torch.tensor(target[None], dtype=self.torch.float32))
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.scene.update(DT)
            self.tick += 1
            pose, _ = self.tool()
            curr = Rotation.from_quat(pose[3:])
            prev = Rotation.from_quat(self.previous_pose[3:])
            speed = np.r_[(pose[:3].astype(float)-self.previous_pose[:3])/DT,
                          (curr*prev.inv()).as_rotvec()/DT]
            axis = curr.as_matrix()[:, 0]
            self.honey.update(DT, pose[:3], axis, speed[:3], speed[3:])
            self.record(pose, speed)
            self.previous_pose = pose
        self.steps += 1

    def grade(self):
        from dataclasses import asdict
        return self.scorer.evaluate([asdict(d) for d in self.honey.landings])

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in CAMERAS:
            sensor = self.scene[name]
            sensor.update(1/HZ, force_recompute=True)
            images[name] = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
        return images

    def save(self):
        np.save(self.out/"trajectory.npy", self.trajectory)
        np.save(self.out/"actions.npy", self.actions)
        m = self.honey.measurements()
        m["control_steps"] = self.steps
        (self.out/"measurements.json").write_text(json.dumps(m, indent=2, allow_nan=False)+"\n")
        return m

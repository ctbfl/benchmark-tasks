"""One calibrated YAM arena used by the policy, expert and trajectory renderer."""

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
from inspect_robots import StepResult
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from liquid import CAMERAS, CONFIG, DT, Drain, HZ, INSTRUCTION

ENV_ROOT = "/World/envs/env_0"
ROOT = ENV_ROOT + "/separator"
SIM = CONFIG["simulation"]
SUBSTEPS = SIM["physics_hz"] // HZ
READY_ACTION = np.array([*READY, 1., *READY, 1.])


def scene_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("scene.py", "geometry.py", "liquid.py", "dream-defaults.json")}


def camera_calibration(cfg):
    from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

    camera = cfg.spawn
    lens = camera.distortion
    camera.horizontal_aperture = cfg.width * camera.focal_length / lens.fx
    camera.vertical_aperture = cfg.height * camera.focal_length / lens.fy
    camera.horizontal_aperture_offset = (cfg.width/2-lens.cx) * camera.focal_length / lens.fx
    camera.vertical_aperture_offset = (lens.cy-cfg.height/2) * camera.focal_length / lens.fy
    settings = CONFIG["renderer"]
    cfg.renderer_cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
        ambient_light_intensity=settings["ambient_light_intensity"], enable_dlssg=False,
        carb_settings={
            "/rtx/rendermode": settings["mode"],
            "/rtx/post/aa/op": settings["antialiasing_op"],
            "/rtx-transient/post/aa/limitedOps": False,
        }))
    return cfg


class GravyStation(YamArenaEmbodiment):
    def __init__(self, out, seed, *, record=True, still_camera=False):
        width, height = CONFIG["observation"]["render_size_wh"]
        super().__init__(device=SIM["device"], cam_width=width, cam_height=height,
                         control_hz=HZ, object_jitter=CONFIG["episode"]["object_jitter_m"],
                         scene=CONFIG["station"]["scene"], top_cam=CONFIG["station"]["top_camera"])
        self.info = replace(self.info, name="yam_bottom_separator", supported_target_kinds=frozenset(),
            environment_id="yam-bottom-release-fat-separator",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out = out
        self.seed = seed
        self.record = record
        self.still_camera = still_camera
        self.app = None
        self.drain = Drain(seed)
        self.peak_stroke = self.peak_contact = 0.

    def _ensure_sim(self, goal=None):
        if self._env is not None:
            return
        from isaaclab.app import AppLauncher

        assert SIM["backend"] == "physx" and SIM["physics_hz"] % HZ == 0
        self.app = AppLauncher(headless=True, enable_cameras=True,
            kit_args="--/carb/tasking/plugin/threadCount=12 "
                     "--/omni/tbb.globalcontrol/maxThreadCount=12 "
                     "--/persistent/rtx/modes/rt/enabled=false "
                     "--/persistent/rtx/modes/rt2/enabled=true").app
        import carb.settings
        import omni.physics.tensors as tensors
        import omni.usd
        import torch
        from pxr import Gf, UsdLux
        from isaaclab.sim.utils import get_current_stage_id
        from isaaclab.sensors import CameraCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose
        from yam_arena import station
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_room

        from geometry import LiquidVisual, XY, build_asset

        robot = BimanualYamEmbodiment(enable_cameras=True, cam_width=self.cam_width,
                                     cam_height=self.cam_height, rig=self.rig, contact_sensors=False)
        camera_names = CAMERAS
        if self.still_camera:
            top = robot._camera_cfg(1080, 1920, overview=False).top_cam
            top.prim_path = "{ENV_REGEX_NS}/TopCam1080"
            robot.camera_config = make_configclass("StillCameraCfg", [("still", CameraCfg, top)],
                                                   bases=(type(robot.camera_config),))()
            camera_names = [*CAMERAS, "still"]
        for name in camera_names:
            camera_calibration(getattr(robot.camera_config, name))
        asset_path = self.out / "separator.usda"
        build_asset(asset_path)
        fixture = Object(name="separator", object_type=ObjectType.BASE, usd_path=str(asset_path),
                         initial_pose=Pose(position_xyz=(*XY, station.TABLE_TOP), rotation_xyzw=(0,0,0,1)))

        def configure(cfg):
            cfg = env_cfg_callback(HZ)(cfg)
            cfg.sim.dt = DT
            cfg.decimation = cfg.sim.render_interval = SUBSTEPS
            cfg.sim.device = SIM["device"]
            cfg.sim.gravity = tuple(SIM["gravity_m_s2"])
            cfg.sim.use_fabric = SIM["use_fabric"]
            cfg.sim.physics.solver_type = SIM["solver_type"]
            cfg.sim.log_dir = cfg.log_dir = str(self.out / "isaac")
            return cfg

        environment = IsaacLabArenaEnvironment(name="bottom_release_separator", embodiment=robot,
            scene=Scene([*task_room(self.scene, self.rig.top_cam), fixture]), env_cfg_callback=configure)
        cfg = ArenaEnvBuilderCfg(num_envs=SIM["num_envs"], solve_relations=False, device=SIM["device"])
        self._env = ArenaEnvBuilder(environment, cfg).make_registered()
        self._base = self._env.unwrapped
        self._torch = torch
        self._ready = torch.as_tensor(READY_ACTION[None], dtype=torch.float32)
        self._ids = {name: (
            self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        self.stage = omni.usd.get_context().get_stage()
        for path, kind, intensity in (("/World/Light", UsdLux.DomeLight, 1050.),
                                       (ENV_ROOT+"/lamp", UsdLux.SphereLight, 14500.)):
            light = kind(self.stage.GetPrimAtPath(path))
            light.GetColorAttr().Set(Gf.Vec3f(1.))
            light.GetIntensityAttr().Set(intensity)
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.paddle = self.physics.create_rigid_body_view(ROOT + "/Paddle")
        assert self.paddle.count == 1
        self.contacts = [self.physics.create_rigid_contact_view(
            ENV_ROOT + "/" + side + "Arm" + robot.link_paths["gripper"] + "/tip_*",
            [ROOT+"/Paddle"]) for side in ("Left", "Right")]
        self.visual = LiquidVisual(self.stage, ROOT)

    def reset(self, scene_spec, *, seed=None):
        self._ensure_sim()
        self._instruction = scene_spec.instruction
        self._env.reset(seed=seed)
        for _ in range(HZ // 2):
            self._env.step(self._ready)
        self.drain = Drain(self.seed)
        self._t = self.peak_stroke = self.peak_contact = 0.
        self.initial_paddle = self.paddle.get_transforms().copy()
        from yam_arena import station
        from geometry import PADDLE, XY
        np.testing.assert_allclose(self.initial_paddle[0,:3],
            [XY[0]+PADDLE[0],XY[1],station.TABLE_TOP+PADDLE[2]], atol=.001)
        if self.record:
            self.states = (self.out / "physics-f64.bin").open("xb")
            self.actions = (self.out / "actions-f64.bin").open("xb")
            self.save_state(0.)
        for _ in range(4):
            obs = self.capture()
        for frame in obs.images.values():
            assert frame.shape == (self.cam_height, self.cam_width, 3) and frame.std() > 8
        return obs

    def save_state(self, stroke):
        row = [self.drain.ticks * DT, stroke]
        for arm in ARMS:
            data = self._base.scene[arm].data
            row.extend(data.joint_pos.torch[0].cpu().numpy())
            row.extend(data.joint_vel.torch[0].cpu().numpy())
        values = np.asarray(row, dtype="<f8")
        assert np.isfinite(values).all()
        self.states.write(values.tobytes())

    def capture(self):
        self.visual.update(self.drain)
        self.render()
        for name in CAMERAS:
            self._base.scene[name].update(1 / HZ, force_recompute=True)
        return self._observation()

    def render(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        self._base.sim.render()
        ensure_isaac_rtx_render_update(force=True)

    def advance(self, action):
        request = np.asarray(action, dtype=np.float64)
        assert request.shape == (14,) and np.isfinite(request).all()
        target = np.clip(request, self._low, self._high)
        if self.record:
            self.actions.write(np.asarray([request,target], dtype="<f8").tobytes())
        base = self._base
        base.action_manager.process_action(self._torch.as_tensor(target[None], dtype=self._torch.float32))
        for _ in range(SUBSTEPS):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            stroke = float(self.initial_paddle[0,2] - self.paddle.get_transforms()[0,2])
            self.drain.step(stroke)
            if self.record:
                self.save_state(stroke)
            self.peak_stroke = max(self.peak_stroke, stroke)
            for view in self.contacts:
                forces = view.get_contact_force_matrix(dt=DT)
                self.peak_contact = max(self.peak_contact, float(np.linalg.norm(forces, axis=-1).max()))
        self._t = self.drain.ticks * DT
        if self.record:
            self.states.flush()
            self.actions.flush()

    def step(self, action):
        self.advance(action.data)
        info = self.metrics()
        solved = info["score"] == 1.
        return StepResult(observation=self.capture(), terminated=solved,
                          termination_reason="success" if solved else None, info=info)

    def metrics(self):
        return dict(**self.drain.metrics(), peak_stroke_m=self.peak_stroke,
                    peak_pad_contact_n=self.peak_contact)

    def close(self):
        if hasattr(self, "states"):
            self.states.close()
            self.actions.close()
        super().close()

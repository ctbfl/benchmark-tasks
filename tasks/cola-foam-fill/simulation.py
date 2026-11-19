"""Shared YAM station, physical button, process and evidence recording."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from inspect_robots import StepResult
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, YamArenaEmbodiment

from process import AREA, CONDITIONS, JET_OFFSETS, OPEN_AT, OUTLET_ABOVE_FLOOR, Pour
from recording import FPS, Recording

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "dream-defaults.json").read_text())
PHYS = CFG["simulation"]
DT = 1 / PHYS["physics_hz"]
HZ = PHYS["control_hz"]
SUBSTEPS = PHYS["physics_hz"] // HZ
LIMIT = CFG["episode"]["max_control_steps"]
CAMERAS = tuple(CFG["observation"]["camera_names"])
INSTRUCTION = (
    "Fill the clear cup with cola to at least the blue line once the foam settles. "
    "Hold the red button to pour and release it to stop. Watch the foam and let it fall "
    "before topping up. Use at most three pours, never spill liquid or foam over the rim, "
    "and finish with the button released."
)
ROOT = "/World/envs/env_0/fountain"


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


class Fountain(YamArenaEmbodiment):
    def __init__(self, out, *, film=True, still=False):
        super().__init__(device=PHYS["device"], cam_width=CFG["observation"]["render_size_wh"][0],
                         cam_height=CFG["observation"]["render_size_wh"][1],
                         control_hz=HZ, scene=CFG["station"]["scene"],
                         top_cam=CFG["station"]["top_camera"], object_jitter=0)
        self.info = replace(self.info, name="yam_cola_fountain",
                            environment_id="yam-cola-foam-fill",
                            supported_target_kinds=frozenset(), docs=INSTRUCTION)
        self.out = out
        self.film = film
        self.still = still
        self.movie = None
        self.frames = 0
        self.crest_max = 0.
        self.trace = self.actions = self.poses = None

    def _ensure_sim(self, goal=None):
        import omni.physics.tensors as tensors
        import omni.physx
        import torch
        from isaaclab.sim.utils import get_current_stage_id
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        from pxr import Gf, UsdLux
        from yam_arena import station
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_room

        from geometry import CUP, DrinkVisual, build_asset

        if self._env is not None:
            return
        asset = self.out / "fountain.usda"
        build_asset(asset)
        robot = BimanualYamEmbodiment(enable_cameras=True, cam_width=self.cam_width,
                                      cam_height=self.cam_height, contact_sensors=False, rig=self.rig)
        render = CFG["renderer"]
        renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=render["ambient_light_intensity"],
            max_bounces=5, enable_dlssg=False, carb_settings={
                "/rtx/rendermode": render["mode"], "/rtx/post/aa/op": render["antialiasing_op"],
            }))
        for name in CAMERAS:
            camera = getattr(robot.camera_config, name)
            # Camera distortion and USD intrinsics retain the same calibrated frustum.
            lens, cfg = camera.spawn.distortion, camera.spawn
            cfg.horizontal_aperture = camera.width * cfg.focal_length / lens.fx
            cfg.vertical_aperture = camera.height * cfg.focal_length / lens.fy
            cfg.horizontal_aperture_offset = (camera.width/2-lens.cx) * cfg.focal_length/lens.fx
            cfg.vertical_aperture_offset = (lens.cy-camera.height/2) * cfg.focal_length/lens.fy
            camera.renderer_cfg = renderer

        fixture = Object(name="fountain", object_type=ObjectType.BASE, usd_path=str(asset),
                         initial_pose=Pose(position_xyz=(0,0,station.TABLE_TOP),
                                           rotation_xyzw=(0,0,0,1)))

        def configure(cfg):
            cfg = env_cfg_callback(HZ)(cfg)
            cfg.sim.dt = DT
            cfg.decimation = SUBSTEPS
            cfg.sim.render_interval = SUBSTEPS
            cfg.sim.device = PHYS["device"]
            cfg.sim.gravity = tuple(PHYS["gravity_m_s2"])
            cfg.sim.physics.solver_type = PHYS["solver_type"]
            cfg.sim.enable_scene_query_support = True
            cfg.sim.use_fabric = PHYS["use_fabric"]
            cfg.sim.log_dir = cfg.log_dir = str(self.out / "isaac-logs")
            return cfg

        spec = IsaacLabArenaEnvironment(name="cola_foam_fill", embodiment=robot,
                scene=ArenaScene([*task_room("station", self.rig.top_cam), fixture]), env_cfg_callback=configure)
        self._env = ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(
            num_envs=PHYS["num_envs"], device=PHYS["device"], solve_relations=False)).make_registered()
        self._env.reset()
        self._base = self._env.unwrapped
        self.torch = torch
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.valve = self.physics.create_articulation_view(ROOT + "/Switch")
        assert self.valve.count == 1 and self.valve.max_dofs == 1
        assert abs(float(self.valve.get_dof_stiffnesses()[0, 0]) - 420.) < .01
        self._ids = {name: (self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                            self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0])
                     for name in ARMS}
        self.stage = self._base.sim.stage
        self.scene_query = omni.physx.get_physx_scene_query_interface()
        self.jet_origin = (CUP[0], CUP[1], station.TABLE_TOP+CUP[2]+OUTLET_ABOVE_FLOOR)
        light = UsdLux.DomeLight(self.stage.GetPrimAtPath("/World/Light"))
        light.GetColorAttr().Set(Gf.Vec3f(1))
        light.GetIntensityAttr().Set(970.)
        lamp = UsdLux.SphereLight(self.stage.GetPrimAtPath("/World/envs/env_0/lamp"))
        lamp.GetColorAttr().Set(Gf.Vec3f(1))
        lamp.GetIntensityAttr().Set(13500.)
        lamp.GetRadiusAttr().Set(.27)
        self.visual = DrinkVisual(self.stage, ROOT)

    def reset(self, scene, *, seed=None):
        self._ensure_sim()
        self._instruction = scene.instruction
        # Inspect hashes evaluation seeds. The declared scene seed selects the process condition.
        self.seed = scene.init_seed
        assert isinstance(self.seed, int)
        self._env.reset(seed=self.seed)
        self.pour = Pour(CONDITIONS[self.seed % len(CONDITIONS)])
        self.ticks = 0
        self.last_rendered = -1
        self._t = 0.
        self.trace = (self.out / "process.csv").open("x", newline="")
        self.actions = (self.out / "actions.jsonl").open("x")
        self.poses = (self.out / "poses-f64.bin").open("xb")
        self.writer = csv.DictWriter(self.trace, fieldnames=list(self.pour.measure()))
        self.writer.writeheader()
        write_json(self.out / "trace-schema.json", dict(
            dtype="<f8", dt=DT, seed=self.seed, note="Initial state then one row per physics tick.",
            columns=["valve_m", *[f"{arm}.{joint}" for arm in ARMS
                                  for joint in self._base.scene[arm].joint_names]]))
        self.sample()
        self.visual.update(self.pour)
        if self.film:
            self.movie = Recording(self, still=self.still)
            self.movie.frame()
        return self.capture()

    def sample(self):
        self.writer.writerow(self.pour.measure())
        q = [self._base.scene[name].data.joint_pos.torch[0].cpu().numpy() for name in ARMS]
        row = np.asarray([self.pour.travel, *q[0], *q[1]], dtype="<f8")
        assert np.isfinite(row).all()
        self.poses.write(row.tobytes())
        self.frames += 1

    def capture(self):
        if self.last_rendered != self.ticks:
            self.visual.update(self.pour)
            self.render_cameras()
        return self._observation()

    def render_cameras(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        self._base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self._base.scene[name].update(DT, force_recompute=True)
        self.last_rendered = self.ticks

    def metrics(self):
        p = self.pour
        balance = p.emitted - p.liquid - p.pending - p.spill_liquid - p.intercepted
        assert abs(balance) < 1e-7
        return p.measure() | {"max_crest_mm": self.crest_max, "mass_balance_error_ml": balance,
                              "physics_samples": self.frames, "seed": self.seed}

    def jet_hits(self, offsets=JET_OFFSETS):
        """Read entry/exit depths from the current PhysX collision shapes."""
        length = OUTLET_ABOVE_FLOOR
        x, y, z = self.jet_origin
        hits = []

        def cast(origin, direction):
            depths = {}

            def collect(hit):
                shape = hit.collision
                if shape.startswith(("/World/envs/env_0/LeftArm/", "/World/envs/env_0/RightArm/")):
                    depths[shape] = min(depths.get(shape, length), hit.distance)
                return True

            self.scene_query.raycast_all(origin, direction, length, collect)
            return depths

        for lane, (dx, dy) in enumerate(offsets):
            near = cast((x+dx, y+dy, z), (0., 0., -1.))
            if near:
                reverse = cast((x+dx, y+dy, z-length), (0., 0., 1.))
                hits.extend((lane, d, max(d, length-reverse.get(shape, length)), shape)
                            for shape, d in sorted(near.items()))
        return tuple(hits)

    def advance(self, applied):
        target = self.torch.tensor(applied[None, :], dtype=self.torch.float32)
        base = self._base
        base.action_manager.process_action(target)
        for _ in range(SUBSTEPS):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            travel = float(self.valve.get_dof_positions()[0, 0])
            active = travel >= OPEN_AT or self.pour.open or self.pour.tail or self.pour.air
            self.pour.step(travel, DT, self.jet_hits() if active else ())
            self.crest_max = max(self.crest_max, 1000*self.pour.crest / AREA)
            self.ticks += 1
            self.sample()
            if self.movie is not None and self.ticks % (PHYS["physics_hz"] // FPS) == 0:
                self.visual.update(self.pour)
                self.render_cameras()
                self.movie.frame()
        self._t = self.pour.time

    def drive(self, requested):
        assert requested.shape == (14,) and np.isfinite(requested).all()
        applied = np.clip(requested, self._low, self._high)
        self.actions.write(json.dumps(dict(time_s=self.pour.time, requested=requested.tolist(),
                                           applied=applied.tolist()))+"\n")
        self.advance(applied)

    def step(self, action):
        self.drive(np.asarray(action.data))
        p = self.pour
        finished = p.score == 1.
        if round(p.time * HZ) % (HZ * 5) == 0:
            print(f"{p.time:.1f} s  retained {p.liquid:.1f} mL  releases {p.openings}  score {p.score:.3f}",
                  flush=True)
        return StepResult(observation=self.capture(), terminated=finished,
                          termination_reason="success" if finished else None, info=self.metrics())

    def close(self):
        if self.movie is not None:
            self.movie.close()
        for file in (self.trace, self.actions, self.poses):
            if file is not None:
                file.close()
        if self._env is not None:
            self._env.close()

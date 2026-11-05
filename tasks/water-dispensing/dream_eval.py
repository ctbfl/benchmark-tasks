"""One saved Dream-1 trial. --check exercises the adapter without calling Dream."""

import base64
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from inspect_robots_yam.config import DEFAULT_CAMERAS
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from config import DT, TOLERANCE_G
from water import WaterTask

HERE = Path(__file__).resolve().parent
CONTROL_HZ = 20
SUBSTEPS = round(1 / (CONTROL_HZ * DT))
MAX_STEPS = 2400
TARGET_G = 150.0
MODEL = "dreamlabs/dream-1-beta"
INSTRUCTION = (
    "Dispense 150 grams of tea into the cup on the digital scale. "
    "Turn the faucet handle to either side to open it; turning farther increases the flow. "
    "Return the handle to the central closed position to stop the flow. "
    "The handle does not spring back. Use the scale display and allow the remaining "
    f"falling tea to land. Finish within {TOLERANCE_G:g} grams of 150 grams, with the faucet closed "
    "and no overflow."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(file, value):
    file.write(json.dumps(value, allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


def reset_rtx_history():
    """Keep DLAA denoising from retaining previously lit scale segments."""
    import carb.eventdispatcher
    import carb.settings
    carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
    carb.eventdispatcher.get_eventdispatcher().dispatch_event(
        event_name="omni.rtx.ForceResetAccumulation")


class SavedDreamSession(DreamSession):
    """Persist exact JPEG observations and complete replies before executing them."""

    def __init__(self, out):
        super().__init__()
        self.out = out / "dream"
        self.out.mkdir()
        self.tick = 0
        self.observations = (self.out / "observations.jsonl").open("x")
        self.replies = (self.out / "replies.jsonl").open("x")
        for camera in DEFAULT_CAMERAS:
            (self.out / camera).mkdir()

    def observe(self, frames, state, prompt, timestamp):
        paths = {}
        for camera, jpeg in frames.items():
            path = self.out / camera / f"{self.tick:06d}.jpg"
            with path.open("xb") as file:
                file.write(base64.b64decode(jpeg))
                file.flush()
                os.fsync(file.fileno())
            paths[camera] = str(path.relative_to(self.out))
        append_json(self.observations, dict(tick=self.tick, time_s=timestamp,
                    frames=paths, joint_pos=state.tolist(), prompt=prompt))
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies, dict(observation_tick=self.tick - 1,
                    commanded=None if commanded is None else commanded.tolist(),
                    **chunk.to_json()))
        print(f"Saved Dream reply at tick {self.tick - 1}: {len(chunk.rows)} rows, "
              f"{chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class WaterEmbodiment(YamArenaEmbodiment):
    def __init__(self, out, *, camera_size=(640, 360), asset_dir=HERE / "assets"):
        super().__init__(device="cpu", cam_width=camera_size[0], cam_height=camera_size[1],
                         control_hz=CONTROL_HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_water", supported_target_kinds=frozenset(),
            environment_id="yam-water-dispensing-150g-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out = out
        self.asset_dir = asset_dir
        self.app = None
        self.water = WaterTask(TARGET_G)
        self.physics_rows = 0
        self.state_file = (out / "physics-f64.bin").open("xb")
        self.actions_file = (out / "executed-actions.jsonl").open("x")

    def _ensure_sim(self, goal=None):
        if self._env is not None:
            return
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=True,
            kit_args="--/carb/tasking/plugin/threadCount=12 "
                     "--/omni/tbb.globalcontrol/maxThreadCount=12 "
                     "--/persistent/rtx/modes/rt/enabled=false "
                     "--/persistent/rtx/modes/rt2/enabled=true").app

        import carb
        import omni.physics.tensors as tensors
        import omni.usd
        import torch
        from pxr import Gf, UsdLux
        from isaaclab.sim.utils import get_current_stage_id
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene
        from isaaclab_arena.utils.pose import Pose
        from yam_arena import station
        from yam_arena.embodiment import BimanualYamEmbodiment
        from yam_arena.scene import env_cfg_callback, task_room
        from model import WaterVisual
        from yam_preview import pathtraced_camera
        from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

        rig = station.rig()
        robot = BimanualYamEmbodiment(enable_cameras=True, cam_width=self.cam_width, cam_height=self.cam_height,
            contact_sensors=False, robot_contact_offset=0.0005, rig=rig)
        for name in DEFAULT_CAMERAS:
            camera = pathtraced_camera(getattr(robot.camera_config, name))
            camera.renderer_cfg.global_settings = IsaacRtxRendererGlobalSettingsCfg(
                ambient_light_intensity=0., enable_dlssg=False, enable_dl_denoiser=True,
                max_bounces=3, samples_per_pixel=2, carb_settings={
                    "/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 4,
                    "/rtx-transient/post/aa/limitedOps": False,
                    "/rtx/dldenoiser/responsiveDenoising": True,
                    "/rtx/rtpt/cached/enabled": False,
                    "/rtx/rtpt/lightcache/cached/enabled": False,
                })
        tea = Object(name="water_task", object_type=ObjectType.BASE,
            usd_path=str(self.asset_dir / "water-task.usda"),
            initial_pose=Pose(position_xyz=(0.46, 0, station.TABLE_TOP),
                              rotation_xyzw=(0, 0, -math.sqrt(0.5), math.sqrt(0.5))))

        def configure(cfg):
            cfg = env_cfg_callback(CONTROL_HZ)(cfg)
            cfg.sim.use_fabric = False
            cfg.sim.log_dir = cfg.log_dir = str(self.out / "isaac-logs")
            return cfg

        spec = IsaacLabArenaEnvironment(name="yam_water_dream", embodiment=robot,
            scene=ArenaScene([*task_room("station", rig.top_cam), tea]),
            env_cfg_callback=configure)
        self._env = ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=1,
            solve_relations=False, device="cpu")).make_registered()
        self._env.reset()
        self._base = self._env.unwrapped
        self._torch = torch
        self._ready = torch.tensor([[*READY, 1., *READY, 1.]], dtype=torch.float32)
        self._ids = {name: (
            self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0])
            for name in ARMS}
        stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        fill = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        fill.GetColorAttr().Set(Gf.Vec3f(1))
        fill.GetIntensityAttr().Set(900.)
        key = UsdLux.SphereLight(stage.GetPrimAtPath("/World/envs/env_0/lamp"))
        key.GetColorAttr().Set(Gf.Vec3f(1))
        key.GetIntensityAttr().Set(15000.)
        key.GetRadiusAttr().Set(.30)
        root = "/World/envs/env_0/water_task"
        self.visual = WaterVisual(stage, root)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.valve = self.physics.create_articulation_view(root + "/Dispenser")
        assert self.valve.count == 1 and self.valve.max_dofs == 1
        assert np.all(self.valve.get_dof_stiffnesses() == 0)
        columns = ["time_s", "valve_rad", "valve_vel", "cup_g", "airborne_g",
                   "emitted_g", "spilled_g", "flow_g_s"]
        for name in ARMS:
            names = self._base.scene[name].joint_names
            columns += [f"{name}.{kind}.{joint}" for kind in ("pos", "vel") for joint in names]
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=columns,
            physics_dt=DT, robot_roots_fixed=True, task_pose=[.46, 0, station.TABLE_TOP],
            task_rotation_xyzw=[0, 0, -math.sqrt(.5), math.sqrt(.5)],
            water_update="WaterTask.step(mean(previous_valve_rad, valve_rad), physics_dt)"))

    def set_valve_for_reset(self, angle=0.):
        indices = np.array([0], dtype=np.uint32)
        self.valve.set_dof_positions(np.array([[angle]], dtype=np.float32), indices)
        self.valve.set_dof_velocities(np.zeros((1, 1), dtype=np.float32), indices)

    def reset(self, scene, *, seed=None):
        self._ensure_sim()
        self._instruction = scene.instruction
        self._env.reset(seed=seed)
        self.set_valve_for_reset()
        for _ in range(CONTROL_HZ // 2):
            self._env.step(self._ready)
        self.set_valve_for_reset()
        self.water.reset(TARGET_G)
        self._t = 0.
        self.save_state()
        for _ in range(4):
            observation = self.capture()
        return observation

    def save_state(self):
        w = self.water
        row = [w.time, float(self.valve.get_dof_positions()[0, 0]),
               float(self.valve.get_dof_velocities()[0, 0]), w.cup_g, w.airborne_g,
               w.emitted_g, w.spilled_g, w.flow]
        for name in ARMS:
            data = self._base.scene[name].data
            row.extend(data.joint_pos.torch[0].cpu().numpy())
            row.extend(data.joint_vel.torch[0].cpu().numpy())
        array = np.asarray(row, dtype="<f8")
        assert np.isfinite(array).all(), "Non-finite physical state"
        self.state_file.write(array.tobytes())
        self.physics_rows += 1

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.visual.update(self.water)
        reset_rtx_history()
        self._base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in DEFAULT_CAMERAS:
            self._base.scene[name].update(1 / CONTROL_HZ, force_recompute=True)
        return self._observation()

    def metrics(self):
        w = self.water
        return dict(success=w.success, time_s=w.time, target_g=TARGET_G, cup_g=w.cup_g,
            reading_g=w.reading_g, error_g=w.cup_g - TARGET_G, airborne_g=w.airborne_g,
            emitted_g=w.emitted_g, spilled_g=w.spilled_g, flow_g_s=w.flow,
            quiet_s=w.quiet_s, mass_balance_error_g=w.emitted_g - w.cup_g - w.airborne_g - w.spilled_g,
            physics_rows=self.physics_rows)

    def step(self, action):
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions_file, dict(time_s=self._t, requested=requested.tolist(),
                                           applied=target.tolist()))
        base = self._base
        base.action_manager.process_action(self._torch.tensor([target.tolist()], dtype=self._torch.float32))
        for _ in range(SUBSTEPS):
            previous = float(self.valve.get_dof_positions()[0, 0])
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            angle = float(self.valve.get_dof_positions()[0, 0])
            self.water.step((previous + angle) / 2, DT)
            self.save_state()
        self.state_file.flush()
        os.fsync(self.state_file.fileno())
        self._t = self.water.time
        info = self.metrics()
        assert abs(info["mass_balance_error_g"]) < 1e-6
        if round(self._t * CONTROL_HZ) % (CONTROL_HZ * 5) == 0:
            print(f"Trial {self._t:.1f}s: {self.water.reading_g:.1f} g, "
                  f"valve {math.degrees(angle):.1f} deg", flush=True)
            write_json(self.out / "progress.json", info)
        return StepResult(observation=self.capture(), terminated=self.water.success,
            termination_reason="success" if self.water.success else None, info=info)

    def close(self):
        self.state_file.close()
        self.actions_file.close()
        if self._env is not None:
            self._env.close()
            self._env = self._base = None


def check(embodiment, scene):
    from PIL import Image
    obs = embodiment.reset(scene, seed=0)
    assert obs.state["joint_pos"].shape == (14,)
    for name, rgb in obs.images.items():
        assert rgb.shape == (embodiment.cam_height, embodiment.cam_width, 3) and rgb.std() > 10
        Image.fromarray(rgb).save(embodiment.out / f"initial-{name}.png")
    hold = Action(data=np.array([*READY, 1., *READY, 1.]))
    # Only this offline check directly places the valve; evaluation never does.
    embodiment.set_valve_for_reset(math.pi / 3)
    for _ in range(20):
        embodiment.step(hold)
    assert embodiment.water.cup_g > 5 and embodiment.water.airborne_g > 0
    before, airborne = embodiment.water.cup_g, embodiment.water.airborne_g
    embodiment.set_valve_for_reset()
    for _ in range(12):
        embodiment.step(hold)
    assert embodiment.water.airborne_g == 0
    assert embodiment.water.cup_g >= before + airborne - 1e-6
    assert abs(embodiment.metrics()["mass_balance_error_g"]) < 1e-6
    write_json(embodiment.out / "flow-check.json", embodiment.metrics())
    embodiment.reset(scene, seed=0)
    assert embodiment.water.emitted_g == 0 and embodiment.water.cup_g == 0
    assert abs(float(embodiment.valve.get_dof_positions()[0, 0])) < 1e-5
    write_json(embodiment.out / "check.json", dict(passed=True, api_inference_calls=0))
    print("OFFLINE CHECK PASSED", flush=True)


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check_only = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / f"dream-{'check' if check_only else '150g'}-{stamp}"
    out.mkdir(parents=True)
    (out / "videos").mkdir()
    source = out / "source"
    source.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(HERE / "assets", source / "assets")
    write_json(out / "run.json", dict(model=MODEL, check_only=check_only, target_g=TARGET_G,
        tolerance_g=TOLERANCE_G, max_steps=MAX_STEPS, control_hz=CONTROL_HZ, physics_dt=DT,
        camera_size=[640, 360], model_input_size=[320, 176], seed=0, instruction=INSTRUCTION,
        renderer="RTX Real-Time 2.0, DLAA", layout="fixed, calibrated YAM station",
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    print(f"RUN_DIRECTORY={out}", flush=True)
    embodiment = WaterEmbodiment(out)
    policy = None
    exit_code = 0
    try:
        scene = Scene(id="dispense-150g", instruction=INSTRUCTION, init_seed=0)
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(out))
            task = Task(name="water-dispensing-150g", scenes=[scene],
                        scorer="success_at_end", max_steps=MAX_STEPS)
            logs = evaluate(task, policy=policy, embodiment=embodiment,
                log_dir=str(out / "eval"), seed=0, fail_on_error=True,
                store_frames=True, store_actions=True)
            print(f"EVALUATION COMPLETE: {len(logs)} log(s)", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        write_json(out / "metrics.json", dict(exit_code=exit_code, **embodiment.metrics()))
        if policy is not None:
            write_json(out / "dream-transcript.json", policy.transcript())
            policy.close()
        embodiment.close()
        print(f"Saved outputs: {out}", flush=True)
        if embodiment.app is not None:
            embodiment.app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

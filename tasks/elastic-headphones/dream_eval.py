"""Archive one real Dream-1 trial; --check validates the adapter without inference."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from inspect_robots import Action, Scene, Task, StepResult, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment
from scoring import FinalScore, HeadphoneScore, JOINTS

HERE = Path(__file__).resolve().parent
HZ, DT, MAX_STEPS = 20, 1 / 240, 2400
CAMERAS = ("left_cam", "top_cam", "right_cam")
MODEL = "dreamlabs/dream-1-beta"
INSTRUCTION = (
    "Unfold the folded headphones on the right side of the table, then hang them on "
    "the stand on the left. Open the inward-folded earcup and rotate both earcups into "
    "their normal wearing positions, with the cushions facing inward. The headband "
    "is elastic and the earcups have passive folding and swivel joints. Rest the top "
    "of the headband inside the stand's U-shaped cradle. Release the headphones and "
    "move both hands clear, leaving the headphones hanging stably for at least 5 seconds. "
    "Complete the task within 120 seconds."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(file, value):
    file.write(json.dumps(value, allow_nan=False) + "\n")
    file.flush()
    os.fsync(file.fileno())


class SavedDreamSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out = out / "dream"
        self.out.mkdir()
        self.tick = 0
        self.observations = (self.out / "observations.jsonl").open("x")
        self.replies = (self.out / "replies.jsonl").open("x")
        for name in CAMERAS:
            (self.out / name).mkdir()

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        paths = {}
        for name, jpeg in frames.items():
            path = self.out / name / f"{self.tick:06d}.jpg"
            with path.open("xb") as file:
                file.write(base64.b64decode(jpeg))
                file.flush()
                os.fsync(file.fileno())
            paths[name] = str(path.relative_to(self.out))
        append_json(self.observations, dict(tick=self.tick, time_s=timestamp,
            frames=paths, joint_pos=state.tolist(), prompt=prompt))
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies, dict(observation_tick=self.tick - 1,
            commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print(f"Dream reply at {(self.tick - 1) / HZ:.2f}s: {len(chunk.rows)} actions, "
              f"{chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class HeadphoneEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_elastic_headphones",
            supported_target_kinds=frozenset(), environment_id="yam-elastic-headphones-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out, self.physics_ticks, self.control_steps = out, 0, 0
        self.last_info = {}
        self.scorer = HeadphoneScore(HZ)
        self.states = (out / "physics-f64.bin").open("xb")
        self.actions = (out / "executed-actions.jsonl").open("x")
        self.measurements = (out / "measurements.jsonl").open("x")

    def _ensure_sim(self):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from pxr import PhysxSchema, Sdf
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.robot import asset_link_paths
        from yam_scene import START_ANGLES
        from yam_scene import make_env

        assert self._env is None, "One episode per archive"
        self._env = make_env(enable_cameras=True, control_hz=HZ, log_dir=self.out / "logs")
        self._base = self.base = self._env.unwrapped
        self._torch = torch
        root = "/World/envs/env_0/headphones"
        for name in START_ANGLES:
            prim = self.base.sim.stage.GetPrimAtPath(root + "/joints/" + name)
            assert prim.ApplyAPI("PhysxJointAxisAPI", "angular")
            for attr, value in (("staticFrictionEffort", .07), ("dynamicFrictionEffort", .05)):
                prim.CreateAttribute("physxJointAxis:angular:" + attr, Sdf.ValueTypeNames.Float).Set(value)
        self._env.reset()
        self.hp, self.stand = self.base.scene["headphones"], self.base.scene["headphone_stand"]
        self.robots = [self.base.scene[name] for name in ARMS]
        self._ids = {name: (self.base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
                           self.base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        self.view = self.physics.create_articulation_view(root)
        assert self.view.count == 1 and self.view.max_dofs == 14 and not self.hp.is_fixed_base
        assert not self.hp.actuators and abs(self.base.physics_dt - DT) < 1e-10
        for name in self.hp.body_names:
            PhysxSchema.PhysxContactReportAPI.Apply(
                self.base.sim.stage.GetPrimAtPath(root + "/" + name)).CreateThresholdAttr(0)
        paths = asset_link_paths()
        robot_links = [f"/World/envs/env_0/{side}Arm{path}"
                       for side in ("Left", "Right") for path in paths.values()]
        self.contacts = self.physics.create_rigid_contact_view(
            [root + "/" + name for name in self.hp.body_names], [robot_links] * self.hp.num_bodies)
        self.stand_contacts = self.physics.create_rigid_contact_view(
            "/World/envs/env_0/headphone_stand", robot_links)
        self.support = self.physics.create_rigid_contact_view(root + "/band_*",
            ["/World/envs/env_0/headphone_stand"])
        self.spring_stiffness = self.view.get_dof_stiffnesses().copy()
        self.spring_targets = self.view.get_dof_position_targets().copy()
        self.joint_ids = {name: self.hp.joint_names.index(name) for name in JOINTS}
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=192, physics_dt=DT,
            quaternion_order="xyzw", fields={"time": [0, 1], "robot_q": [1, 17], "robot_dq": [17, 33],
                "root": [33, 40], "root_velocity": [40, 46], "joints": [46, 60],
                "joint_velocity": [60, 74], "stand": [74, 81], "stand_velocity": [81, 87], "links": [87, 192]},
            headphone_joint_names=self.hp.joint_names, headphone_link_names=self.hp.body_names,
            robot_joint_names={name: self.base.scene[name].joint_names for name in ARMS}))

    def reset(self, scene, *, seed=None):
        self._ensure_sim()
        self._instruction = scene.instruction
        self._env.reset(seed=seed)
        ready = self._torch.tensor([[*READY, 1., *READY, 1.]], dtype=self._torch.float32)
        for _ in range(HZ):
            self._env.step(ready)
        self._t = 0.
        self.save_state()
        for _ in range(4):
            observation = self.capture()
        self.support_force = self.robot_force = 0.
        self.last_info = self.measure()
        assert self.last_info["score"] == 0., self.last_info
        return observation

    def save_state(self):
        def array(data):
            return data.torch[0].cpu().numpy().ravel()
        row = np.concatenate(([self.physics_ticks * DT],
            *[array(robot.data.joint_pos) for robot in self.robots],
            *[array(robot.data.joint_vel) for robot in self.robots],
            self.view.get_root_transforms()[0], self.view.get_root_velocities()[0],
            self.view.get_dof_positions()[0], self.view.get_dof_velocities()[0],
            array(self.stand.data.root_pose_w), array(self.stand.data.root_vel_w),
            array(self.hp.data.body_link_pose_w))).astype("<f8")
        assert row.shape == (192,) and np.isfinite(row).all(), "Invalid physical state"
        self.states.write(row.tobytes())

    def capture(self):
        import carb.settings
        import carb.eventdispatcher
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self.base.scene[name].update(1 / HZ, force_recompute=True)
        return self._observation()

    def step(self, action):
        assert self.control_steps < MAX_STEPS
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions, dict(time_s=self._t, requested=requested.tolist(), applied=target.tolist()))
        self.base.action_manager.process_action(self._torch.tensor(target[None], dtype=self._torch.float32))
        self.support_force = self.robot_force = 0.
        for _ in range(12):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self.physics_ticks += 1
            self.save_state()
            self.support_force = max(self.support_force,
                float(np.linalg.norm(self.support.get_contact_force_matrix(DT), axis=-1).sum()))
            self.robot_force = max(self.robot_force, *[
                float(np.linalg.norm(view.get_contact_force_matrix(DT), axis=-1).sum())
                for view in (self.contacts, self.stand_contacts)])
        self.control_steps += 1
        self._t = self.control_steps / HZ
        self.states.flush()
        os.fsync(self.states.fileno())
        assert np.array_equal(self.spring_stiffness, self.view.get_dof_stiffnesses())
        assert np.array_equal(self.spring_targets, self.view.get_dof_position_targets())
        self.last_info = self.measure()
        append_json(self.measurements, self.last_info)
        if self.control_steps % (5 * HZ) == 0:
            write_json(self.out / "progress.json", self.last_info)
            print(f"Trial {self._t:.1f}s: score={self.last_info['score']}, "
                  f"angles={self.last_info['angles_deg']}", flush=True)
        return StepResult(observation=self.capture(), terminated=self.last_info["success"],
            termination_reason="success" if self.last_info["success"] else None,
            truncated=self.control_steps >= MAX_STEPS and not self.last_info["success"], info=self.last_info)

    def measure(self):
        angles = np.rad2deg(self.view.get_dof_positions()[0])
        return self.scorer.update(dict(time_s=self._t, root=self.view.get_root_transforms()[0],
            stand=self.stand.data.root_pose_w.torch[0].cpu().numpy(),
            links=self.hp.data.body_link_pose_w.torch[0].cpu().numpy(),
            angles_deg={name: float(angles[i]) for name, i in self.joint_ids.items()},
            support_force_N=self.support_force, robot_contact_N=self.robot_force))

    def close(self):
        self.states.close()
        self.actions.close()
        self.measurements.close()
        if self._env is not None:
            self._env.close()


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check_only = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / f"dream-{'check' if check_only else 'eval'}-{stamp}"
    for directory in ("logs", "frames", "videos", "source"):
        (out / directory).mkdir(parents=True)
    print(f"RUN_DIRECTORY={out}", flush=True)
    source = out / "source"
    for name in ("dream_eval.py", "scoring.py", "yam_scene.py", "demo.py", "record_demo.py", "demo_actions.npy", "render.py", "score.py"):
        shutil.copy2(HERE / name, source / name)
    shutil.copytree(HERE / "assets", source / "assets")
    shutil.copytree(HERE / "fixtures", source / "fixtures")
    write_json(out / "run.json", dict(model=MODEL, check_only=check_only, instruction=INSTRUCTION,
        max_steps=MAX_STEPS, control_hz=HZ, physics_dt=DT, camera_size=[640, 360], model_input_size=[320, 176],
        seed=0, renderer="RealTimePathTracing", scorer=FinalScore.name,
        scoring_weights=dict(unfold=.4, left_earcup=.2, right_earcup=.2, hang=.2),
        input_state_keys=["joint_pos"], input_cameras=list(CAMERAS), api_wait_advances_physics=False,
        gripper_gains="Native YAM 5000 N/m, 100 N s/m, 50 N limit; no policy gain override",
        expert_difference="Demo replays fixed recorded actions without observations; both policies use native YAM gains",
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob("*") if p.is_file()},
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
        f"--/log/file={out}/logs/kit.log").app
    embodiment = HeadphoneEmbodiment(out)
    policy, exit_code, error = None, 0, None
    started = time.perf_counter()
    try:
        scene = Scene(id="elastic-headphones", instruction=INSTRUCTION, init_seed=0)
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(out))
            task = Task(name="elastic-headphones", scenes=[scene], scorer=FinalScore(), max_steps=MAX_STEPS)
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out / "eval"),
                seed=0, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
            print("EVALUATION COMPLETE", flush=True)
    except BaseException:
        exit_code = 1
        error = traceback.format_exc()
        traceback.print_exc()
    finally:
        write_json(out / "metrics.json", dict(exit_code=exit_code, error=error,
            wall_time_s=time.perf_counter() - started, physics_ticks=embodiment.physics_ticks,
            control_steps=embodiment.control_steps, **embodiment.last_info))
        if policy is not None:
            write_json(out / "dream-transcript.json", policy.transcript())
            policy.close()
        embodiment.close()
        print(f"Saved outputs: {out}", flush=True)
        app.close(exit_code=exit_code)


def check(embodiment, scene):
    from PIL import Image
    obs = embodiment.reset(scene, seed=0)
    assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"} and not obs.extra
    for name, rgb in obs.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.std() > 10
        Image.fromarray(rgb).save(embodiment.out / "frames" / f"initial-{name}.png")
    q = obs.state["joint_pos"].copy()
    target = q.copy()
    target[0] += .12
    for _ in range(HZ):
        result = embodiment.step(Action(data=target))
    difference = float(np.abs(result.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean())
    assert abs(result.observation.state["joint_pos"][0] - q[0]) > .05 and difference > 1
    assert embodiment.physics_ticks == 240 and result.observation.state_time == 1.
    assert result.info["score"] == 0.
    # Positive control only: restore a released hanging pose, then let PhysX
    # support it on the stand. This placement is never called during inference.
    final = json.loads((HERE / "fixtures/hanging.json").read_text())
    tensor = lambda x: embodiment._torch.tensor(np.asarray(x)[None], dtype=embodiment._torch.float32)
    embodiment.hp.write_root_pose_to_sim(tensor(final["root"]))
    embodiment.hp.write_root_velocity_to_sim(tensor(np.zeros(6)))
    embodiment.hp.write_joint_state_to_sim(tensor(np.deg2rad(final["joint_degrees"])), tensor(np.zeros(14)))
    embodiment.stand.write_root_pose_to_sim(tensor(final["stand"]))
    embodiment.stand.write_root_velocity_to_sim(tensor(np.zeros(6)))
    max_support = 0.
    for _ in range(150):
        result = embodiment.step(Action(data=target))
        max_support = max(max_support, result.info["support_force_N"])
        if result.info["success"]:
            break
    assert result.info["score"] == 1. and max_support > .5, result.info
    for name, rgb in result.observation.images.items():
        Image.fromarray(rgb).save(embodiment.out / "frames" / f"hanging-{name}.png")
    write_json(embodiment.out / "check.json", dict(passed=True, api_calls=0,
        moving_camera_rgb_mae=difference, moving_physics_ticks=240, initial_and_idle_score=0,
        released_hanging_score=result.info["score"], positive_support_force_N=max_support))
    print("OFFLINE CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()

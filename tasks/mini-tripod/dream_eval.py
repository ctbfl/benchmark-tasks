"""One archived Dream-1 trial; --check validates the adapter without API calls."""
import base64
from collections import deque
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

ROOT = Path(__file__).resolve().parent
CONTROL_HZ = 20
PHYSICS_DT = 1 / 480
MAX_SECONDS = 120
MAX_STEPS = MAX_SECONDS * CONTROL_HZ
STABLE_SECONDS = 5
CAMERAS = ("left_cam", "top_cam", "right_cam")
MODEL = "dreamlabs/dream-1-beta"
INSTRUCTION = (
    "Open the mini tripod lying on the left side of the table. Hold its central body "
    "and unfold all three fixed-length legs outward to their mechanical stops. "
    "Place it upright and centered inside the red tape square, with all three feet "
    "resting on the tabletop. Release the tripod and move both grippers clear. "
    "Leave it standing stably for at least 5 seconds. Any yaw orientation is acceptable. "
    "Complete the task within 120 seconds."
)

if __name__ == "__main__":
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    CHECK = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT = ROOT / "output" / f"dream-{'check' if CHECK else 'eval'}-{stamp}"
    (OUT / "logs").mkdir(parents=True)
    print(f"RUN_DIRECTORY={OUT}", flush=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 "
        "--/plugins/omni.tbb.globalcontrol/maxThreadCount=6 "
        f"--/log/file={OUT}/logs/kit.log").app

import numpy as np
from scipy.spatial.transform import Rotation
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment


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
        for camera in CAMERAS:
            (self.out / camera).mkdir()

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
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
            commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print(f"Dream reply at {(self.tick - 1) / CONTROL_HZ:.2f}s: "
              f"{len(chunk.rows)} actions, {chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class TripodEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=CONTROL_HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_mini_tripod", supported_target_kinds=frozenset(),
            environment_id="yam-mini-tripod-fixed-legs-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out = out
        self.physics_ticks = self.control_steps = 0
        self.last_info = {}
        self.window = deque(maxlen=STABLE_SECONDS * CONTROL_HZ)
        self.state_file = (out / "states-30hz-f64.bin").open("xb")
        self.actions_file = (out / "executed-actions.jsonl").open("x")
        self.measurements = (out / "measurements.jsonl").open("x")

    def _ensure_sim(self):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.robot import asset_link_paths
        import yam_scene
        assert self._env is None, "One episode per archive"
        yam_scene.OUT = self.out
        self._env, self.layout = yam_scene.make_env(enable_cameras=True, control_rate=CONTROL_HZ)
        self._base = self.base = self._env.unwrapped
        self._torch = torch
        self._ids = {name: (
            self.base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self.base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        self._env.reset()
        self.tripod = self.base.scene["mini_tripod"]
        self.robots = [self.base.scene[name] for name in ARMS]
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        root = "/World/envs/env_0/mini_tripod"
        self.view = self.physics.create_articulation_view(root)
        assert self.view.count == 1 and self.view.max_dofs == 3 and not self.tripod.is_fixed_base
        assert np.all(self.view.get_dof_stiffnesses() == 0) and np.all(self.view.get_dof_dampings() == 0)
        self.vertices = dict(np.load(ROOT / "assets/collision_vertices.npz"))
        paths = asset_link_paths()
        hands = [f"/World/envs/env_0/{side}Arm{paths[name]}" for side in ("Left", "Right")
                 for name in ("gripper", "tip_left", "tip_right")]
        self.hand_contacts = self.physics.create_rigid_contact_view(root + "/*", hands)
        self.feet = self.physics.create_rigid_contact_view(root + "/leg_*",
            ["/World/envs/env_0/table/geometry/mesh"])
        write_json(self.out / "layout.json", self.layout)
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=36, fps=30,
            fields={"time": [0, 1], "robot_q": [1, 17], "hub_pose": [17, 24],
                    "tripod_q": [24, 27], "hub_velocity": [27, 33], "joint_velocity": [33, 36]},
            robot_q_shape=[2, 8], quaternion_order="xyzw", physics_dt=PHYSICS_DT))

    def reset(self, scene, *, seed=None):
        assert scene.instruction == INSTRUCTION
        self._ensure_sim()
        self._instruction = INSTRUCTION
        ready = self._torch.tensor([[*READY, 1., *READY, 1.]], dtype=self._torch.float32)
        peak = 0.
        for _ in range(2 * CONTROL_HZ):
            self._env.step(ready)
            peak = max(peak, float(np.linalg.norm(self.feet.get_contact_force_matrix(PHYSICS_DT), axis=-1).max()))
        assert peak > .03, "Initial tabletop contact reporting failed"
        self._t = 0.
        self.save_state()
        self.last_info = self.measure()
        assert not self.last_info["success"] and max(np.abs(self.last_info["hinges_deg"])) < 3
        write_json(self.out / "reset.json", dict(seed=seed, randomization=False,
            table_contact_peak_N=peak, initial=self.last_info))
        for _ in range(4):
            observation = self.capture()
        from PIL import Image
        for name, rgb in observation.images.items():
            Image.fromarray(rgb).save(self.out / f"initial-{name}.png")
        return observation

    def save_state(self):
        row = np.r_[self.physics_ticks * PHYSICS_DT,
            np.concatenate([r.data.joint_pos.torch[0].cpu().numpy() for r in self.robots]),
            self.tripod.data.root_link_pose_w.torch[0].cpu().numpy(),
            self.tripod.data.joint_pos.torch[0].cpu().numpy(),
            self.tripod.data.root_link_vel_w.torch[0].cpu().numpy(),
            self.tripod.data.joint_vel.torch[0].cpu().numpy()].astype("<f8")
        assert row.shape == (36,) and np.isfinite(row).all()
        row[0] = self.physics_ticks * PHYSICS_DT
        self.state_file.write(row.tobytes())

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self.base.scene[name].update(1 / CONTROL_HZ, force_recompute=True)
        return self._observation()

    def measure(self):
        from yam_arena import station
        pose = self.tripod.data.root_link_pose_w.torch[0].cpu().numpy()
        angles = np.rad2deg(self.tripod.data.joint_pos.torch[0].cpu().numpy())
        clouds = {name: Rotation.from_quat(p[3:]).apply(self.vertices[name]) + p[:3]
            for name, p in zip(self.view.shared_metatype.link_names, self.view.get_link_transforms()[0])}
        cloud = np.concatenate(list(clouds.values()))
        lo, hi = cloud[:, :2].min(0), cloud[:, :2].max(0)
        margins = np.r_[lo - self.layout["goal_low_xy_m"], self.layout["goal_high_xy_m"] - hi]
        center_error = float(np.linalg.norm((lo + hi) / 2 - self.layout["goal_center_xy_m"]))
        tilt = float(np.rad2deg(np.arccos(np.clip(Rotation.from_quat(pose[3:]).as_matrix()[2, 2], -1, 1))))
        foot_height = np.array([clouds[f"leg_{i}"][:, 2].min() - station.TABLE_TOP for i in range(3)])
        forces = np.linalg.norm(self.feet.get_contact_force_matrix(PHYSICS_DT), axis=-1).reshape(3)
        hand_force = float(np.linalg.norm(self.hand_contacts.get_contact_force_matrix(PHYSICS_DT), axis=-1).max())
        hand_points = np.concatenate([r.data.body_link_pose_w.torch[0,
            [r.body_names.index(n) for n in ("gripper", "tip_left", "tip_right")], :3].cpu().numpy()
            for r in self.robots])
        clearance = float(np.linalg.norm(cloud[:, None, :] - hand_points[None, :, :], axis=-1).min())
        checks = dict(all_three_legs_fully_open=bool(np.max(np.abs(angles - 32)) < 1),
            upright=tilt < 5, inside_red_square=bool(margins.min() > 0),
            centered=center_error < .005, feet_on_table=bool(np.max(np.abs(foot_height)) < .0015),
            hands_clear=hand_force < .01 and clearance > .02)
        if self.control_steps:
            self.window.append((pose[:3].copy(), forces.copy(), all(checks.values())))
        drift = float(np.linalg.norm(np.ptp([x[0] for x in self.window], axis=0))) if self.window else 0.
        support = np.max([x[1] for x in self.window], axis=0) if self.window else forces
        stable = len(self.window) == self.window.maxlen and all(x[2] for x in self.window)
        checks["stable_5_seconds"] = stable and drift < .002 and bool(np.all(support > .03))
        assert np.all(self.view.get_dof_actuation_forces() == 0), "Tripod must stay passive"
        return dict(time_s=self._t, hinges_deg=angles.tolist(), hub_pose=pose.tolist(),
            tilt_deg=tilt, tape_clearances_m=margins.tolist(), center_error_m=center_error,
            foot_heights_m=foot_height.tolist(), foot_contact_N=forces.tolist(),
            hand_contact_N=hand_force, gripper_clearance_m=clearance, drift_last_5s_m=drift,
            checks=checks, success=bool(all(checks.values())), score=int(all(checks.values())))

    def step(self, action):
        assert self.control_steps < MAX_STEPS
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions_file, dict(time_s=self._t, requested=requested.tolist(), applied=target.tolist()))
        self.base.action_manager.process_action(self._torch.tensor(target[None], dtype=self._torch.float32))
        for _ in range(round(1 / (CONTROL_HZ * PHYSICS_DT))):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(PHYSICS_DT)
            self.physics_ticks += 1
            if self.physics_ticks % 16 == 0:
                self.save_state()
        self.control_steps += 1
        self._t = self.control_steps / CONTROL_HZ
        self.state_file.flush()
        os.fsync(self.state_file.fileno())
        self.last_info = self.measure()
        append_json(self.measurements, self.last_info)
        if self.control_steps % (5 * CONTROL_HZ) == 0:
            write_json(self.out / "progress.json", self.last_info)
            print(f"Trial {self._t:.1f}s: hinges={np.round(self.last_info['hinges_deg'], 1)}, "
                  f"success={self.last_info['success']}", flush=True)
        return StepResult(observation=self.capture(), terminated=self.last_info["success"],
            termination_reason="success" if self.last_info["success"] else None,
            truncated=self.control_steps >= MAX_STEPS and not self.last_info["success"], info=self.last_info)

    def close(self):
        self.state_file.close()
        self.actions_file.close()
        self.measurements.close()
        rows = np.fromfile(self.out / "states-30hz-f64.bin", dtype="<f8").reshape(-1, 36)
        np.savez_compressed(self.out / "trajectory.npz", time=rows[:, 0],
            robot_q=rows[:, 1:17].reshape(-1, 2, 8), hub_pose=rows[:, 17:24], tripod_q=rows[:, 24:27],
            hub_velocity=rows[:, 27:33], joint_velocity=rows[:, 33:36],
            phase=np.full(len(rows), "Dream-1"))
        if self._env is not None:
            self._env.close()


def main():
    source = OUT / "source"
    source.mkdir()
    for name in ("dream_eval.py", "yam_scene.py", "model.py", "replay.py"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "assets", source / "assets")
    write_json(OUT / "run.json", dict(model=MODEL, check_only=CHECK, instruction=INSTRUCTION,
        max_seconds=MAX_SECONDS, max_steps=MAX_STEPS, stable_seconds=STABLE_SECONDS,
        control_hz=CONTROL_HZ, physics_dt=PHYSICS_DT, camera_size=[640, 360], model_input_size=[320, 176],
        seed=0, renderer="RealTimePathTracing", scorer="success_at_end", input_state_keys=["joint_pos"],
        api_wait_advances_physics=False, input_cameras=list(CAMERAS),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob("*") if p.is_file()},
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    embodiment = TripodEmbodiment(OUT)
    policy = None
    started, exit_code = time.perf_counter(), 0
    try:
        scene = Scene(id="mini-tripod", instruction=INSTRUCTION, init_seed=0)
        if CHECK:
            obs = embodiment.reset(scene, seed=0)
            assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"} and not obs.extra
            assert all(rgb.shape == (360, 640, 3) and rgb.std() > 10 for rgb in obs.images.values())
            q = obs.state["joint_pos"].copy()
            target = q.copy()
            target[0] += .12
            for _ in range(CONTROL_HZ):
                result = embodiment.step(Action(data=target))
            difference = float(np.abs(result.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean())
            assert abs(result.observation.state["joint_pos"][0] - q[0]) > .05 and difference > 1
            assert embodiment.physics_ticks == 480 and result.observation.state_time == 1.
            assert not result.info["success"]
            write_json(OUT / "check.json", dict(passed=True, api_calls=0,
                moving_camera_rgb_mae=difference, physics_ticks=480, max_seconds=MAX_SECONDS))
            print("OFFLINE CHECK PASSED", flush=True)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(OUT))
            task = Task(name="mini-tripod", scenes=[scene], scorer="success_at_end", max_steps=MAX_STEPS)
            logs = evaluate(task, policy=policy, embodiment=embodiment,
                log_dir=str(OUT / "eval"), seed=0, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
            print("EVALUATION COMPLETE", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        write_json(OUT / "metrics.json", dict(exit_code=exit_code, wall_total_s=time.perf_counter() - started,
            physics_ticks=embodiment.physics_ticks, **embodiment.last_info))
        if policy is not None:
            write_json(OUT / "dream-transcript.json", policy.transcript())
            policy.close()
        embodiment.close()
        print(f"Saved outputs: {OUT}", flush=True)
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

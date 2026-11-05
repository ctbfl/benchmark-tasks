"""One archived Dream-1 evaluation; --check validates the adapter without API calls."""

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
from scipy.spatial.transform import Rotation
from scoring import MAX_SECONDS, score_placement
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

HERE = Path(__file__).resolve().parent
MODEL = "dreamlabs/dream-1-beta"
CAMERAS = ("left_cam", "top_cam", "right_cam")
CONTROL_HZ = 20
PHYSICS_DT = 1 / 480
MAX_STEPS = MAX_SECONDS * CONTROL_HZ
INSTRUCTION = json.loads((HERE / "task.json").read_text())["instruction"]


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
            path.write_bytes(base64.b64decode(jpeg))
            paths[camera] = str(path.relative_to(self.out))
        append_json(self.observations, dict(tick=self.tick, time_s=timestamp,
                    frames=paths, joint_pos=state.tolist(), prompt=prompt))
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies, dict(observation_tick=self.tick - 1,
                    commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print(f"Dream reply at {((self.tick - 1) / CONTROL_HZ):.2f}s: "
              f"{len(chunk.rows)} actions, {chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class PhoneHolderEmbodiment(YamArenaEmbodiment):
    def __init__(self, out, record):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=CONTROL_HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_phone_holder", supported_target_kinds=frozenset(),
            environment_id="yam-use-phoneholder-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out, self.record, self.app, self.camera = out, record, None, None
        self.phase = "Dream-1 evaluation"
        self.physics_ticks, self.control_steps = 0, 0
        self.last_info = {}
        self.state_file = (out / "physics-f64.bin").open("xb")
        self.actions_file = (out / "executed-actions.jsonl").open("x")
        self.measurements = (out / "measurements.jsonl").open("x")
        self.surface = np.load(HERE / "assets/iphone17pro/surface_points.npy")

    @property
    def step_id(self):
        return self.physics_ticks // 16

    def panel_angle(self):
        return self.monitor.panel_angle()

    def _ensure_sim(self):
        assert self._env is None, "One trial per archive"
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=True,
            kit_args="--/plugins/carb.tasking.plugin/threadCount=8 "
                     "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
        import torch
        from camera_ready import Recorder
        from expert import Expert
        from yam_scene import make_env
        from yam_arena.robot import asset_link_paths

        self._env = make_env(control_hz=CONTROL_HZ, enable_cameras=True,
                            log_dir=self.out / "isaac-logs", use_fabric=True,
                            device="cpu", physics_dt=PHYSICS_DT)
        self._base = self.base = self._env.unwrapped
        self._torch = torch
        self._ids = {name: (
            self.base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self.base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        self._env.reset()
        # Reuse only pose/geometry/contact measurements; never call expert motion methods.
        self.monitor = Expert(self._env)
        assert not self.monitor.holder.is_fixed_base
        assert bool((self.monitor.passive_view.get_dof_stiffnesses() == 0).all())
        paths = asset_link_paths()
        hands = [f"/World/envs/env_0/{side}Arm{paths[name]}" for side in ("Left", "Right")
                 for name in ("gripper", "tip_left", "tip_right")]
        self.holder_hands = self.monitor.physics.create_rigid_contact_view(
            "/World/envs/env_0/phoneholder/*", hands)
        self.phone_hands = self.monitor.physics.create_rigid_contact_view(
            "/World/envs/env_0/phone", hands)
        self.camera = Recorder(self._env, self.out, self.record, filename="dream.mp4")
        self.bodies = self.monitor.physics.create_rigid_body_view([
            "/World/envs/env_0/phoneholder/*", "/World/envs/env_0/phone"])
        columns = ["time_s"]
        for name in ARMS:
            columns += [f"{name}.{kind}.{j}" for kind in ("pos", "vel")
                        for j in self.base.scene[name].joint_names]
        columns += [f"object.{path}.{k}" for path in self.bodies.prim_paths
                    for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")]
        columns += [f"holder.{kind}.{j}" for kind in ("pos", "vel") for j in self.monitor.holder.joint_names]
        self.columns = columns
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=columns,
            physics_dt=PHYSICS_DT, robot_roots_fixed=True,
            object_prim_paths=self.bodies.prim_paths, quaternion_order="xyzw"))

    def reset(self, scene, *, seed=None):
        assert scene.instruction == INSTRUCTION
        write_json(self.out / "reset.json", dict(derived_seed=seed, layout="fixed, no randomization"))
        self._ensure_sim()
        self._instruction = INSTRUCTION
        ready = self._torch.tensor([[*READY, 1., *READY, 1.]], dtype=self._torch.float32)
        table_contact_peak = 0.
        for _ in range(CONTROL_HZ // 2):
            self._env.step(ready)
            force = self.monitor.phone_contacts.get_contact_force_matrix(PHYSICS_DT)[0, 1]
            table_contact_peak = max(table_contact_peak, float(force.norm()))
        assert table_contact_peak > .1, "Resting phone did not produce table contact"
        write_json(self.out / "initial-contact-check.json", dict(table_contact_peak_N=table_contact_peak))
        self._t = 0.
        self.save_state()
        self.last_info = self.measure()
        assert self.last_info["holder_frame_margin_m"] > 0
        assert abs(self.last_info["joints_deg"][1]) < 3
        self.camera.capture(self, "initial.png")
        return self.capture()

    def save_state(self):
        row = [self.physics_ticks * PHYSICS_DT]
        for name in ARMS:
            data = self.base.scene[name].data
            row.extend(data.joint_pos.torch[0].cpu().numpy())
            row.extend(data.joint_vel.torch[0].cpu().numpy())
        bodies = np.concatenate((self.bodies.get_transforms().cpu().numpy(),
                                 self.bodies.get_velocities().cpu().numpy()), axis=1)
        row.extend(bodies.ravel())
        row.extend(self.monitor.holder.data.joint_pos.torch[0].cpu().numpy())
        row.extend(self.monitor.holder.data.joint_vel.torch[0].cpu().numpy())
        array = np.asarray(row, dtype="<f8")
        assert len(array) == len(self.columns) and np.isfinite(array).all()
        self.state_file.write(array.tobytes())

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self.base.scene[name].update(1 / CONTROL_HZ, force_recompute=True)
        return self._observation()

    def measure(self):
        m = self.monitor
        cp, cr = m.body("cradle")
        pp, pr = m.phone_pose()
        bp, br = m.body("base")
        surface_w = self.surface @ pr.T + pp
        hand_force = max(float(view.get_contact_force_matrix(PHYSICS_DT).norm(dim=-1).max())
                         for view in (self.holder_hands, self.phone_hands))
        holder_points = np.concatenate([local @ m.body(name)[1].T + m.body(name)[0]
                                        for name, local in m.holder_points.items()])
        object_points = np.concatenate((holder_points, surface_w))
        clearance = min(float(np.linalg.norm(object_points - m.tcp(side)[0], axis=1).min())
                        for side in ("left", "right"))
        phone_contacts = m.phone_contacts.get_contact_force_matrix(PHYSICS_DT)[0].cpu().numpy()
        row = dict(time_s=self._t, control_steps=self.control_steps,
            joints_deg=m.angles().tolist(), panel_angle_deg=m.panel_angle(),
            phone_pos=pp.tolist(), phone_quat=Rotation.from_matrix(pr).as_quat().tolist(),
            base_pos=bp.tolist(), cradle_pos=cp.tolist(), cradle_quat=Rotation.from_matrix(cr).as_quat().tolist(),
            holder_frame_margin_m=m.frame_margin(), holder_heading_error_deg=m.heading_error(cr),
            phone_heading_error_deg=m.heading_error(pr),
            phone_side_tilt_deg=float(np.rad2deg(np.arcsin(np.clip(pr[2, 0], -1, 1)))),
            base_tilt_deg=float(np.rad2deg(np.arccos(np.clip(br[2, 2], -1, 1)))),
            hand_contact_N=hand_force, gripper_clearance_m=clearance,
            phone_contacts_N=phone_contacts.tolist())
        row.update(score_placement(row, self.surface, m.station.TABLE_TOP, self._t))
        return row

    def step(self, action):
        assert self.control_steps < MAX_STEPS
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions_file, dict(time_s=self._t, requested=requested.tolist(), applied=target.tolist()))
        self.base.action_manager.process_action(self._torch.tensor([target.tolist()], dtype=self._torch.float32))
        for _ in range(round(1 / (CONTROL_HZ * PHYSICS_DT))):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(PHYSICS_DT)
            self.physics_ticks += 1
            self.save_state()
            if self.record and self.physics_ticks % 16 == 0:
                self.camera.capture(self)
        self.control_steps += 1
        self._t = self.control_steps / CONTROL_HZ
        self.state_file.flush()
        os.fsync(self.state_file.fileno())
        self.last_info = self.measure()
        append_json(self.measurements, self.last_info)
        if self.control_steps % (5 * CONTROL_HZ) == 0:
            write_json(self.out / "progress.json", self.last_info)
            print(f"Trial {self._t:.1f}s: panel={self.last_info['panel_angle_deg']:.1f} deg, "
                  f"phone={self.last_info['phone_angle_deg']:.1f} deg / "
                  f"{1000*self.last_info['phone_bottom_height_m']:.1f} mm, "
                  f"placed={self.last_info['checks']['phone_placed_on_holder']}, "
                  f"success={self.last_info['success']}", flush=True)
        return StepResult(observation=self.capture(), terminated=self.last_info["success"],
            termination_reason="success" if self.last_info["success"] else None,
            truncated=self.control_steps >= MAX_STEPS and not self.last_info["success"], info=self.last_info)

    def close(self):
        self.state_file.close()
        self.actions_file.close()
        self.measurements.close()
        if self.camera is not None:
            self.camera.close()
        if self._env is not None:
            self._env.close()


def check(embodiment, scene):
    from PIL import Image
    obs = embodiment.reset(scene, seed=0)
    assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"}
    assert not obs.extra and obs.instruction == INSTRUCTION
    q = obs.state["joint_pos"].copy()
    assert q.shape == (14,) and not embodiment.last_info["success"]
    for name, rgb in obs.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8 and rgb.std() > 10
        Image.fromarray(rgb).save(embodiment.out / f"initial-{name}.png")
    moved = q.copy()
    moved[0] += .12
    for _ in range(CONTROL_HZ):
        result = embodiment.step(Action(data=moved))
    changed = float(np.abs(result.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean())
    assert abs(result.observation.state["joint_pos"][0] - q[0]) > .05 and changed > 1
    assert not result.info["success"] and result.observation.state_time == 1.
    assert embodiment.physics_ticks == 480
    write_json(embodiment.out / "check.json", dict(passed=True, api_inference_calls=0,
        camera_size=[640, 360], moving_camera_rgb_mae=changed, physics_ticks=480,
        state_keys=list(obs.state), cameras=list(obs.images), max_seconds=MAX_SECONDS))
    print("OFFLINE CHECK PASSED", flush=True)


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check_only = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "output" / f"dream-{'check' if check_only else 'eval'}-{stamp}"
    out.mkdir(parents=True)
    source = out / "source"
    source.mkdir()
    for name in ("dream_eval.py", "expert.py", "yam_scene.py", "camera_ready.py", "scoring.py", "task.json"):
        shutil.copy2(HERE / name, source / name)
    shutil.copytree(HERE / "assets", source / "assets")
    write_json(out / "run.json", dict(model=MODEL, check_only=check_only, instruction=INSTRUCTION,
        max_steps=MAX_STEPS, max_seconds=MAX_SECONDS, control_hz=CONTROL_HZ, physics_dt=PHYSICS_DT,
        camera_size=[640, 360], model_input_size=[320, 176],
        seed=0, renderer="RealTimePathTracing", scorer="success_at_end", layout="fixed YAM station",
        input_state_keys=["joint_pos"], input_cameras=list(CAMERAS),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    print(f"RUN_DIRECTORY={out}", flush=True)
    embodiment = PhoneHolderEmbodiment(out, record=not check_only)
    policy = None
    started, exit_code = time.perf_counter(), 0
    try:
        scene = Scene(id="use-phoneholder", instruction=INSTRUCTION, init_seed=0)
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(out))
            task = Task(name="use-phoneholder", scenes=[scene], scorer="success_at_end", max_steps=MAX_STEPS)
            logs = evaluate(task, policy=policy, embodiment=embodiment,
                log_dir=str(out / "eval"), seed=0, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
            embodiment.camera.capture(embodiment, "final.png")
            print("EVALUATION COMPLETE", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        write_json(out / "metrics.json", dict(exit_code=exit_code, wall_total_s=time.perf_counter()-started,
            physics_rows=embodiment.physics_ticks + 1, **embodiment.last_info))
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

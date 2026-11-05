"""One archived Dream-1 trial; --check validates the adapter without inference."""

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
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment
from yam_scene import LAMP_MAX_TILT, LAMP_MIN_HEIGHT, TASK_SECONDS

ROOT = Path(__file__).resolve().parent
MODEL = "dreamlabs/dream-1-beta"
CAMERAS = ("left_cam", "top_cam", "right_cam")
CONTROL_HZ = 20
PHYSICS_DT = 1 / 480
MAX_SECONDS = TASK_SECONDS
MAX_STEPS = MAX_SECONDS * CONTROL_HZ
INSTRUCTION = (
    "Unfold the desk lamp and turn it on to illuminate the closed book within 120 seconds. "
    "With the right arm holding the base flat on the table, first turn the folded lamp "
    "toward the book and slide its entire base into the red tape rectangle behind and "
    "to the left of the book. With the left arm, grasp the lamp head from above near "
    "its front end, closer to the right hand, and pull straight upward. Raise the entire "
    f"bottom of the lamp shade at least {LAMP_MIN_HEIGHT*100:g} cm above the table, with "
    f"the emitting face pointing down within {LAMP_MAX_TILT:g} degrees of horizontal. Keep the right hand "
    "holding the base throughout these adjustments; release it only to tap the sun "
    "symbol on the base with the right hand. One tap turns the lamp on; another turns "
    "it off. Finish with the lamp on and its entire base inside the tape. "
    "Avoid collisions between the arms."
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
        print(f"Dream reply at {(self.tick-1)/CONTROL_HZ:.2f}s: "
              f"{len(chunk.rows)} actions, {chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class LampEmbodiment(YamArenaEmbodiment):
    def __init__(self, out, record):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=CONTROL_HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_foldable_lamp", supported_target_kinds=frozenset(),
            environment_id="yam-foldable-lamp-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out, self.record, self.app, self.camera, self.monitor = out, record, None, None, None
        self.physics_ticks = self.control_steps = 0
        self.last_info = {}
        self.max_base_lift = self.max_base_tilt = 0.
        self.states = (out / "physics-f64.bin").open("xb")
        self.actions = (out / "executed-actions.jsonl").open("x")
        self.measurements = (out / "measurements.jsonl").open("x")

    def _ensure_sim(self):
        assert self._env is None, "One episode per archive"
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=True,
            kit_args="--/carb/tasking/plugin/threadCount=12 "
                     "--/omni/tbb/globalcontrol/maxThreadCount=12").app
        import torch
        import expert
        from yam_arena.robot import asset_link_paths
        from yam_scene import make_env

        self._env = make_env(control_hz=CONTROL_HZ, enable_cameras=True, log_dir=self.out / "logs")
        self._base = self.base = self._env.unwrapped
        self._torch = torch
        self._ids = {name: (
            self.base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self.base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        self._env.reset()
        # Only measurement, touch and recording helpers are reused. No expert motion is called.
        self.monitor = expert.Expert(self._env, self.out)
        self.monitor.phase = "Dream-1 evaluation"
        assert not self.monitor.lamp.is_fixed_base
        passive = self.monitor.physics.create_articulation_view("/World/envs/env_0/foldable_lamp/base")
        assert passive.count == 1 and bool((passive.get_dof_stiffnesses() == 0).all())
        assert bool((passive.get_dof_dampings() == 0).all())
        paths = asset_link_paths()
        hands = [f"/World/envs/env_0/{side}Arm{paths[name]}" for side in ("Left", "Right")
                 for name in ("gripper", "tip_left", "tip_right")]
        self.hand_contacts = self.monitor.physics.create_rigid_contact_view(
            "/World/envs/env_0/foldable_lamp/*", hands)
        self.table_contacts = self.monitor.physics.create_rigid_contact_view(
            "/World/envs/env_0/foldable_lamp/base", ["/World/envs/env_0/table/geometry/mesh"])
        self.bodies = self.monitor.physics.create_rigid_body_view([
            "/World/envs/env_0/foldable_lamp/*", "/World/envs/env_0/book"])
        columns = ["time_s"]
        for name in ARMS:
            columns += [f"{name}.{kind}.{j}" for kind in ("pos", "vel")
                        for j in self.base.scene[name].joint_names]
        columns += [f"object.{path}.{k}" for path in self.bodies.prim_paths
                    for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")]
        columns += [f"lamp.{kind}.{j}" for kind in ("pos", "vel") for j in self.monitor.lamp.joint_names]
        columns += ["lamp_on"]
        self.columns = columns
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=columns,
            physics_dt=PHYSICS_DT, robot_roots_fixed=True, quaternion_order="xyzw",
            object_prim_paths=self.bodies.prim_paths))
        expert.RECORD_VIDEO = self.record
        self.camera = expert.Recorder(self.base.sim.stage, self.out)

    def reset(self, scene, *, seed=None):
        assert scene.instruction == INSTRUCTION
        self._ensure_sim()
        self._instruction = INSTRUCTION
        ready = self._torch.tensor([[*READY, 1., *READY, 1.]], dtype=self._torch.float32)
        peak = 0.
        for _ in range(CONTROL_HZ):
            self._env.step(ready)
            peak = max(peak, float(self.table_contacts.get_contact_force_matrix(PHYSICS_DT).norm()))
        assert peak > .1, "Initial table contact reporting failed"
        self.monitor.switch.time = 0.
        self.monitor.arm_contacts.clear()
        self.monitor.head_body_contacts.clear()
        self._t = 0.
        self.initial_book = self.base.scene["book"].data.root_link_pose_w.torch[0].cpu().numpy().copy()
        self.save_state()
        self.last_info = self.measure()
        assert not self.last_info["success"] and not self.last_info["lamp_on"]
        assert np.max(np.abs(self.monitor.angles())) < 3
        write_json(self.out / "reset.json", dict(seed=seed, randomization=False,
            table_contact_peak_N=peak, initial=self.last_info))
        self.camera.capture(self.monitor, "initial.png")
        return self.capture()

    def save_state(self):
        row = [self.physics_ticks * PHYSICS_DT]
        for name in ARMS:
            data = self.base.scene[name].data
            row.extend(data.joint_pos.torch[0].cpu().numpy())
            row.extend(data.joint_vel.torch[0].cpu().numpy())
        row.extend(np.concatenate((self.bodies.get_transforms().cpu().numpy(),
                   self.bodies.get_velocities().cpu().numpy()), axis=1).ravel())
        row.extend(self.monitor.lamp.data.joint_pos.torch[0].cpu().numpy())
        row.extend(self.monitor.lamp.data.joint_vel.torch[0].cpu().numpy())
        row.append(self.monitor.switch.state.Get())
        array = np.asarray(row, dtype="<f8")
        assert len(array) == len(self.columns) and np.isfinite(array).all()
        self.states.write(array.tobytes())

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self.base.scene[name].update(1 / CONTROL_HZ, force_recompute=True)
        return self._observation()

    def measure(self):
        m = self.monitor
        bp, br = m.body("base")
        placement, alignment = m.placement(), m.alignment()
        height = m.head_height()
        book = self.base.scene["book"].data.root_link_pose_w.torch[0].cpu().numpy()
        displacement = float(np.linalg.norm(book[:3] - self.initial_book[:3]))
        hand_force = float(self.hand_contacts.get_contact_force_matrix(PHYSICS_DT).norm(dim=-1).max())
        tilt = float(np.rad2deg(np.arccos(np.clip(br[2, 2], -1, 1))))
        self.max_base_lift = max(self.max_base_lift, float(bp[2] - m.station.TABLE_TOP))
        self.max_base_tilt = max(self.max_base_tilt, tilt)
        events = m.switch.events
        right_touch = len(events) == 1 and events[0]["on"] and any("RightArm" in p for p in events[0]["colliders"])
        clearance = min(float(np.linalg.norm(m.tcp(side)[0] - m.body(name)[0]))
                        for side in ("left", "right") for name in m.links)
        checks = dict(base_inside_frame=placement["footprint_inside"],
            lamp_bottom_high_enough=height >= LAMP_MIN_HEIGHT,
            lamp_face_horizontal=alignment["lamp_face_from_horizontal_deg"] <= LAMP_MAX_TILT,
            lamp_on=bool(m.switch.state.Get()), within_time_limit=self._t <= MAX_SECONDS)
        row = dict(time_s=self._t, control_steps=self.control_steps, base_pos=bp.tolist(),
            joints_deg=m.angles().tolist(), lamp_on=bool(m.switch.state.Get()),
            placement=placement, alignment=alignment, book_displacement_m=displacement,
            lamp_bottom_height_m=height, right_hand_turned_on=bool(right_touch),
            hand_contact_N=hand_force, hand_clearance_m=clearance,
            max_base_lift_m=self.max_base_lift, max_base_tilt_deg=self.max_base_tilt,
            inter_arm_contact_count=len(m.arm_contacts))
        row.update(checks=checks, success=bool(all(checks.values())), score=int(all(checks.values())))
        return row

    def step(self, action):
        assert self.control_steps < MAX_STEPS
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions, dict(step=self.control_steps, time_s=self._t,
            requested=requested.tolist(), applied=target.tolist()))
        self.base.action_manager.process_action(self._torch.tensor(target[None], dtype=self._torch.float32))
        for _ in range(round(1 / (CONTROL_HZ * PHYSICS_DT))):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(PHYSICS_DT)
            self.physics_ticks += 1
            self.save_state()
            if self.record and self.physics_ticks % 16 == 0:
                self.monitor.step = self.physics_ticks // 16
                self.camera.capture(self.monitor)
        self.control_steps += 1
        self._t = self.control_steps / CONTROL_HZ
        self.states.flush()
        os.fsync(self.states.fileno())
        self.last_info = self.measure()
        append_json(self.measurements, self.last_info)
        if self.control_steps % (5 * CONTROL_HZ) == 0:
            write_json(self.out / "progress.json", self.last_info)
            print(f"Trial {self._t:.1f}s: joints={np.round(self.monitor.angles(), 1)}, "
                  f"lamp_on={self.last_info['lamp_on']}, success={self.last_info['success']}", flush=True)
        return StepResult(observation=self.capture(), terminated=self.last_info["success"],
            termination_reason="success" if self.last_info["success"] else None,
            truncated=self.control_steps >= MAX_STEPS and not self.last_info["success"], info=self.last_info)

    def close(self):
        for file in (self.states, self.actions, self.measurements):
            file.close()
        if self.monitor is not None:
            write_json(self.out / "touch-events.json", self.monitor.switch.events)
            write_json(self.out / "arm-contacts.json", self.monitor.arm_contacts)
            self.monitor.arm_subscription = None
            self.monitor.subscription = None
            self.monitor.switch.close()
        if self.camera is not None:
            self.camera.close()
            if self.record:
                (self.out / "videos/expert.mp4").rename(self.out / "videos/dream.mp4")
        if self._env is not None:
            self._env.close()


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / f"dream-{'check' if check else 'eval'}-{stamp}"
    source = out / "source"
    source.mkdir(parents=True)
    for name in ("dream_eval.py", "expert.py", "yam_scene.py", "model.py", "touch.py"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(ROOT / "assets", source / "assets")
    write_json(out / "run.json", dict(model=MODEL, check_only=check, instruction=INSTRUCTION,
        max_seconds=MAX_SECONDS, max_steps=MAX_STEPS, control_hz=CONTROL_HZ, physics_dt=PHYSICS_DT,
        min_lamp_bottom_height_m=LAMP_MIN_HEIGHT, max_lamp_face_tilt_deg=LAMP_MAX_TILT,
        camera_size=[640, 360], model_input_size=[320, 176], seed=0,
        renderer="RealTimePathTracing", scorer="success_at_end", input_state_keys=["joint_pos"],
        input_cameras=list(CAMERAS), api_wait_advances_physics=False,
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob("*") if p.is_file()},
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    print(f"RUN_DIRECTORY={out}", flush=True)
    embodiment, policy = LampEmbodiment(out, record=not check), None
    started, exit_code = time.perf_counter(), 0
    try:
        scene = Scene(id="foldable-lamp", instruction=INSTRUCTION, init_seed=0)
        if check:
            obs = embodiment.reset(scene, seed=0)
            assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"} and not obs.extra
            assert all(rgb.shape == (360, 640, 3) and rgb.std() > 10 for rgb in obs.images.values())
            q = obs.state["joint_pos"].copy()
            target = q.copy()
            target[0] += .12
            for _ in range(CONTROL_HZ):
                step = embodiment.step(Action(data=target))
            changed = float(np.abs(step.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean())
            assert abs(step.observation.state["joint_pos"][0] - q[0]) > .05 and changed > 1
            assert embodiment.physics_ticks == 480 and step.observation.state_time == 1.
            assert not step.info["success"]
            write_json(out / "check.json", dict(passed=True, api_calls=0,
                moving_camera_rgb_mae=changed, physics_ticks=480, max_seconds=MAX_SECONDS))
            print("OFFLINE CHECK PASSED", flush=True)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(out))
            task = Task(name="foldable-lamp", scenes=[scene], scorer="success_at_end", max_steps=MAX_STEPS)
            logs = evaluate(task, policy=policy, embodiment=embodiment,
                log_dir=str(out / "eval"), seed=0, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
            embodiment.camera.capture(embodiment.monitor, "final.png")
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

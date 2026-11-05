"""One archived Dream-1 trial; --check validates the adapter without API calls."""

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
from inspect_robots import Action, Observation, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.inspect_embodiment import JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

import task

HERE = Path(__file__).resolve().parent
MODEL = "dreamlabs/dream-1-beta"
CAMERAS = {"left_cam": "left_wrist", "top_cam": "top", "right_cam": "right_wrist"}


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
        assert set(frames) == set(CAMERAS) and prompt == task.INSTRUCTION
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
        print(f"Saved Dream reply at tick {self.tick - 1}: {len(chunk.rows)} rows, "
              f"{chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class LaptopEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=task.CONTROL_HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_laptop_holder", supported_target_kinds=frozenset(),
            environment_id="yam-use-laptopholder-highest-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], task.INSTRUCTION)))
        self.out, self.app, self.env, self.camera = out, None, None, None
        self.physics_rows = 0
        self.last_info = {}
        self.best_score = 0.
        self.state_file = (out / "physics-f64.bin").open("xb")
        self.actions_file = (out / "executed-actions.jsonl").open("x")
        self.measurements = (out / "measurements.jsonl").open("x")

    def _ensure_sim(self):
        if self.env is not None:
            return
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=True,
            kit_args="--/plugins/carb.tasking.plugin/threadCount=8 "
                     "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
        from benchmark import LaptopHolderTask
        from camera_ready import Recorder
        self.env = LaptopHolderTask(cameras=False)
        self.camera = Recorder(self.env, self.out / "camera_setup", record=False)
        columns = ["time_s"]
        for side, robot in zip(("left", "right"), self.env.robots):
            columns += [f"{side}.{kind}.{name}" for kind in ("pos", "vel") for name in robot.joint_names]
        columns += [f"holder.root.{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        columns += [f"holder.root_vel.{k}" for k in ("vx", "vy", "vz", "wx", "wy", "wz")]
        columns += [f"holder.{kind}.{i}" for kind in ("dof_pos", "dof_vel")
                    for i in range(self.env.art.max_dofs)]
        columns += [f"holder.{name}.{k}" for name in self.env.holder.names
                    for k in ("x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz")]
        self.columns = columns
        write_json(self.out / "replay-schema.json", dict(dtype="<f8", columns=columns,
            physics_dt=self.env.sim.cfg.dt, holder_body_order=self.env.holder.names,
            holder_articulation_path="/World/Holder/links/deck_L", robot_roots_fixed=True,
            cameras=self.camera.cameras, observer_fixed_world_transform=self.camera.observer_pose.tolist()))

    def reset(self, scene, *, seed=None):
        assert scene.instruction == task.INSTRUCTION, "Unexpected task prompt"
        assert self.physics_rows == 0, "One trial per archive"
        write_json(self.out / "reset.json", dict(derived_seed=seed, layout="fixed, no randomization"))
        self._ensure_sim()
        self.env.reset()
        self.save_state()
        self.env.physics_callback = self.save_state
        return self.capture()

    def save_state(self):
        env = self.env
        row = [self.physics_rows * env.sim.cfg.dt]
        for robot in env.robots:
            row.extend(robot.data.joint_pos.torch[0].cpu().numpy())
            row.extend(robot.data.joint_vel.torch[0].cpu().numpy())
        row.extend(env.art.get_root_transforms()[0])
        row.extend(env.art.get_root_velocities()[0])
        row.extend(env.art.get_dof_positions()[0])
        row.extend(env.art.get_dof_velocities()[0])
        bodies = np.concatenate((env.holder.bodies.get_transforms(), env.holder.bodies.get_velocities()), axis=1)
        row.extend(bodies.ravel())
        array = np.asarray(row, dtype="<f8")
        assert len(array) == len(self.columns) and np.isfinite(array).all()
        self.state_file.write(array.tobytes())
        self.physics_rows += 1

    def capture(self):
        views = self.camera.read_views()
        images = {name: views[source] for name, source in CAMERAS.items()}
        return Observation(images=images, state={"joint_pos": self.env.joint_positions()},
            instruction=task.INSTRUCTION, image_times=dict.fromkeys(images, self.env.steps / task.CONTROL_HZ),
            state_time=self.env.steps / task.CONTROL_HZ)

    def metrics(self):
        if self.env is None:
            return dict(physics_rows=self.physics_rows)
        return dict(**self.last_info, time_s=self.env.steps / task.CONTROL_HZ,
            control_steps=self.env.steps, physics_rows=self.physics_rows,
            best_partial_score=self.best_score, holder=self.env.holder.measure())

    def step(self, action):
        requested = np.asarray(action.data, dtype=np.float32)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = requested.copy()
        for i, (robot, (arm_ids, _)) in enumerate(zip(self.env.robots, self.env.ids)):
            limits = robot.data.joint_pos_limits.torch[0, arm_ids].cpu().numpy()
            target[i*7:i*7+6] = np.clip(target[i*7:i*7+6], limits[:, 0], limits[:, 1])
            target[i*7+6] = np.clip(target[i*7+6], 0, 1)
        append_json(self.actions_file, dict(step=self.env.steps, time_s=self.env.steps / task.CONTROL_HZ,
                    requested=requested.tolist(), applied=target.tolist()))
        _, self.last_info = self.env.step(target)
        self.state_file.flush()
        os.fsync(self.state_file.fileno())
        self.best_score = max(self.best_score, self.last_info["score"])
        metrics = self.metrics()
        append_json(self.measurements, metrics)
        if self.env.steps % (5 * task.CONTROL_HZ) == 0:
            write_json(self.out / "progress.json", metrics)
            print(f"Trial {metrics['time_s']:.1f}s: score={metrics['score']:.2f}, "
                  f"width={metrics['holder']['width']*1000:.1f} mm, "
                  f"seated={metrics['seated']}, contained={metrics['contained']}", flush=True)
        return StepResult(observation=self.capture(), terminated=self.last_info["success"],
            termination_reason="success" if self.last_info["success"] else None,
            truncated=self.last_info["truncated"], info=metrics)

    def close(self):
        self.state_file.close()
        self.actions_file.close()
        self.measurements.close()
        if self.camera is not None:
            self.camera.close()
        if self.env is not None:
            self.env.physics_callback = None
            self.env.sim.stop()


def check(embodiment, scene):
    from PIL import Image
    from inspect_robots.rollout import derive_seed
    obs = embodiment.reset(scene, seed=derive_seed(0, scene.init_seed, 0))
    assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"}
    assert not obs.extra and obs.instruction == task.INSTRUCTION
    q = obs.state["joint_pos"].copy()
    assert q.shape == (14,) and embodiment.env.holder.measure()["width"] < .06
    for name, rgb in obs.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.dtype == np.uint8 and rgb.std() > 10
        Image.fromarray(rgb).save(embodiment.out / f"initial-{name}.png")
    moved = q.copy()
    moved[0] += .12
    for _ in range(20):
        result = embodiment.step(Action(data=moved))
    assert abs(result.observation.state["joint_pos"][0] - q[0]) > .05
    changed = np.abs(result.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean()
    assert changed > 1 and not result.info["success"] and result.info["score"] == 0
    assert embodiment.physics_rows == 1 + round(1 / embodiment.env.sim.cfg.dt)
    assert result.observation.state_time == 1.
    embodiment.camera.checkpoint("01 Adapter movement check")
    write_json(embodiment.out / "check.json", dict(passed=True, api_inference_calls=0,
        camera_size=[640, 360], moving_camera_rgb_mae=changed, physics_rows=embodiment.physics_rows,
        state_keys=list(obs.state), cameras=list(obs.images)))
    print("OFFLINE CHECK PASSED", flush=True)


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check_only = bool(sys.argv[1:])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "outputs" / f"dream-{'check' if check_only else 'highest'}-{stamp}"
    out.mkdir(parents=True)
    (out / "videos").mkdir()
    source = out / "source"
    source.mkdir()
    for path in HERE.glob("*.py"):
        shutil.copy2(path, source / path.name)
    shutil.copytree(HERE / "assets", source / "assets")
    write_json(out / "run.json", dict(model=MODEL, check_only=check_only, instruction=task.INSTRUCTION,
        target_gear=task.TARGET_GEAR, max_steps=task.MAX_STEPS, control_hz=task.CONTROL_HZ,
        camera_size=[640, 360], model_input_size=[320, 176], seed=0,
        renderer="RealTimePathTracing", scorer="success_at_end", layout="fixed, calibrated YAM station",
        input_state_keys=["joint_pos"], input_cameras=list(CAMERAS),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    print(f"RUN_DIRECTORY={out}", flush=True)
    embodiment = LaptopEmbodiment(out)
    policy = None
    started = time.perf_counter()
    exit_code = 0
    try:
        scene = Scene(id="unfold-laptop-holder-highest", instruction=task.INSTRUCTION, init_seed=0)
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=MODEL, session=SavedDreamSession(out))
            evaluation = Task(name="use-laptopholder-highest", scenes=[scene],
                              scorer="success_at_end", max_steps=task.MAX_STEPS)
            logs = evaluate(evaluation, policy=policy, embodiment=embodiment,
                log_dir=str(out / "eval"), seed=0, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
            print(f"EVALUATION COMPLETE: {len(logs)} log(s)", flush=True)
            embodiment.camera.checkpoint("02 Final")
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        write_json(out / "metrics.json", dict(exit_code=exit_code, wall_total_s=time.perf_counter()-started,
                                             **embodiment.metrics()))
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

"""Run one archived Dream-1 trial, or validate the adapter without inference."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from scene import CONFIG

HERE = Path(__file__).resolve().parent
CAMERAS = tuple(CONFIG["observation"]["camera_names"])
HZ = CONFIG["simulation"]["control_hz"]
INSTRUCTION = (
    "Wait for the desk timer to count down to zero. Use the left wrist camera to read "
    "the temperature of the thermometer suspended in the beaker. Then close the right "
    "gripper and use its fingertips to enter the measured temperature in degrees Celsius "
    "on the calculator, followed by the equals key. The timer is already running. "
    "Keep the thermometer suspended in the water. Complete the task within 120 seconds."
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
        append_json(self.observations, {"tick": self.tick, "time_s": timestamp,
            "frames": paths, "joint_pos": state.tolist(), "prompt": prompt})
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        append_json(self.replies, {"observation_tick": self.tick-1,
            "commanded": None if commanded is None else commanded.tolist(), **chunk.to_json()})
        print(f"Dream reply at {(self.tick-1)/HZ:.2f}s: {len(chunk.rows)} actions, "
              f"{chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class ThermometerEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device="cpu", cam_width=640, cam_height=360,
                         control_hz=HZ, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_read_thermometer",
            supported_target_kinds=frozenset(), environment_id="yam-read-thermometer-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out, self.sim = out, None
        self.actions = (out / "executed-actions.jsonl").open("x")
        self.last_info = {}

    def reset(self, scene, *, seed=None):
        from simulation import Simulation
        assert self.sim is None, "One episode per archive"
        # The evaluator derives an epoch seed; this task uses its fixed configured reset.
        assert scene.init_seed == CONFIG["episode"]["seed"]
        self.sim = Simulation(self.out)
        self._env, self._base = self.sim.env, self.sim.base
        self._ids = dict(zip(ARMS, self.sim.ids))
        self._instruction = scene.instruction
        self._t = 0.
        self.last_info = self.sim.runtime.state()
        for _ in range(4):
            observation = self.capture()
        assert self.last_info["score"] == 0
        return observation

    def capture(self):
        self.sim.capture()
        return self._observation()

    def step(self, action):
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        target = np.clip(requested, self._low, self._high)
        append_json(self.actions, {"time_s": self._t, "requested": requested.tolist(),
                                  "applied": target.tolist()})
        self.sim.step(target)
        self._t = self.sim.time
        self.last_info = self.sim.runtime.state()
        if self.sim.steps % (5 * HZ) == 0:
            write_json(self.out / "progress.json", self.last_info)
            print("TRIAL", self.last_info, flush=True)
        success = bool(self.last_info["score"])
        return StepResult(observation=self.capture(), terminated=success,
            termination_reason="success" if success else None,
            truncated=self.sim.steps >= CONFIG["episode"]["max_control_steps"] and not success,
            info=self.last_info)

    def close(self):
        self.actions.close()
        if self.sim is not None:
            self.sim.close()


class FinalScore:
    name = "temperature_submission"

    def __call__(self, record, target):
        from inspect_robots.scorer import Score
        info = record.steps[-1].result.info
        return Score(value=info["score"], explanation="The calculator's submitted result equals 45 degrees Celsius.",
                     metadata={"submitted": info["submitted"]})


def check(embodiment, scene):
    from PIL import Image
    obs = embodiment.reset(scene, seed=CONFIG["episode"]["seed"])
    assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"} and not obs.extra
    for name, rgb in obs.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.std() > 10
        Image.fromarray(rgb).save(embodiment.out / (name + "-initial.png"))
    target = obs.state["joint_pos"].copy()
    target[0] += .12
    for _ in range(HZ):
        result = embodiment.step(Action(data=target))
    difference = float(np.abs(result.observation.images["left_cam"].astype(float) - obs.images["left_cam"]).mean())
    assert abs(result.observation.state["joint_pos"][0] - obs.state["joint_pos"][0]) > .05
    assert difference > 1 and embodiment.sim.ticks == 240 and result.observation.state_time == 1.
    assert result.info["score"] == 0
    write_json(embodiment.out / "check.json", {"passed": True, "api_calls": 0,
        "moving_camera_rgb_mae": difference, "physics_ticks": 240, "score": 0})
    print("OFFLINE CHECK PASSED", flush=True)


def main():
    assert sys.argv[1:] in ([], ["--check"]), "Usage: python dream_eval.py [--check]"
    check_only = bool(sys.argv[1:])
    if not check_only:
        assert (Path.home() / ".dreamlabs/credentials").is_file(), "Configure ~/.dreamlabs/credentials"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / "runs" / f"dream-{'check' if check_only else 'eval'}-{stamp}"
    out.mkdir(parents=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    from simulation import archive_source
    archive_source(out)
    write_json(out / "run.json", {"instruction": INSTRUCTION, "config": CONFIG,
        "check_only": check_only, "model": CONFIG["model"],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}})
    print("RUN_DIRECTORY", out, flush=True)
    embodiment = ThermometerEmbodiment(out)
    policy, error, exit_code = None, None, 0
    started = time.perf_counter()
    try:
        scene = Scene(id="read-thermometer", instruction=INSTRUCTION, init_seed=CONFIG["episode"]["seed"])
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=CONFIG["model"], session=SavedDreamSession(out))
            task = Task(name="read-thermometer", scenes=[scene], scorer=FinalScore(),
                        max_steps=CONFIG["episode"]["max_control_steps"])
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out / "eval"),
                seed=CONFIG["episode"]["seed"], fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
    except BaseException:
        error, exit_code = traceback.format_exc(), 1
        traceback.print_exc()
    finally:
        if embodiment.sim is not None:
            embodiment.sim.save("success" if error is None else "failed", error)
        write_json(out / "metrics.json", {"exit_code": exit_code, "error": error,
            "wall_time_s": time.perf_counter()-started, **embodiment.last_info})
        if policy is not None:
            write_json(out / "dream-transcript.json", policy.transcript())
            policy.close()
        embodiment.close()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

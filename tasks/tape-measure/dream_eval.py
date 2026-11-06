"""Evaluate one Dream episode or check the adapter without inference."""

import base64
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from inspect_robots import Action, Scene, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from scene import CONFIG

CAMERAS = ("left_cam", "top_cam", "right_cam")
INSTRUCTION = (
    "Use the tape measure on the right to measure the longest edge of the wooden "
    "bar behind it. Enter the measured length as an integer in millimetres on the "
    "calculator on the left, then press = to submit and finish. The tape measure "
    "retracts automatically when released; it has no lock. Complete the task "
    "within 120 seconds."
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def append_json(file, value):
    file.write(json.dumps(value, allow_nan=False)+"\n")
    file.flush()
    os.fsync(file.fileno())


class SavedDreamSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out = out/"dream"
        self.out.mkdir()
        self.tick = 0
        self.observations = (self.out/"observations.jsonl").open("x")
        self.replies = (self.out/"replies.jsonl").open("x")
        for name in CAMERAS:
            (self.out/name).mkdir()

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        paths = {}
        for name, jpeg in frames.items():
            path = self.out/name/f"{self.tick:06d}.jpg"
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
        append_json(self.replies, dict(observation_tick=self.tick-1,
            commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print("Dream reply", self.tick-1, len(chunk.rows), "actions", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class TapeEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        super().__init__(device=CONFIG["simulation"]["device"], cam_width=640,
            cam_height=360, control_hz=20, scene="station", object_jitter=0)
        self.info = replace(self.info, name="yam_tape_measure",
            supported_target_kinds=frozenset(), environment_id="yam-tape-measure-v1",
            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out, self.sim = out, None

    def reset(self, scene, *, seed=None):
        from simulation import Simulation
        assert self.sim is None and scene.init_seed == CONFIG["episode"]["seed"]
        write_json(self.out/"reset.json", dict(framework_seed=seed,
            scene_seed=scene.init_seed, physics_seed=CONFIG["episode"]["seed"]))
        self.sim = Simulation(self.out, cameras=True)
        self.sim.phase = "dream"
        self._base, self._env = self.sim.base, self.sim.env
        self._ids = {name: (
            self._base.scene[name].find_joints(ARM_JOINTS, preserve_order=True)[0],
            self._base.scene[name].find_joints(FINGER_JOINTS, preserve_order=True)[0]) for name in ARMS}
        self._instruction = scene.instruction
        for _ in range(8):
            observation = self.capture()
        return observation

    def capture(self):
        from render import refresh
        refresh(self.sim)
        for name in CAMERAS:
            self._base.scene[name].update(self.sim.control_dt, force_recompute=True)
        self._t = self.sim.time
        return self._observation()

    def step(self, action):
        requested = np.asarray(action.data, dtype=np.float64)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        self.sim.step(np.clip(requested, self._low, self._high))
        info = self.sim.instruments.result(self.sim.time)
        ended = self.sim.instruments.calculator.ended
        if self.sim.steps % 100 == 0:
            print("Trial", round(self.sim.time, 2), "score", info["score"], flush=True)
        return StepResult(observation=self.capture(), terminated=ended,
            termination_reason="submitted" if ended else None,
            truncated=self.sim.steps >= CONFIG["episode"]["max_control_steps"] and not ended,
            info=info)

    def close(self):
        if self.sim is not None:
            self.sim.close()


class FinalScore:
    name = "submitted_length"

    def __call__(self, record, target):
        from inspect_robots.scorer import Score
        info = record.steps[-1].result.info
        return Score(value=info["score"], explanation="Correct integer millimetres submitted with = within 120 seconds.",
                     metadata={key: info[key] for key in ("submitted", "submission_time_s", "length_mm")})


def check(embodiment, scene):
    from PIL import Image
    from inspect_robots.rollout import derive_seed
    observation = embodiment.reset(scene, seed=derive_seed(CONFIG["episode"]["seed"], scene.init_seed, 0))
    assert set(observation.images) == set(CAMERAS)
    assert set(observation.state) == {"joint_pos"} and not observation.extra
    for name, rgb in observation.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.std() > 5
        Image.fromarray(rgb).save(embodiment.out/(name+".png"))
    q = observation.state["joint_pos"].copy()
    target = q.copy()
    target[0] += .12
    for _ in range(20):
        result = embodiment.step(Action(data=target))
    difference = np.abs(result.observation.images["left_cam"].astype(float)-observation.images["left_cam"]).mean()
    assert abs(result.observation.state["joint_pos"][0]-q[0]) > .05 and difference > 1
    assert embodiment.sim.ticks == 240 and result.observation.state_time == 1.
    assert result.info["score"] == 0.
    write_json(embodiment.out/"check.json", dict(passed=True, api_calls=0,
        moving_camera_rgb_mae=float(difference), physics_ticks=240, initial_and_idle_score=0))


def run(out, check_only=False):
    from simulation import archive_source
    assert not CONFIG["episode"]["advance_physics_during_api_wait"]
    if not check_only:
        assert (Path.home()/".dreamlabs/credentials").is_file(), "Configure ~/.dreamlabs/credentials"
    source_hashes = archive_source(out)
    write_json(out/"run.json", dict(instruction=INSTRUCTION, config=CONFIG, check_only=check_only,
        input_state_keys=["joint_pos"], input_cameras=CAMERAS, model_input_size_wh=[320, 176],
        source_sha256=source_hashes,
        versions={name: importlib.metadata.version(name) for name in
                  ("inspect-robots", "inspect-robots-dream", "dreamlabs-client")}))
    embodiment = TapeEmbodiment(out)
    scene = Scene(id="tape-measure", instruction=INSTRUCTION, init_seed=CONFIG["episode"]["seed"])
    policy = None
    try:
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=CONFIG["model"], session=SavedDreamSession(out))
            task = Task(name="tape-measure", scenes=[scene], scorer=FinalScore(),
                        max_steps=CONFIG["episode"]["max_control_steps"])
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out/"eval"),
                seed=CONFIG["episode"]["seed"], fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
    except Exception as error:
        if embodiment.sim is not None:
            embodiment.sim.save("error", str(error))
        raise
    else:
        embodiment.sim.save("check_passed" if check_only else "success")
    finally:
        if policy is not None:
            write_json(out/"dream-transcript.json", policy.transcript())
            policy.close()
        embodiment.close()

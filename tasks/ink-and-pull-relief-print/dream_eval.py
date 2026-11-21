"""Evaluate Dream-1 via Inspect Robots using only YAM camera images and joints."""
import base64
from dataclasses import replace
import json
import os
import traceback

import numpy as np
from inspect_robots import Scene, StepResult, Task, eval as evaluate
from inspect_robots.scorer import Score
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from scene import CAMERAS, CONFIG, HZ, INSTRUCTION, MAX_STEPS, SEED
from simulation import Simulation, write_json


class RecordedSession(DreamSession):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        folder.mkdir()
        self.tick = 0
        self.transcript = (folder / "exchange.jsonl").open("x")

    def write(self, item):
        self.transcript.write(json.dumps(item, allow_nan=False) + "\n")
        self.transcript.flush()
        os.fsync(self.transcript.fileno())

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        files = {}
        for name, jpeg in frames.items():
            filename = f"{self.tick:06d}-{name}.jpg"
            (self.folder / filename).write_bytes(base64.b64decode(jpeg))
            files[name] = filename
        self.write({"type":"observation", "step":self.tick, "time_s":timestamp,
                    "frames":files, "joint_pos":state.tolist(), "prompt":prompt})
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.write({"type":"action_reply", **chunk.to_json()})
        print("DREAM REPLY", self.tick, len(chunk.rows), chunk.rtt_s, flush=True)
        return chunk

    def close(self):
        super().close()
        self.transcript.close()


class ReliefPrintEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        width, height = CONFIG["observation"]["render_size_wh"]
        super().__init__(device=CONFIG["simulation"]["device"], cam_width=width, cam_height=height,
                         control_hz=HZ, scene=CONFIG["station"]["scene"],
                         object_jitter=CONFIG["episode"]["object_jitter_m"])
        self.info = replace(self.info, name="yam_relief_print", supported_target_kinds=frozenset(),
                            environment_id="yam-relief-print-v1",
                            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))
        self.out = out
        self.sim = None

    def reset(self, scene, *, seed=None):
        assert self.sim is None and scene.init_seed == SEED
        self.sim = Simulation(self.out, cameras=True)
        self._base = self.sim.base
        self._ids = dict(zip(ARMS, self.sim.ids, strict=True))
        self._instruction = scene.instruction
        self._t = 0.
        for _ in range(5):
            self.sim.capture()
        observation = self._observation()
        assert set(observation.images) == set(CAMERAS) and set(observation.state) == {"joint_pos"}
        assert observation.state["joint_pos"].shape == (14,) and not observation.extra
        assert all(img.shape == (360, 640, 3) and img.std() > 10 for img in observation.images.values())
        assert self.sim.info["score"] == 0.
        return observation

    def step(self, action):
        self.sim.step(np.asarray(action.data, dtype=np.float32))
        self._t = self.sim.time
        self.sim.capture()
        done = self.sim.info["score"] == 1.
        if self.sim.steps % (5*HZ) == 0:
            print("TRIAL", json.dumps(self.sim.info), flush=True)
        return StepResult(observation=self._observation(), terminated=done,
                          termination_reason="success" if done else None,
                          truncated=self.sim.steps >= MAX_STEPS and not done,
                          info=self.sim.info.copy())

    def close(self):
        if self.sim is not None:
            self.sim.close()


class FinalPrintScore:
    name = "relief_print"

    def __call__(self, record, target):
        result = record.steps[-1].result.info
        return Score(value=result["score"],
                     explanation="Coverage and excess ink of the actual pulled sheet.",
                     metadata=result)


def run_dream(out):
    assert not CONFIG["episode"]["advance_physics_during_api_wait"]
    embodiment = ReliefPrintEmbodiment(out)
    scene = Scene(id="relief-print", instruction=INSTRUCTION, init_seed=SEED)
    task = Task(name="relief-print", scenes=[scene], scorer=FinalPrintScore(), max_steps=MAX_STEPS)
    error = None
    try:
        with DreamPolicy(model=CONFIG["model"], session=RecordedSession(out / "dream")) as policy:
            try:
                logs = evaluate(task, policy=policy, embodiment=embodiment,
                                log_dir=str(out/"eval"), seed=SEED, fail_on_error=True,
                                store_frames=False, store_actions=True)
                assert len(logs) == 1 and logs[0].status == "success"
            finally:
                write_json(out / "dream-transcript.json", policy.transcript())
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        if embodiment.sim is not None:
            embodiment.sim.save("dream", "success" if error is None else "failed", error)
        embodiment.close()

"""Run one Dream-1 trial through Inspect Robots and archive the executed evidence."""
import base64
from dataclasses import replace
import json
import sys
import traceback

import numpy as np
from inspect_robots import Scene, StepResult, Task, eval as evaluate
from inspect_robots.scorer import Score
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.inspect_embodiment import ARMS, JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from geometry import HERE
from scene import CAMERAS, CONFIG, CONTROL_HZ, LIMIT, SEED
from simulation import Simulation, run_directory, start_app, write_json

INSTRUCTION = json.loads((HERE / "task.json").read_text())["instruction"]


class RecordedDream(DreamSession):
    """Capture the actual image bytes and replies on the standard Dream transport."""
    def __init__(self, path):
        super().__init__()
        self.path = path
        path.mkdir(parents=True)
        self.index = 0
        for name in CAMERAS:
            (path / name).mkdir()
        self.observations = (path / "observations.jsonl").open("x", buffering=1)
        self.replies = (path / "replies.jsonl").open("x", buffering=1)

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        paths = {}
        for name, jpeg in frames.items():
            paths[name] = f"{name}/{self.index:05d}.jpg"
            (self.path / paths[name]).write_bytes(base64.b64decode(jpeg))
        self.observations.write(json.dumps({"step": self.index, "time_s": timestamp, "prompt": prompt,
                                           "joint_pos": state.tolist(), "frames": paths})+"\n")
        self.index += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.replies.write(json.dumps({"observation_step": self.index-1, **chunk.to_json()})+"\n")
        print(f"Dream reply at {(self.index-1)/CONTROL_HZ:.2f}s: {len(chunk.rows)} actions "
              f"in {chunk.rtt_s:.2f}s", flush=True)
        return chunk

    def close(self):
        super().close()
        self.observations.close()
        self.replies.close()


class WandEmbodiment(YamArenaEmbodiment):
    def __init__(self, out):
        width, height = CONFIG["observation"]["render_size_wh"]
        super().__init__(device=CONFIG["simulation"]["device"], cam_width=width, cam_height=height,
                         control_hz=CONTROL_HZ, scene=CONFIG["station"]["scene"],
                         top_cam=CONFIG["station"]["top_camera"], object_jitter=0.)
        self.info = replace(self.info, name="yam_two_stick_soap_bubble",
                            supported_target_kinds=frozenset(), environment_id="yam-two-stick-soap-bubble",
                            docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"])))
        self.out = out
        self.sim = None

    def reset(self, scene, *, seed=None):
        assert self.sim is None and scene.init_seed == SEED
        self.sim = Simulation(self.out, cameras=True)
        self._env, self._base = self.sim.env, self.sim.base
        self._ids = dict(zip(ARMS, self.sim.ids))
        self._instruction = scene.instruction
        self._t = 0.
        self.sim.phase = "dream"
        for _ in range(5):
            self.sim.capture()
        return self._observation()

    def step(self, action):
        self.sim.step(action.data)
        self._t = self.sim.steps/CONTROL_HZ
        self.sim.capture()
        result = self.sim.film.result()
        success = result["score"] == 1.
        if self.sim.steps % 100 == 0:
            print("TRIAL", self.sim.steps, result, flush=True)
        return StepResult(observation=self._observation(), terminated=success,
                          termination_reason="film formed and bubble floating" if success else None,
                          truncated=self.sim.steps >= LIMIT and not success, info=result)


class FinalScore:
    name = "two_stick_soap_bubble"

    def __call__(self, record, target):
        result = record.steps[-1].result.info
        return Score(value=result["score"], explanation="A film spans the lifted loop and a closed bubble floats independently.",
                     metadata=result)


def check(embodiment, scene):
    from PIL import Image
    from inspect_robots import Action

    obs = embodiment.reset(scene)
    assert set(obs.images) == set(CAMERAS) and set(obs.state) == {"joint_pos"} and not obs.extra
    for name, rgb in obs.images.items():
        assert rgb.shape == (360, 640, 3) and rgb.std() > 8.
        Image.fromarray(rgb).save(embodiment.out / f"{name}.png")
    command = obs.state["joint_pos"].copy()
    command[[0, 7]] += [.12, -.12]
    for _ in range(20):
        step = embodiment.step(Action(data=command))
    assert step.info["score"] == 0. and step.observation.state_time == 1.
    assert np.max(np.abs(step.observation.state["joint_pos"]-obs.state["joint_pos"])) > .06
    diff = np.abs(step.observation.images["left_cam"].astype(float)-obs.images["left_cam"]).mean()
    assert diff > 1.
    write_json(embodiment.out / "observation-check.json",
               {"control_steps": 20, "api_calls": 0, "score": 0., "wrist_rgb_mae": float(diff)})


def main():
    args = sys.argv[1:]
    check_only = args[:1] == ["--check"]
    if check_only:
        args = args[1:]
    assert len(args) <= 1, "Usage: python dream_eval.py [--check] [RUN_DIRECTORY]"
    out = run_directory("observation-check" if check_only else "dream", args)
    # Initialize before binding the policy so that the run archive exists even if API setup fails.
    app = start_app()
    error = None
    policy = None
    embodiment = WandEmbodiment(out)
    scene = Scene(id="two-stick-soap-bubble", instruction=INSTRUCTION, init_seed=SEED)
    try:
        if check_only:
            check(embodiment, scene)
        else:
            policy = DreamPolicy(model=CONFIG["model"], session=RecordedDream(out.parent / (out.name + "-transport")))
            task = Task(name="two-stick-soap-bubble", scenes=[scene], scorer=FinalScore(), max_steps=LIMIT)
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out / "eval"),
                            seed=SEED, fail_on_error=True, store_actions=True, store_frames=False)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        if embodiment.sim is not None:
            embodiment.sim.save("check" if check_only else "dream", error)
            embodiment.sim.close()
        if policy is not None:
            write_json(out.parent / (out.name + "-transcript.json"), policy.transcript())
            policy.close()
        app.close(exit_code=int(error is not None))


if __name__ == "__main__":
    main()

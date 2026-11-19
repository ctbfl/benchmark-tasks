"""Run the scripted demo, an offline camera check, or a real Dream-1 trial."""
import base64
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from inspect_robots import Observation, Scene, Score, StepResult, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession
from yam_arena.inspect_embodiment import JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

from demo import plan
from scoring import rescore
from world import CAMERAS, CONFIG, HERE, HZ, MAX_STEPS, SIM, World

INSTRUCTION = (
    "Grip the dry handle of the wooden honey dipper in the teal pot. Lift it and carry it "
    "to the toast while rolling the horizontal handle around its long axis to keep the honey on. "
    "Over the toast, stop rolling and slowly sweep across the slice while the honey drizzles. "
    "Keep honey off the plate and table. Return the dipper to its original notched rest on "
    "the teal pot, with the grooved head in the bowl. Open the gripper and move clear. "
    "Leave the dipper stable for one second. Finish within 60 seconds."
)
SEED = CONFIG["episode"]["seed"]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


class ArchiveSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out = out/"model"
        self.out.mkdir()
        self.index = 0
        for name in CAMERAS:
            (self.out/name).mkdir()

    def append(self, name, row):
        with (self.out/name).open("a") as f:
            f.write(json.dumps(row, allow_nan=False)+"\n")

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
        paths = {}
        for name, jpeg in frames.items():
            path = Path(name)/f"{self.index:05d}.jpg"
            (self.out/path).write_bytes(base64.b64decode(jpeg))
            paths[name] = str(path)
        self.append("observations.jsonl", dict(time_s=timestamp, frames=paths, joint_pos=state.tolist()))
        self.index += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.append("replies.jsonl", dict(observation_index=self.index-1,
            commanded=None if commanded is None else commanded.tolist(), **chunk.to_json()))
        print(f"Dream returned {len(chunk.rows)} actions in {chunk.rtt_s:.2f}s", flush=True)
        return chunk


class HoneyEmbodiment(YamArenaEmbodiment):
    def __init__(self, world):
        super().__init__(device=SIM["device"], control_hz=HZ, scene="station", object_jitter=0.,
                         cam_width=640, cam_height=360)
        self.info = replace(self.info, name="twirl_honey_dipper", environment_id="twirl-honey-dipper-v2",
                            docs=JOINT_GUIDE+"\n\n"+ROOM_DOCS["station"],
                            supported_target_kinds=frozenset())
        self.world = world

    def reset(self, scene, *, seed=None):
        from visuals import HoneyVisual
        assert scene.instruction == INSTRUCTION
        self.world.reset(SEED)
        self.vis = HoneyVisual(self.world.base.sim.stage)
        for _ in range(3):
            observation = self.observe()
        return observation

    def observe(self):
        world = self.world
        self.vis.update(world.honey, world.previous_pose)
        images = world.capture()
        t = world.steps/HZ
        assert list(images) == CAMERAS and all(im.shape == (360, 640, 3) for im in images.values())
        return Observation(images=images, state={"joint_pos": world.state().astype(np.float64)},
                           instruction=INSTRUCTION, image_times=dict.fromkeys(images, t), state_time=t)

    def step(self, action):
        self.world.step(action.data)
        grade = self.world.grade()
        full = grade["score"] == 1.
        return StepResult(observation=self.observe(), terminated=full,
                          termination_reason="success" if full else None,
                          truncated=self.world.steps == MAX_STEPS and not full, info=grade)


class FinalScore:
    name = "honey_dipper"

    def __call__(self, record, target):
        if not record.steps:
            return Score(0.)
        info = record.steps[-1].result.info
        return Score(info["score"], metadata=info)


def run(mode, world):
    out = world.out
    if mode == "demo":
        world.reset(SEED)
        phase = plan(world, dwell_s=5., settle_s=3., return_to_rest=True)
        save_json(out/"demo-phases.json", phase)
        return world.grade()["score"]

    scene = Scene(id="twirl-honey-dipper", instruction=INSTRUCTION, init_seed=SEED)
    embodiment = HoneyEmbodiment(world)
    if mode == "check":
        from PIL import Image
        first = embodiment.reset(scene, seed=SEED)
        assert set(first.state) == {"joint_pos"} and not first.extra
        for name, im in first.images.items():
            assert np.std(im) > 12, name
            Image.fromarray(im).save(out/f"{name}.png")
        q = first.state["joint_pos"].copy()
        q[7] += .12
        for _ in range(HZ):
            world.step(q)
        last = embodiment.observe()
        assert abs(last.state["joint_pos"][7] - first.state["joint_pos"][7]) > .05
        assert np.abs(last.images["right_cam"].astype(float)-first.images["right_cam"]).mean() > 1.
        assert last.state_time == 1 and world.grade()["score"] == 0.
        save_json(out/"camera-check.json", dict(api_calls=0, steps=world.steps,
                                              cameras=CAMERAS, rgb_only=True, state_key="joint_pos"))
        return 0.

    assert mode == "dream"
    assert (Path.home()/".dreamlabs/credentials").is_file(), "Configure ~/.dreamlabs/credentials"
    task = Task(name="twirl-honey-dipper", scenes=[scene], scorer=FinalScore(), max_steps=MAX_STEPS)
    with DreamPolicy(model=CONFIG["model"], session=ArchiveSession(out)) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=embodiment, log_dir=str(out/"eval"),
                            seed=SEED, fail_on_error=True, store_frames=True, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
        finally:
            save_json(out/"dream-transcript.json", policy.transcript())
    return world.grade()["score"]


def main():
    from isaaclab.app import AppLauncher
    mode, folder = sys.argv[1:]
    assert mode in ("demo", "dream", "check"), "Usage: python run.py demo|dream|check RUN_DIRECTORY"
    out = Path(folder).resolve()
    out.mkdir(parents=True, exist_ok=False)
    code = [*HERE.glob("*.py"), HERE/"dream-defaults.json", *(HERE/"assets").glob("*.usda")]
    run_id = os.environ["PRODUCTION_GPU_RUN_ID"] if "PRODUCTION_GPU_RUN_ID" in os.environ else None
    save_json(out/"run.json", dict(kind=mode, gpu_run_id=run_id, config=CONFIG,
        instruction=INSTRUCTION, packages={p: version(p) for p in (
            "isaaclab", "isaaclab-arena", "inspect-robots", "inspect-robots-dream", "dreamlabs-client")},
        source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}))
    app = AppLauncher(headless=True, enable_cameras=mode != "demo",
        kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    world = None
    exit_code = 0
    try:
        world = World(out, cameras=mode != "demo")
        score = run(mode, world)
        measured = world.save()
        grade = rescore(out)
        assert grade["score"] == score and grade == world.grade(), (grade, world.grade())
        save_json(out/"scorer.json", grade)
        if mode == "demo":
            assert score == 1., grade
        if mode in ("demo", "dream"):
            save_json(out/"result.json", dict(kind=mode, status="success", score=score, max_score=1,
                gpu_run_id=run_id, control_steps=world.steps, evidence={
                    "scorer": str(out/"scorer.json"), "landings": str(out/"measurements.json"),
                    "trajectory": str(out/"trajectory.npy"), "actions": str(out/"actions.npy"),
                }, measurements={**grade, "landed_g": measured["landed_g"],
                                  "retained_g": measured["retained_g"]}))
        print(mode, grade, flush=True)
    except BaseException:
        exit_code = 1
        error = traceback.format_exc()
        traceback.print_exc()
        save_json(out/"failure.json", dict(kind=mode, status="error", gpu_run_id=run_id, error=error))
        if world is not None and hasattr(world, "trajectory"):
            world.save()
    finally:
        if world is not None:
            world.env.close()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

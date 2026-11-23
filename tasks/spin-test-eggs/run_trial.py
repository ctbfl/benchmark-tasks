"""Run a scripted or Dream-1 trial using the same scene, native actions and scorer."""
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import traceback

import numpy as np
from isaaclab.app import AppLauncher


def run_dream(sim, video, tracker, out, instruction, config, trial_seed):
    from inspect_robots import Observation, Scene, StepResult, Task, eval as evaluate
    from inspect_robots.scorer import Score
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession
    from yam_arena.inspect_embodiment import JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

    limit = config["episode"]["max_control_steps"]

    class LoggedSession(DreamSession):
        def __init__(self):
            super().__init__()
            self.audit = gzip.open(out / "dream-requests.jsonl.gz", "wt")

        def observe(self, frames, state, prompt, timestamp):
            assert set(frames) == set(config["observation"]["camera_names"])
            self.audit.write(json.dumps({"images_jpeg_base64": frames, "joints": state.tolist(),
                                         "instruction": prompt, "time_s": timestamp}) + "\n")
            self.audit.flush()
            return super().observe(frames, state, prompt, timestamp)

        def act(self, commanded):
            chunk = super().act(commanded)
            self.audit.write(json.dumps({"reply": chunk.to_json()}) + "\n")
            self.audit.flush()
            return chunk

        def close(self):
            self.audit.close()
            super().close()

    class EggEmbodiment(YamArenaEmbodiment):
        def __init__(self):
            hz = config["simulation"]["control_hz"]
            super().__init__(device=config["simulation"]["device"], control_hz=hz,
                             cam_width=640, cam_height=360, scene="station", object_jitter=0)
            self.info = replace(self.info, name="yam_spin_test_eggs",
                supported_target_kinds=frozenset(), environment_id="yam-classify-egg-v1",
                docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], instruction)))

        def get_observation(self):
            images, state = video.observe(tracker)
            time_s = sim.ticks / config["simulation"]["physics_hz"]
            return Observation(images=images, state={"joint_pos": state}, instruction=instruction,
                               image_times=dict.fromkeys(images, time_s), state_time=time_s)

        def reset(self, scene, *, seed=None):
            assert not sim.actions and scene.init_seed == trial_seed
            return self.get_observation()

        def step(self, action):
            requested = np.asarray(action.data, dtype=np.float64)
            assert requested.shape == (14,) and np.isfinite(requested).all()
            start = len(sim.rows)
            sim.step(np.clip(requested, self._low, self._high))
            for row in sim.rows[start:]:
                tracker.sample(row)
            result = tracker.result()
            success = result["score"] == 1.
            return StepResult(observation=self.get_observation(), terminated=success,
                              termination_reason="success" if success else None,
                              truncated=len(sim.actions) >= limit and not success, info=result)

        def close(self):
            pass

    class FinalScore:
        name = "classify_egg"

        def __call__(self, record, target):
            info = record.steps[-1].result.info
            return Score(value=info["score"], explanation="Spin, arrest, and timely hands-off response before correct placement; shell break scores zero.",
                         metadata=info)

    scene = Scene(id="classify-egg", instruction=instruction, init_seed=trial_seed)
    task = Task(name="classify-egg", scenes=[scene], scorer=FinalScore(), max_steps=limit)
    with DreamPolicy(model=config["model"], session=LoggedSession()) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=EggEmbodiment(),
                            log_dir=str(out / "eval"), seed=trial_seed, fail_on_error=True,
                            store_frames=False, store_actions=True)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
        finally:
            (out/"dream-transcript.json").write_text(json.dumps(policy.transcript(), indent=2) + "\n")


def archive_source(out):
    root = Path(__file__).resolve().parent
    dest = out/"source"
    dest.mkdir()
    for file in [*root.glob("*.py"), root/"dream-defaults.json"]:
        shutil.copy2(file, dest/file.name)
    shutil.copytree(root/"assets", dest/"assets")
    files = (p for p in dest.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/"source-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main():
    kind, path, *replay = sys.argv[1:]
    assert kind in ("demo", "dream", "screenshot")
    assert len(replay) <= 1, "usage: run_trial.py {demo,dream,screenshot} output [replay_seed]"
    seed = int(replay[0]) if replay else secrets.randbits(31)
    assert 0 <= seed < 2**31
    trial = {"trial_seed": seed, "seed_source": "replay" if replay else "evaluator_random"}
    out = Path(path)
    run_id = os.environ["PRODUCTION_GPU_RUN_ID"]
    if kind != "screenshot":
        out.mkdir(parents=True)
        archive_source(out)
        (out/"trial.json").write_text(json.dumps({"kind": kind, "gpu_run_id": run_id, **trial},
                                                indent=2) + "\n")
    launcher = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8")
    from PIL import Image
    from scene import CONFIG, INSTRUCTION
    from scoring import Score, evaluate
    from simulation import Simulation, write_json
    from video import Video

    sim = video = None
    error = None
    try:
        if kind == "screenshot":
            sim = Simulation(out.parent, seed=seed, screenshot=True)
            for _ in range(4):
                image = sim.capture(names=("top_cam",))["top_cam"]
            assert image.shape == (1080, 1920, 3) and image.std() > 5
            Image.fromarray(image).save(out)
        else:
            sim = Simulation(out, seed=seed)
            tracker = Score(sim.raw)
            tracker.sample(sim.rows[0])
            if kind == "dream":
                from dream_recording import DreamRecording
                video = DreamRecording(sim, out)
                run_dream(sim, video, tracker, out, INSTRUCTION, CONFIG, seed)
            else:
                from sort_demo import sort
                video = Video(sim, out)
                video.capture()
                sort(sim, video.capture)
            score = evaluate(sim.rows, sim.raw)
            if kind == "dream":
                assert tracker.result() == score
            else:
                assert score["score"] == 1., score
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        if sim is not None and kind != "screenshot":
            with gzip.open(out/"scorer-physics.jsonl.gz", "wt") as file:
                for row in sim.rows:
                    file.write(json.dumps(row, allow_nan=False) + "\n")
            np.save(out/"actions.npy", np.asarray(sim.actions, dtype=np.float32))
            measured = evaluate(sim.rows, sim.raw)
            result = {"kind": kind, "status": "failed" if error else "success", "error": error,
                      "max_score": 1, "gpu_run_id": run_id, **trial, **measured,
                      "raw_scorer_log": "scorer-physics.jsonl.gz", "actions": "actions.npy",
                      "recorded_frame_ticks": video.ticks if video is not None else []}
            write_json(out/"result.json", result)
            print("EXECUTED SCORE", result["score"], "CONTROL STEPS", len(sim.actions), flush=True)
        if video is not None:
            if kind == "dream":
                video.flush(tracker)
            video.close()
        if sim is not None:
            sim.close()
        launcher.app.close(exit_code=bool(error))


if __name__ == "__main__":
    main()

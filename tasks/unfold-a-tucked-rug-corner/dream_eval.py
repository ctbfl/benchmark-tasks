"""Run one Dream-1 episode in the same native FEM environment as the expert."""
from dataclasses import replace
import base64
import json
import signal
from pathlib import Path
import sys
import traceback
import numpy as np
from scene import CONFIG, CONTROL_HZ, EPISODE, INSTRUCTION, SIM


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")


def append_json(file, data):
    file.write(json.dumps(data, allow_nan=False) + "\n")
    file.flush()


def dream(trial):
    from inspect_robots import Observation, Scene, StepResult, Task, eval as evaluate
    from inspect_robots.scorer import Score
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession
    from yam_arena.inspect_embodiment import JOINT_GUIDE, ROOM_DOCS, YamArenaEmbodiment

    class RecordedSession(DreamSession):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.frames = trial.out / "dream-input"
            self.frames.mkdir()
            self.observations = (self.frames / "observations.jsonl").open("x")
            self.replies = (self.frames / "replies.jsonl").open("x")

        def observe(self, frames, state, prompt, timestamp):
            assert set(frames) == set(CONFIG["observation"]["camera_names"]) and prompt == INSTRUCTION
            jpeg_paths = {}
            for name, jpeg in frames.items():
                path = f"{self.index:05d}-{name}.jpg"
                (self.frames / path).write_bytes(base64.b64decode(jpeg))
                jpeg_paths[name] = path
            append_json(self.observations, {"tick": self.index, "time_s": timestamp,
                        "jpeg": jpeg_paths, "joint_pos": state.tolist()})
            self.index += 1
            super().observe(frames, state, prompt, timestamp)

        def act(self, commanded):
            chunk = super().act(commanded)
            append_json(self.replies, {"tick": self.index-1, **chunk.to_json()})
            print("DREAM", self.index-1, len(chunk.rows), round(chunk.rtt_s, 2), flush=True)
            return chunk

    class Embodiment(YamArenaEmbodiment):
        def __init__(self):
            w, h = CONFIG["observation"]["render_size_wh"]
            super().__init__(device=SIM["device"], cam_width=w, cam_height=h, control_hz=CONTROL_HZ,
                             scene=CONFIG["station"]["scene"], top_cam=CONFIG["station"]["top_camera"],
                             object_jitter=EPISODE["object_jitter_m"])
            self.info = replace(self.info, name="yam_tucked_rug_corner", supported_target_kinds=frozenset(),
                environment_id="yam-tucked-rug-corner",
                docs="\n\n".join((JOINT_GUIDE, ROOM_DOCS["station"], INSTRUCTION)))

        def observation(self):
            images = trial.images()
            stamp = trial.steps/CONTROL_HZ
            return Observation(images=images, instruction=INSTRUCTION, state={"joint_pos": trial.state()},
                               state_time=stamp, image_times=dict.fromkeys(images, stamp))

        def reset(self, scene, *, seed=None):
            assert trial.steps == 0
            return self.observation()

        def step(self, action):
            trial.step(np.asarray(action.data), np.asarray(action.data))
            done = trial.measure()["score"] == 1.
            return StepResult(observation=self.observation(), terminated=done,
                truncated=trial.steps == EPISODE["max_control_steps"] and not done,
                termination_reason="success" if done else None, info=trial.measure())

    class FinalScore:
        name = "unfolded_rug_in_frame"

        def __call__(self, record, target):
            if not record.steps:
                return Score(value=0.)
            data = record.steps[-1].result.info
            return Score(value=float(data["score"]), metadata=data)

    session = RecordedSession()
    scene = Scene(id="unfold-a-tucked-rug-corner", instruction=INSTRUCTION, init_seed=EPISODE["seed"])
    task = Task(name=scene.id, scenes=[scene], scorer=FinalScore(), max_steps=EPISODE["max_control_steps"])
    policy = DreamPolicy(model=CONFIG["model"], session=session)
    try:
        logs = evaluate(task, policy=policy, embodiment=Embodiment(), log_dir=str(trial.out/"inspect"),
                        seed=EPISODE["seed"], fail_on_error=True, store_actions=True, store_frames=False)
        assert len(logs) == 1 and logs[0].status == "success"
    finally:
        write_json(trial.out/"dream-transcript.json", policy.transcript())
        policy.close()
        session.observations.close()
        session.replies.close()



if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python dream_eval.py RUN_DIRECTORY"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Execution interrupted by SIGTERM"))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=SIM["device"], enable_cameras=True,
        kit_args="--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app
    try:
        from runtime import Experiment
        trial = Experiment(Path(sys.argv[1]).resolve(), cameras=True)
        try:
            dream(trial)
        finally:
            trial.save()
            trial.close()
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

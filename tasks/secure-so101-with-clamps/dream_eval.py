"""Evaluate Dream-1 using the standard three YAM cameras and joint state."""

from pathlib import Path
import sys
import traceback


def run(mode, out):
    import base64
    import json
    from inspect_robots import Scene, Task, eval as evaluate
    from inspect_robots.scorer import Score
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession
    from PIL import Image

    from scene import CAMERAS, CONFIG, INSTRUCTION, MAX_STEPS
    from trial import Trial, write_json

    class SavedSession(DreamSession):
        def __init__(self):
            super().__init__()
            self.archive = out / "model"
            self.archive.mkdir()
            self.observations = (self.archive / "observations.jsonl").open("w")
            self.replies = (self.archive / "replies.jsonl").open("w")
            self.tick = 0

        def observe(self, frames, state, prompt, timestamp):
            assert set(frames) == set(CAMERAS) and prompt == INSTRUCTION
            bundle = {}
            for name, jpeg in frames.items():
                path = f"{self.tick:05d}-{name}.jpg"
                (self.archive / path).write_bytes(base64.b64decode(jpeg))
                bundle[name] = path
            self.observations.write(json.dumps(dict(step=self.tick, timestamp=timestamp, joint_pos=state.tolist(),
                                                   instruction=prompt, frames=bundle)) + "\n")
            self.tick += 1
            super().observe(frames, state, prompt, timestamp)

        def act(self, commanded):
            self.observations.flush()
            reply = super().act(commanded)
            self.replies.write(json.dumps(dict(step=self.tick - 1, **reply.to_json())) + "\n")
            self.replies.flush()
            print(f"Dream reply at step {self.tick-1} with {len(reply.rows)} rows", flush=True)
            return reply

        def close(self):
            super().close()
            self.observations.close()
            self.replies.close()

    class FinalScore:
        name = "secure_so101_final"

        def __call__(self, record, target):
            score = record.steps[-1].result.info["score"]
            return Score(value=score, explanation="Final hands-off retention after supported clamp installation")

    world = Trial(out, cameras=True)
    seed = CONFIG["episode"]["seed"]
    scene = Scene(id="secure-so101", instruction=INSTRUCTION, init_seed=seed)
    write_json(out / "run.json", dict(mode=mode, instruction=INSTRUCTION, config=CONFIG,
                                     observation_keys=["left_cam", "top_cam", "right_cam", "joint_pos"]))
    try:
        if mode == "check":
            observation = world.reset(scene, seed=seed)
            assert set(observation.images) == set(CAMERAS) and set(observation.state) == {"joint_pos"}
            for name, rgb in observation.images.items():
                assert rgb.shape == (360, 640, 3) and rgb.std() > 8
                Image.fromarray(rgb).save(out / f"{name}.png")
            assert world.last["score"] == 0.
            command = observation.state["joint_pos"].copy()
            command[0] += .13
            world.phase = "adapter check"
            for _ in range(20):
                world.perform(command)
            after = world.observe()
            assert abs(after.state["joint_pos"][0] - observation.state["joint_pos"][0]) > .06
            assert (after.images["left_cam"] != observation.images["left_cam"]).mean() > .05
            print("Camera, standard action and reset check passed", flush=True)
            return
        task = Task(name="secure-so101", scenes=[scene], scorer=FinalScore(), max_steps=MAX_STEPS)
        session = SavedSession()
        try:
            with DreamPolicy(model=CONFIG["model"], session=session) as policy:
                try:
                    logs = evaluate(task, policy=policy, embodiment=world, log_dir=str(out / "eval"),
                                    seed=seed, fail_on_error=True, store_frames=True, store_actions=True)
                    assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
                    result = world.result("dream")
                    print("DREAM SCORE", result["score"], "steps", result["steps"], flush=True)
                finally:
                    write_json(out / "transcript.json", policy.transcript())
        finally:
            session.close()
    finally:
        world.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    mode, path = sys.argv[1:]
    assert mode in ("check", "dream")
    app = AppLauncher(headless=True, enable_cameras=True, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app
    try:
        run(mode, Path(path))
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

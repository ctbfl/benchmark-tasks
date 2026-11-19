"""Dream-1 evaluation with standard station observations and an auditable transcript."""
import base64
import json

from inspect_robots import Task, eval as evaluate
from inspect_robots.scorer import Score
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession

from scene import CONFIG


class FinalScore:
    name = "cinnamon_roll_final"

    def __call__(self, record, target):
        measures = record.steps[-1].result.info
        return Score(value=measures["score"], explanation="Measured final cut, shape and placement",
                     metadata=measures)


class RecordedSession(DreamSession):
    def __init__(self, out):
        super().__init__()
        self.out = out
        out.mkdir()
        self.count = 0
        self.archive = (out/"events.jsonl").open("x", buffering=1)

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CONFIG["observation"]["camera_names"])
        paths = {}
        for name, jpeg in frames.items():
            path = self.out/f"{self.count:06d}-{name}.jpg"
            path.write_bytes(base64.b64decode(jpeg))
            paths[name] = path.name
        self.archive.write(json.dumps(dict(frame=self.count, time_s=timestamp, images=paths,
                                           joint_pos=state.tolist(), instruction=prompt))+"\n")
        self.count += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.archive.write(json.dumps(dict(frame=self.count-1, **chunk.to_json()))+"\n")
        print(f"Dream request at observation {self.count}: {len(chunk.rows)} actions", flush=True)
        return chunk


def dream_trial(sim, scene):
    session = RecordedSession(sim.out/"dream-io")
    task = Task(name=scene.id, scenes=[scene], scorer=FinalScore(),
                max_steps=CONFIG["episode"]["max_control_steps"])
    with DreamPolicy(model=CONFIG["model"], session=session) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=sim, log_dir=str(sim.out/"inspect"),
                            seed=CONFIG["episode"]["seed"], fail_on_error=True,
                            store_actions=True, store_frames=False)
            assert len(logs) == 1 and logs[0].status == "success", logs
        finally:
            (sim.out/"dream-transcript.json").write_text(json.dumps(policy.transcript(), indent=2)+"\n")
            session.close()
            session.archive.close()

"""Dream-1 evaluation through Inspect Robots, with an auditable policy transcript."""

import json

from inspect_robots import Scene, Task, eval as evaluate
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession

from scene import CAMERAS, CONFIG, EPISODE, INSTRUCTION


class FinalScore:
    name = "rosette"

    def __call__(self, record, target):
        from inspect_robots.scorer import Score

        measurements = record.steps[-1].result.info
        return Score(value=measurements["score"], explanation="Continuous meshed stroke and closure",
                     metadata=measurements)


class RecordedSession(DreamSession):
    def __init__(self, path):
        super().__init__()
        self.output = path.open("x", buffering=1)
        self.tick = 0

    def observe(self, frames, state, prompt, timestamp):
        assert set(frames) == set(CAMERAS) and state.shape == (14,) and prompt == INSTRUCTION
        item = dict(observation=self.tick, time_s=timestamp, joint_pos=state.tolist(), jpeg=frames)
        self.output.write(json.dumps(item) + "\n")
        self.tick += 1
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.output.write(json.dumps(dict(observation=self.tick-1, **chunk.to_json())) + "\n")
        print(f"Dream reply {self.tick-1}: {len(chunk.rows)} actions, {chunk.rtt_s:.2f}s round trip", flush=True)
        return chunk

    def close(self):
        super().close()
        self.output.close()


def evaluate_dream(robot):
    scene = Scene(id="spirograph-rosette", instruction=INSTRUCTION, init_seed=EPISODE["seed"])
    task = Task(name=scene.id, scenes=[scene], scorer=FinalScore(), max_steps=EPISODE["max_control_steps"])
    session = RecordedSession(robot.out / "dream-wire.jsonl")
    with DreamPolicy(model=CONFIG["model"], session=session) as policy:
        try:
            logs = evaluate(task, policy=policy, embodiment=robot, log_dir=str(robot.out / "eval"),
                seed=EPISODE["seed"], fail_on_error=True, store_actions=True, store_frames=False)
            assert len(logs) == 1 and logs[0].status == "success", [log.error for log in logs]
        finally:
            (robot.out / "dream-transcript.json").write_text(json.dumps(policy.transcript(), indent=2) + "\n")

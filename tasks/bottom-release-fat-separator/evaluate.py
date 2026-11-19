"""Run and record one scored expert or Dream-1 trial."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import traceback

from inspect_robots import Scene, Task, eval as evaluate
from inspect_robots.scorer import Score
from inspect_robots_dream.policy import DreamPolicy
from inspect_robots_dream.session import DreamSession

from liquid import CONFIG, INSTRUCTION, LIMIT
from scene import GravyStation, scene_hashes


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class RecordedSession(DreamSession):
    """Keep the adapter's model frames, state, prompt and action chunks."""

    def __init__(self, log):
        super().__init__()
        self.log = log

    def save(self, entry):
        self.log.write(json.dumps(entry, allow_nan=False) + "\n")
        self.log.flush()

    def observe(self, frames, state, prompt, timestamp):
        self.save(dict(time_s=timestamp, jpeg_b64=frames, joint_pos=state.tolist(), prompt=prompt))
        super().observe(frames, state, prompt, timestamp)

    def act(self, commanded):
        chunk = super().act(commanded)
        self.save(dict(reply=chunk.to_json()))
        print(f"Dream reply: {len(chunk.rows)} actions in {chunk.rtt_s:.2f} s", flush=True)
        return chunk


@dataclass(frozen=True)
class GravyRecovery:
    name: str = "gravy_recovery"

    def __call__(self, record, target):
        return Score(record.steps[-1].result.info["score"] if record.steps else 0.)


def main(kind, out, seed):
    out.mkdir(parents=True)
    run_id = os.environ["PRODUCTION_GPU_RUN_ID"]
    env = GravyStation(out, seed)
    scenario = Scene(id="bottom-release-fat-separator", instruction=INSTRUCTION, init_seed=seed)
    code = 1
    try:
        write_json(out / "config.json", CONFIG)
        write_json(out / "trial.json", dict(kind=kind, seed=seed, instruction=INSTRUCTION, gpu_run_id=run_id, scene_sha256=scene_hashes()))
        if kind == "demo":
            from PIL import Image
            obs = env.reset(scenario, seed=seed)
            for name, image in obs.images.items():
                Image.fromarray(image).save(out / f"initial-{name}.png")
            from expert import run
            run(env)
        else:
            task = Task(name=scenario.id, scenes=[scenario], scorer=GravyRecovery(), max_steps=LIMIT)
            with (out / "model-io.jsonl").open("x") as audit, DreamPolicy(
                    model=CONFIG["model"], session=RecordedSession(audit)) as policy:
                try:
                    logs = evaluate(task, policy=policy, embodiment=env, log_dir=str(out / "inspect"),
                                    seed=seed, fail_on_error=True, store_frames=True, store_actions=True)
                    assert len(logs) == 1 and logs[0].status == "success", [log.status for log in logs]
                    assert logs[0].samples[0].reduced["gravy_recovery"] == env.drain.score()
                finally:
                    write_json(out / "dream-transcript.json", policy.transcript())
        metrics = env.metrics()
        write_json(out / "metrics.json", metrics)
        result = dict(kind=kind, status="success", score=metrics["score"], max_score=1,
                      gpu_run_id=run_id, seed=seed, raw_metrics=str(out / "metrics.json"),
                      physics_trajectory=str(out / "physics-f64.bin"),
                      applied_actions=str(out / "actions-f64.bin"), measurements=metrics)
        if kind == "dream":
            result["inspect_logs"] = str(out / "inspect")
            result["model_io"] = str(out / "model-io.jsonl")
        write_json(out / "result.json", result)
        print(json.dumps(result), flush=True)
        code = 0
    except BaseException:
        traceback.print_exc()
        write_json(out / "failure.json", dict(kind=kind, gpu_run_id=run_id, error=traceback.format_exc()))
    finally:
        env.close()
        if env.app is not None:
            env.app.close(exit_code=code)
    return code


if __name__ == "__main__":
    assert 3 <= len(sys.argv) <= 4 and sys.argv[1] in ("demo", "dream"), (
        "Usage: python evaluate.py {demo|dream} RUN_DIR [SEED]")
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else CONFIG["episode"]["seed"]
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), seed))

"""Shared action, sensing, and evidence path for the demo and Dream-1."""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from scene import CAMERAS, CONFIG, EPISODE, HERE, INSTRUCTION


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    assert len(sys.argv) == 3 and sys.argv[1] in ("demo", "arc", "dream", "check"), "Usage: python trial.py demo|arc|dream|check RUN_DIRECTORY"
    from isaaclab.app import AppLauncher

    kind, out = sys.argv[1], Path(sys.argv[2]).resolve()
    out.mkdir(parents=True)
    app = AppLauncher(headless=True, enable_cameras=kind not in ("demo", "arc"), kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6 "
        f"--/log/file={out}/kit.log").app
    robot = None
    try:
        from simulation import DrawingStation

        write_json(out / "run.json", dict(kind=kind, config=CONFIG, instruction=INSTRUCTION,
            gpu_run_id=os.environ["PRODUCTION_GPU_RUN_ID"],
            source={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (*HERE.glob("*.py"), *HERE.glob("*.json"), *(HERE / "assets").glob("*.usda"))},
            trace_layout=["left joints 8", "right joints 8", "wheel xyz+xyzw", "pen xyz+xyzw",
                          "paper, ring and bore normal-force norms in N"]))
        robot = DrawingStation(out, cameras=kind not in ("demo", "arc"))
        if kind == "dream":
            from dream_eval import evaluate_dream
            evaluate_dream(robot)
        else:
            from inspect_robots import Scene

            observation = robot.reset(Scene(id="spirograph-rosette", instruction=INSTRUCTION), seed=EPISODE["seed"])
            if kind == "check":
                from PIL import Image

                assert set(observation.images) == set(CAMERAS) and set(observation.state) == {"joint_pos"}
                assert observation.state["joint_pos"].shape == (14,) and not observation.extra
                for name, frame in observation.images.items():
                    assert frame.shape == (360, 640, 3) and frame.std() > 10
                    Image.fromarray(frame).save(out / f"{name}.png")
                robot.advance(observation.state["joint_pos"])
                assert robot.scorer.result()["score"] == 0.
            else:
                from demo import draw

                draw(robot, short_arc=kind == "arc")
        result = robot.scorer.result()
        write_json(out / "scorer.json", result)
        if kind != "check":
            write_json(out / "result.json", dict(kind=kind, status="success", score=result["score"],
                max_score=1, gpu_run_id=os.environ["PRODUCTION_GPU_RUN_ID"],
                measurements=result, raw_scorer="scorer.json", physics_trace="physics.f32",
                actions="executed-actions.jsonl", control_steps=robot.control_steps))
        print(json.dumps(result), flush=True)
    except Exception:
        if robot is not None:
            write_json(out / "scorer.json", robot.scorer.result())
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    finally:
        if robot is not None:
            robot.close()
    app.close()


if __name__ == "__main__":
    main()

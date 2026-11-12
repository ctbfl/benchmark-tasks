"""Recompute per-coin sorting credit from an archived run's final measured coin poses."""
import json
from pathlib import Path
import sys

from scene import CONFIG
from task_runtime import evaluate


def score(samples):
    limit = CONFIG["episode"]["max_control_steps"] / CONFIG["simulation"]["control_hz"]
    final = samples[-1]
    return evaluate(final["poses"])["score"] if final["time_s"] <= limit + 1e-9 else 0.0


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    out = Path(sys.argv[1])
    samples = json.loads((out / "measurements.json").read_text())
    final = evaluate(samples[-1]["poses"])
    result = {"score": score(samples), "time_s": samples[-1]["time_s"],
              "pennies_in_left_frame": final["pennies_in_left_frame"],
              "quarters_in_right_frame": final["quarters_in_right_frame"]}
    expected = json.loads((out / "result.json").read_text())["score"]
    assert result["score"] == expected, (result, expected)
    (out / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

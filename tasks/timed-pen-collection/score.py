"""Recompute binary success from archived poses, button travel and contact results."""
import json
from pathlib import Path
import sys

import numpy as np
from build_assets import PEN_RADIUS
from scene import CONFIG
from task_runtime import Timer, in_holder


def score(samples):
    if not samples:
        return 0.
    timer = Timer()
    initial = np.asarray(samples[0]["pens"])[:, :3]
    early = start_valid = stop_valid = False
    for sample in samples:
        event = timer.update(sample["time_s"], sample["button_m"])
        contained = [in_holder(pen, sample["cup"]) for pen in sample["pens"]]
        released = all(force < .01 for force in sample["robot_forces_N"])
        if timer.started_at is None:
            early |= any(np.asarray(sample["pens"])[:, 2] - initial[:, 2] > PEN_RADIUS*2)
        if event == "start":
            start_valid = not early and not any(contained) and len(timer.events) == 1
            stop_valid = False
        if event == "stop":
            stop_valid = start_valid and all(contained) and released
    return float(stop_valid and not timer.running and all(contained) and released
                 and len(samples[-1]["supported"]) == 5
                 and samples[-1]["time_s"] <= CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"])


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    out = Path(sys.argv[1])
    samples = json.loads((out/"measurements.json").read_text())
    result = {"score": score(samples), "time_s": samples[-1]["time_s"]}
    expected = json.loads((out/"result.json").read_text())["score"]
    assert result["score"] == expected, (result, expected)
    (out/"score.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))

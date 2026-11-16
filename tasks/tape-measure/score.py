"""Recompute measurement progress and submission score from recorded physics."""

import json
from pathlib import Path
import sys

import numpy as np

from task import measurement, score


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    folder = Path(sys.argv[1])
    result = json.loads((folder/"result.json").read_text())
    config = result["config"]
    assert result["length_mm"] == int(np.random.default_rng(config["episode"]["seed"]).integers(150, 201))
    time = result["submission_time_s"]
    seconds = result["time_s"] if time is None else time
    limit = config["episode"]["max_control_steps"]/config["simulation"]["control_hz"]
    columns, offset = {}, 0
    for name, width in result["record_columns"].items():
        columns[name] = slice(offset, offset+width)
        offset += width
    hooked_at = measured_at = None
    for row in np.load(folder/"trajectory.npz")["states"]:
        t = float(row[columns["time"]][0])
        if t > min(seconds, limit):
            break
        state = measurement(row[columns["tape_zero"]], row[columns["tape_mouth"]],
            row[columns["wood_pose"]], result["length_mm"], float(row[columns["hook_force"]][0]))
        if state["hooked"] and hooked_at is None:
            hooked_at = t
        if state["measured"] and measured_at is None:
            measured_at = t
    assert (hooked_at, measured_at) == (result["hooked_at_s"], result["measured_at_s"])
    value = score(result["submitted"], seconds, result["length_mm"], limit, hooked_at, measured_at)
    assert value == result["score"], "Archived score does not match the submitted result"
    print(json.dumps(dict(score=value, submitted=result["submitted"], submission_time_s=time,
        hooked_at_s=hooked_at, measured_at_s=measured_at)))

"""Recompute the task score from an archived calculator submission."""

import json
from pathlib import Path
import sys

from task import score


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    result = json.loads((Path(sys.argv[1])/"result.json").read_text())
    config = result["config"]
    time = result["submission_time_s"]
    value = score(result["submitted"], result["time_s"] if time is None else time,
        result["length_mm"], config["episode"]["max_control_steps"]/config["simulation"]["control_hz"])
    assert value == result["score"], "Archived score does not match the submitted result"
    print(json.dumps(dict(score=value, submitted=result["submitted"], submission_time_s=time)))

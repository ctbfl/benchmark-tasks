"""Recompute an archived episode's binary score without starting the simulator."""

import json
from pathlib import Path
import sys

from task import score


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    report = json.loads((Path(sys.argv[1])/"result.json").read_text())
    config = report["config"]
    value = score(report["submitted"], report["submission_time_s"],
        config["episode"]["max_control_steps"]/config["simulation"]["control_hz"])
    assert value == report["score"], "Archived score differs from the calculator submission"
    print(json.dumps(dict(score=value, submitted=report["submitted"],
                          submission_time_s=report["submission_time_s"])))

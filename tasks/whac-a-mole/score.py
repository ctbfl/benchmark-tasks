"""Recompute an archived episode's score without starting the simulator."""

import json
from pathlib import Path
import sys

from task import REQUIRED_HITS, score


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    report = json.loads((Path(sys.argv[1])/"result.json").read_text())
    hits = [event["time_s"] for event in report["events"] if event["hit"]]
    assert hits == report["hit_times_s"], "Archived hit list differs from the press events"
    for event in report["events"]:
        assert event["hit"] == (event["lit"] == event["button"]), "Hit flag differs from the lit button"
    value = score(hits, report["limit_s"])
    assert value == report["score"], "Archived score differs from the recorded hits"
    print(json.dumps(dict(score=value, hits=len(hits), required_hits=REQUIRED_HITS, hit_times_s=hits)))

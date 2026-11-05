"""Recompute the final score from a saved physical trial, without inference."""
import json
from pathlib import Path
import sys

import numpy as np

from scoring import HeadphoneScore, JOINTS


def score_demo(run):
    run = Path(run)
    meta = json.loads((run / "metrics.json").read_text())
    rows = json.loads((run / "trajectory.json").read_text())
    assert rows and np.allclose(np.diff([r["t"] for r in rows]), 1 / 30)
    assert rows[-1]["t"] <= 120
    scorer = HeadphoneScore(30)
    for row in rows:
        result = scorer.update(dict(time_s=row["t"], root=row["root"], stand=row["stand"], links=row["links"],
            angles_deg={name: row["joint_degrees"][meta["headphone_joint_names"].index(name)] for name in JOINTS},
            support_force_N=row["support_force_N"], robot_contact_N=row["robot_contact_N"]))
    return result


def score_dream(run):
    run = Path(run)
    schema = json.loads((run / "replay-schema.json").read_text())
    assert schema["columns"] == 192 and schema["physics_dt"] == 1 / 240
    states = np.fromfile(run / "physics-f64.bin", dtype="<f8").reshape(-1, 192)
    rows = [json.loads(line) for line in (run / "measurements.jsonl").read_text().splitlines()]
    assert len(rows) * 12 + 1 == len(states) and len(rows) <= 2400
    assert np.allclose(np.diff(states[:, 0]), 1 / 240)
    scorer = HeadphoneScore(20)
    for state, row in zip(states[12::12], rows):
        result = scorer.update(dict(time_s=float(state[0]), root=state[33:40], stand=state[74:81],
            links=state[87:].reshape(15, 7), angles_deg={name: float(np.rad2deg(
                state[46 + schema["headphone_joint_names"].index(name)])) for name in JOINTS},
            support_force_N=row["support_force_N"], robot_contact_N=row["robot_contact_N"]))
        assert result["score"] == row["score"], (row["time_s"], row["score"], result["score"])
    return result


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] in ("demo", "dream"), "Usage: python score.py demo|dream RUN_DIRECTORY"
    result = {"demo": score_demo, "dream": score_dream}[sys.argv[1]](Path(sys.argv[2]))
    print(json.dumps(result, indent=2, allow_nan=False))

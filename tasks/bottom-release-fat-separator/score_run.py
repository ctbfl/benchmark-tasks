"""Recompute phase amounts and the score from recorded physical valve travel."""

import json
from pathlib import Path
import sys

import numpy as np

from liquid import CONFIG, DT, Drain


def read_trace(folder):
    # Time and valve displacement, then left and right arrays of eight joint
    # positions and eight velocities apiece in the workstation's joint order.
    rows = np.fromfile(folder / "physics-f64.bin", dtype="<f8").reshape(-1, 34)
    np.testing.assert_allclose(rows[:, 0], np.arange(len(rows)) * DT, atol=1e-10)
    sim = CONFIG["simulation"]
    cap = CONFIG["episode"]["max_control_steps"] * sim["physics_hz"] // sim["control_hz"]
    assert 1 < len(rows) <= cap + 1 and np.isfinite(rows).all()
    return rows


def score_run(folder):
    result = json.loads((folder / "result.json").read_text())
    drain = Drain(result["seed"])
    for stroke in read_trace(folder)[1:, 1]:
        drain.step(stroke)
    metrics = drain.metrics()
    # Older recordings list volumes only and may have used a different scoring rule.
    for key, value in result["measurements"].items():
        if key not in ("score", "peak_stroke_m", "peak_pad_contact_n"):
            np.testing.assert_allclose(metrics[key], value, atol=1e-9)
    return metrics


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score_run.py RUN_DIR"
    print(json.dumps(score_run(Path(sys.argv[1])), indent=2))

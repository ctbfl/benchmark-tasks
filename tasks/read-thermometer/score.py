"""Recompute calculator submissions from recorded physical button travel."""
from decimal import Decimal
import json
from pathlib import Path
import sys

import numpy as np
from electronics import CalculatorState, TimerState, pressed_buttons

HERE = Path(__file__).resolve().parent


def score(out):
    out = Path(out)
    schema = json.loads((out / "schema.json").read_text())
    result = json.loads((out / "result.json").read_text())
    metadata = json.loads((out / "source/assets/electronics/metadata.json").read_text())
    rows = np.load(out / "trajectory.npz")["states"]
    dt = 1 / result["config"]["simulation"]["physics_hz"]
    assert np.allclose(rows[:, 0], np.arange(len(rows)) * dt, atol=1e-9)
    assert rows[-1, 0] <= result["config"]["episode"]["max_control_steps"] / 20
    devices = {"timer": TimerState(), "calculator": CalculatorState()}
    pressed = {name: set() for name in devices}
    events = []
    for row in rows[1:]:
        for name, device in devices.items():
            q = row[slice(*schema[name + "_q"])]
            names = result["joint_names"][name]
            positions = {key: float(q[names.index(info["joint"])])
                         for key, info in metadata[name]["buttons"].items()}
            active = pressed_buttons(positions, pressed[name],
                press_threshold=metadata[name]["press_threshold_m"],
                release_threshold=metadata[name]["release_threshold_m"])
            for key in sorted(active - pressed[name]):
                events.append({"time_s": float(row[0]), "device": name, "key": key})
            pressed[name] = active
            device.update_buttons(active)
        devices["timer"].advance(dt)
    submitted = devices["calculator"].last_submission
    value = int(submitted == Decimal(45))
    report = {"score": value, "submitted": None if submitted is None else str(submitted),
        "time_s": float(rows[-1, 0]), "events": events,
        "method": "Recomputed from measured spring-button joint positions at every physics step"}
    assert value == result["score"] and report["submitted"] == result["submitted"]
    (out / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python score.py RUN_DIRECTORY"
    print(json.dumps(score(sys.argv[1]), indent=2))

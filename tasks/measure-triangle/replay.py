"""Replay fixed joint-position commands without observations or task feedback."""

import json
from pathlib import Path

import numpy as np


def run(out):
    from simulation import Simulation, archive_source
    data = np.load(Path(__file__).parent/"demo_actions.npz", allow_pickle=False)
    actions, phases = data["actions"], data["phases"]
    assert actions.ndim == 2 and actions.shape[1] == 14
    assert len(actions) == len(phases) and np.isfinite(actions).all()
    archive_source(out)
    sim = Simulation(out)
    (out/"run.json").write_text(json.dumps(dict(policy="fixed_open_loop", policy_observations=[],
        action_file="demo_actions.npz", control_steps=len(actions),
        control_hz=1/sim.control_dt), indent=2)+"\n")
    try:
        for action, phase in zip(actions, phases):
            sim.phase = str(phase)
            sim.step(action)
        assert sim.instruments.result(sim.time)["score"] == 1., "Open-loop demonstration failed"
        sim.save("success")
    except Exception as error:
        sim.save("error", str(error))
        raise
    finally:
        sim.close()

"""Execute the fixed native joint commands from the standard reset."""

from pathlib import Path
import json

import numpy as np


def run(out):
    from simulation import Simulation, archive_source
    from scene import BAR_LENGTH_MM, CONFIG
    commands = np.load(Path(__file__).parent/"demo_actions.npz")
    assert int(commands["seed"]) == CONFIG["episode"]["seed"] and int(commands["length_mm"]) == BAR_LENGTH_MM, "Recorded demo requires seed 0 (193 mm)"
    actions, phases = commands["actions"], commands["phases"]
    assert actions.ndim == 2 and actions.shape[1] == 14 and len(phases) == len(actions)
    archive_source(out)
    (out/"run.json").write_text(json.dumps(dict(policy="fixed_open_loop", policy_observations=[],
        action_file="demo_actions.npz", control_steps=len(actions), control_hz=20), indent=2)+"\n")
    sim = Simulation(out)
    try:
        for action, phase in zip(actions, phases):
            sim.phase = str(phase)
            sim.step(action)
        assert sim.instruments.result(sim.time)["score"] == 1., "Open-loop demonstration failed"
    except Exception as error:
        sim.save("incomplete", str(error))
        raise
    else:
        sim.save("success")
    finally:
        sim.close()

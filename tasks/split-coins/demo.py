"""Replay fixed joint and gripper commands; measurements do not feed back to control."""
from datetime import datetime, timezone
from pathlib import Path
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    from scene import CONFIG
    from simulation import Simulation, archive_source
    actions = np.load(HERE / "demo_actions.npy")
    assert actions.ndim == 2 and actions.shape[1] == 14 and np.isfinite(actions).all()
    assert len(actions) <= CONFIG["episode"]["max_control_steps"]
    out = HERE / "runs" / ("demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True)
    archive_source(out)
    print("RUN_DIRECTORY", out, flush=True)
    sim = Simulation(out)
    error = None
    try:
        for index, action in enumerate(actions):
            sim.step(action)
            if index % 200 == 199:
                print("DEMO", sim.runtime.state(), flush=True)
        assert sim.runtime.result["score"] == 1, sim.runtime.state()
        print("DEMO COMPLETE", sim.runtime.state(), flush=True)
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        sim.save("success" if error is None else "failed", error)
        sim.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    try:
        main()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

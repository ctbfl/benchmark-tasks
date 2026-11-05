"""Replay recorded 14-D robot commands at 30 Hz, without observation inputs."""
from pathlib import Path

import numpy as np

HZ = 30
ACTION_FILE = Path(__file__).with_name("demo_actions.npy")


def recorded_actions():
    """Return fixed [left joints 1..6, grip, right joints 1..6, grip] commands."""
    actions = np.load(ACTION_FILE, allow_pickle=False)
    assert actions.ndim == 2 and actions.shape[1] == 14
    assert 0 < len(actions) <= 120 * HZ and np.isfinite(actions).all()
    actions.flags.writeable = False
    return actions


if __name__ == "__main__":
    import traceback
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False,
        kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    try:
        from record_demo import run
        run(recorded_actions())
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()

"""Measure the matched pair at normal YAM camera resolution before full trials."""
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def main():
    launcher = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8")
    from PIL import Image
    import numpy as np
    from expert import Arm, spin
    from scene import CONFIG
    from run_trial import archive_source
    from simulation import DT, SUBSTEPS, Simulation, write_json

    dest = Path(sys.argv[1])
    dest.mkdir(parents=True)
    archive_source(dest)
    assert len(sys.argv) == 3, "Provide a new output directory and one matched fixture number"
    seed = int(sys.argv[2])
    sim = None
    replay = None
    exit_code = 0
    try:
        sim = Simulation(dest, seed=seed, pair=True)
        sim.frame_stride = SUBSTEPS
        frames_dir = "policy-frames"
        for raw in (False, True):
            key = "raw" if raw else "hard-boiled"
            out = dest / key
            (out / frames_dir).mkdir(parents=True)
            sim.reset(seed, [raw])

            def frame():
                images = sim.capture()
                for name, rgb in images.items():
                    img = Image.fromarray(rgb)
                    img = img.resize(tuple(CONFIG["observation"]["model_input_size_wh"]),
                                     Image.Resampling.BILINEAR)
                    img.save(out / frames_dir / f"{sim.ticks:05d}-{name}.jpg",
                             quality=CONFIG["observation"]["jpeg_quality"])

            try:
                if replay is None:
                    spin([Arm(sim, side, frame) for side in (0, 1)], sim.pose(0)[:3], sim.shell_angle(0))
                    replay = [(sim.rows[step*SUBSTEPS+1]["phase"], action)
                              for step, action in enumerate(sim.actions)]
                else:
                    for sim.phase, action in replay:
                        sim.step(action, frame_hook=frame)
            finally:
                write_json(out / "trace.json", {"gpu_run_id": os.environ["PRODUCTION_GPU_RUN_ID"],
                    "dt": DT, "rows": sim.rows, "native_contact_link_order": sim.jaw_filter_paths})
                np.save(out / "actions.npy", np.asarray(sim.actions))
            print(key, "overloaded", sim.overloaded.tolist(), "final", sim.pose(0), flush=True)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if sim is not None:
            sim.close()
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()

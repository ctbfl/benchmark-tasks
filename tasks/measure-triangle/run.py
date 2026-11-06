"""Usage: python run.py MODE [output_directory]."""

from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
mode = sys.argv[1]
assert mode in ("build", "check", "preview", "expert", "replay", "dream", "adapter-check", "render", "video", "keyboard-check"), mode
out = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else ROOT/"output"/mode
(out/"logs").mkdir(parents=True, exist_ok=True)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=mode in ("preview", "dream", "adapter-check", "render", "video"),
    kit_args="--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6 "
             f"--/log/file={out}/logs/kit.log").app

try:
    if mode == "build":
        from model import build_all
        build_all()
    elif mode in ("check", "preview"):
        from simulation import Simulation
        sim = Simulation(out, cameras=mode == "preview")
        try:
            if mode == "preview":
                from render import screenshots
                screenshots(sim, out)
            sim.check()
        finally:
            sim.close()
    elif mode == "expert":
        from expert import run
        run(out)
    elif mode == "keyboard-check":
        from keyboard import run
        run(out)
    elif mode == "render":
        from render import trace_frames
        trace_frames(out)
    elif mode == "video":
        from render import video
        video(out)
    elif mode == "replay":
        from replay import run
        run(out)
    elif mode in ("dream", "adapter-check"):
        from dream_eval import run
        run(out, check_only=mode == "adapter-check")
except Exception:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
else:
    app.close()

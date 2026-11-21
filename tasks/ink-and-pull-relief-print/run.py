"""Launch a task run. Usage: python run.py {demo,dream,screenshot,render} PATH."""
from pathlib import Path
import sys
import traceback


def main():
    mode, path = sys.argv[1:]
    assert mode in {"demo", "dream", "screenshot", "render"}
    path = Path(path)
    if mode in {"demo", "dream"}:
        path.mkdir(parents=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=mode in {"dream", "screenshot", "render"},
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    try:
        if mode == "demo":
            from expert import run_demo
            run_demo(path)
        elif mode == "dream":
            from dream_eval import run_dream
            run_dream(path)
        else:
            from media import render, screenshot
            (screenshot if mode == "screenshot" else render)(path)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()


if __name__ == "__main__":
    main()

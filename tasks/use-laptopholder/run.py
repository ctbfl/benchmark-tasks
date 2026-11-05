"""Run with shared yam.env: python run.py [validate|demo|exercise|export|benchmark]."""

import sys
import traceback

MODE = sys.argv[1] if len(sys.argv) >= 2 else "validate"
assert MODE in ("validate", "demo", "exercise", "export", "benchmark", "preview", "cameras", "expert", "expert_video", "camera_ready", "camera_ready_preview"), MODE
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=MODE in ("demo","benchmark","preview","cameras","expert_video","camera_ready","camera_ready_preview"),
                  kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app


def main():
    import model
    model.export()
    if MODE == "export":
        print("Exported assets/laptop_holder.usda", flush=True)
        return
    if MODE in ("demo", "exercise"):
        from demo import run
        run(render=MODE=="demo")
        return
    if MODE == "benchmark":
        from benchmark import smoke
        smoke()
        return
    if MODE == "cameras":
        from benchmark import capture_views
        capture_views()
        return
    if MODE == "camera_ready_preview":
        from camera_ready import preview
        preview()
        return
    if MODE in ("expert","expert_video","camera_ready"):
        from expert import run
        run(render=MODE=="expert_video",camera_ready=MODE=="camera_ready")
        return
    if MODE == "preview":
        from preview import run
        run()
        return
    from validate import run
    from runtime import PHYSICS_HZ
    run(int(sys.argv[2]) if len(sys.argv)==3 else PHYSICS_HZ)


try:
    main()
except Exception:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()

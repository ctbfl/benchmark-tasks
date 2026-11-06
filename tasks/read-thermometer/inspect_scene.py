"""Load the actual workstation, settle it, and capture the initial review images."""
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent


def main():
    (HERE / "validation").mkdir(exist_ok=True)
    (HERE / "submission/uploads").mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from PIL import Image
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena.constants import READY
    from yam_arena.station import rig
    from scene import CONFIG, CUP_Y, make_env
    from task_runtime import TaskRuntime

    env = make_env(review=True)
    base = env.unwrapped
    env.reset(seed=CONFIG["episode"]["seed"])
    ready = torch.tensor([[*READY, 1., *READY, 1.]], dtype=torch.float32)
    for _ in range(40):
        env.step(ready)
    runtime = TaskRuntime(env)
    frozen_tick = base.sim._physics_step_count
    summary = {"physics_dt": base.physics_dt, "control_hz": 20,
               "task_state": runtime.state(), "renderer": CONFIG["renderer"],
               "camera": {"name": "Top Camera", "size": [1920, 1080],
                   "position": rig().top_cam.pos, "rotation_xyzw": rig().top_cam.rot_xyzw,
                   "intrinsics": vars(rig().top_cam.lens.resized(1920, 1080))},
               "objects": {}}
    for name in ("stand", "beaker", "thermometer", "clamp", "timer", "calculator"):
        obj = base.scene[name]
        pose = obj.data.root_pose_w.torch[0].cpu().numpy()
        vel = obj.data.root_vel_w.torch[0].cpu().numpy()
        assert np.isfinite(pose).all() and np.isfinite(vel).all(), name
        item = {"root_pose_xyzw": pose.tolist(), "velocity": vel.tolist()}
        if name in ("clamp", "timer", "calculator"):
            item["joint_names"] = obj.joint_names
            item["joint_positions"] = obj.data.joint_pos.torch[0].cpu().numpy().tolist()
        summary["objects"][name] = item
    (HERE / "validation" / "initial_state.json").write_text(json.dumps(summary, indent=2))
    print("INITIAL_STATE", json.dumps(summary["objects"]), flush=True)

    for _ in range(20):
        base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
    for name in CONFIG["observation"]["camera_names"]:
        camera = base.scene[name]
        camera.update(0.05, force_recompute=True)
        pixels = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
        target = HERE / "submission/uploads/screenshot.png" if name == "top_cam" else HERE / f"validation/{name}_initial.png"
        Image.fromarray(pixels).save(target)
        print("SAVED", target, pixels.shape, flush=True)

    def detail(name, eye, target, focal=55):
        cam = UsdGeom.Camera.Define(base.sim.stage, "/World/envs/env_0/" + name)
        cam.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(36)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.002, 20))
        product = rep.create.render_product(str(cam.GetPath()), (1920, 1080))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        for _ in range(16):
            base.sim.render()
            ensure_isaac_rtx_render_update(force=True)
        pixels = rgb.get_data()[..., :3].copy()
        Image.fromarray(pixels).save(HERE / "validation" / (name + ".png"))
        rgb.detach()
        product.destroy()

    detail("lab_detail", (-.32, -.12, 1.15), (.43, .19, .96), 35)
    detail("clamp_detail", (.24, .42, 1.26), (.43, .245, 1.09), 50)
    detail("devices_detail", (.04, -.07, 1.20), (.36, -.235, .84), 35)
    detail("timer_detail", (.16, -.20, 1.00), (.46, -.235, .82), 42)
    detail("thermometer_detail", (.165, CUP_Y, .96), (.43, CUP_Y, .96), 70)
    assert base.sim._physics_step_count == frozen_tick, "Capture advanced the simulation"
    env.close()


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

"""Replay measured 30 Hz poses through the calibrated YAM cameras."""

from pathlib import Path
import sys
import traceback

FPS = 30
VIEWS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


def products(base, screenshot):
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(base.sim.stage, "/Observer")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.04,-.63,1.58),
        Gf.Vec3d(.36,-.055,1.025), Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(39)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.012,20))
    observer = rep.create.render_product(str(camera.GetPath()), (1920,1080))
    feed = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    feed.attach([observer])
    if not screenshot:
        return observer, feed, None, None
    # The same native top-camera prim and lens at three times the pixel density.
    top_sensor = base.scene["top_cam"]
    from isaaclab.sim.utils import find_matching_prims
    prim_path = str(find_matching_prims(top_sensor.cfg.prim_path, base.sim.stage)[0].GetPath())
    top = rep.create.render_product(prim_path, (1920,1080))
    top_feed = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    top_feed.attach([top])
    return observer, feed, top, top_feed


def run(out, screenshot):
    import json
    import carb
    import carb.eventdispatcher
    import imageio_ffmpeg
    import numpy as np
    import omni.physics.tensors as tensors
    import omni.physx
    import omni.timeline
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    from scene import ARMS, CAMERAS, CLAMPS, make_env
    from trial import TRAJECTORY_COLUMNS, array, write_json

    result = json.loads((out / "result.json").read_text())
    binary = out / "trajectory-f64.bin"
    rows = np.fromfile(binary, dtype="<f8").reshape(-1, TRAJECTORY_COLUMNS)
    assert np.allclose(np.diff(rows[:,0]), 1 / FPS)
    assert abs(rows[-1,0] - result["seconds"]) <= 1 / FPS
    timestep_count = round(result["seconds"] * FPS)
    rows = rows[:timestep_count]
    assert len(rows) == timestep_count
    (out / "videos").mkdir(exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    env = make_env(out, cameras=True)
    env.reset()
    base = env.unwrapped
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    observer, feed, top, top_feed = products(base, screenshot)
    settings = carb.settings.get_settings()
    event_dispatch = carb.eventdispatcher.get_eventdispatcher()
    timeline = omni.timeline.get_timeline_interface()
    original = out / "videos" / f"{result['kind']}-original.mp4"
    writer = imageio_ffmpeg.write_frames(str(original), (1920,1440), fps=FPS,
        codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2, output_params=[
            "-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)

    def tensor(values):
        return torch.tensor(values[None], dtype=torch.float32)

    def label(draw, x, y, title, subtitle=None):
        width = int(font.getlength(title)) + 26
        draw.rectangle((x, y, x+width-1, y+(81 if subtitle else 50)-1), fill="black")
        draw.text((x+13,y+8),title,font=font,fill="white")
        if subtitle:
            draw.text((x+13,y+50),subtitle,font=small,fill=(175,215,240))

    errors = []
    try:
        for i, row in enumerate(rows):
            timeline.set_current_time(row[0])
            for name, q in zip(ARMS, (row[1:9],row[9:17])):
                base.scene[name].write_joint_state_to_sim(tensor(q), torch.zeros((1, 8)))
            base.scene["so101"].write_root_pose_to_sim(tensor(row[17:24]))
            for k,name in enumerate(CLAMPS):
                offset = 24 + 8*k
                clamp = base.scene[name]
                clamp.write_root_pose_to_sim(tensor(row[offset:offset+7]))
                clamp.write_joint_state_to_sim(tensor(row[offset+7:offset+8]), torch.zeros((1, 1)))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1 / FPS)
            error = max(np.max(np.abs(array(base.scene["so101"].data.root_pose_w)[:3] - row[17:20])),
                        *(np.max(np.abs(array(base.scene[n].data.root_pose_w)[:3] - row[24+8*k:27+8*k]))
                          for k,n in enumerate(CLAMPS)))
            errors.append(float(error))
            assert error < .0001
            settings.set("/rtx-transient/post/dlss/forceParamReset", True)
            event_dispatch.dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
            for _ in range(8 if i == 0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pixels = feed.get_data()[...,:3]
            assert pixels.shape == (1080,1920,3)
            frame = Image.new("RGB", (1920,1440))
            frame.paste(Image.fromarray(pixels))
            for k, name in enumerate(CAMERAS):
                sensor = base.scene[name]
                sensor.update(1 / FPS, force_recompute=True)
                rgb = array(sensor.data.output["rgb"])[...,:3]
                assert rgb.shape == (360,640,3)
                frame.paste(Image.fromarray(rgb), (k*640,1080))
            if i == 0 and screenshot:
                image = top_feed.get_data()[...,:3]
                assert image.shape == (1080,1920,3) and image.std() > 10
                Image.fromarray(image).save(screenshot)
                top_feed.detach()
                top.destroy()
            draw = ImageDraw.Draw(frame)
            label(draw, 0, 0, "External Observer View", "Not a robot camera")
            for k, title in enumerate(VIEWS):
                label(draw, k*640, 1080, title)
            if i % 150 == 0 or i == len(rows)-1:
                frame.save(out / "frames" / f"frame-{i:05d}.png")
                print("Rendered",i+1,"of",len(rows),flush=True)
            writer.send(np.asarray(frame))
    finally:
        writer.close()
        feed.detach()
        observer.destroy()
        env.close()
    report = dict(frames=len(rows), fps=FPS, resolution=[1920,1440],
                  source=str(binary), max_translation_error_m=max(errors))
    if "gpu_run_id" in result:
        report["source_gpu_run_id"] = result["gpu_run_id"]
    write_json(out / "render.json", report)


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    assert len(sys.argv) in (2,3)
    app = AppLauncher(headless=True, enable_cameras=True, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app
    try:
        run(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

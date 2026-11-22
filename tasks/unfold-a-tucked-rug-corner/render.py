"""Draw recorded robot joints and measured cloth nodes without advancing physics."""
from pathlib import Path
import json
import sys
import traceback
import numpy as np

assert len(sys.argv) in (2, 3), "Usage: python render.py RUN_DIRECTORY [preview]"
if len(sys.argv) == 3:
    assert sys.argv[2] == "preview"
PREVIEW = len(sys.argv) == 3
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(RUN / "source"))
from scene import SIM
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device=SIM["device"], enable_cameras=True,
    kit_args="--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app


def render():
    import carb
    import carb.eventdispatcher
    import imageio_ffmpeg
    import omni.replicator.core as rep
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena import YAM_URDF, station
    from yam_arena.robot import asset_link_paths
    from yam_arena.urdf import forward_kinematics
    from cloth import update_visual
    from scene import make_env, RUG

    with np.load(RUN / "trajectory.npz") as archive:
        data = {name: archive[name] for name in ("times", "robots", "cloth", "faces")}
    schema = json.loads((RUN / "schema.json").read_text())
    times = data["times"]
    phases = {path.stem: json.loads(path.read_text())["time_s"] for path in (RUN / "frames").glob("*.json")}
    phase_frames = {int(np.argmin(abs(times - t))): label for label, t in phases.items()}
    indices = sorted(phase_frames) if PREVIEW else range(1, len(times))
    if not PREVIEW:
        assert abs((len(times)-1)/30 - schema["duration_s"]) < 1e-8
    env = make_env(cameras=True)
    env.reset(seed=0)
    base = env.unwrapped
    camera = UsdGeom.Camera.Define(base.sim.stage, "/RecordedDemoObserver")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    eye, target = (.78, -.62, 1.20), (.43, 0., .77)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(38.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 20.))
    product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([product])
    link_paths, ops = asset_link_paths(), {}
    for side in ("Left", "Right"):
        for name, path in link_paths.items():
            xf = UsdGeom.Xformable(base.sim.stage.GetPrimAtPath("/World/envs/env_0/" + side + "Arm" + path))
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            ops[side.lower(), name] = xf.AddTransformOp(opSuffix="recorded")
    views = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"), ("right_cam", "Right Wrist Camera"))
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    videos = RUN / "videos"
    videos.mkdir(exist_ok=True)
    writer = None
    if not PREVIEW:
        writer = imageio_ffmpeg.write_frames(str(videos / "demo-original.mp4"), (1920,1440), fps=30,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
        writer.send(None)
    max_error, count = 0., 0
    try:
        for i in indices:
            expected = []
            for side_index, side in enumerate(("left", "right")):
                joints = dict(zip(schema["robot_joint_names"][side_index], data["robots"][i, side_index], strict=True))
                fk = forward_kinematics(YAM_URDF, joints)
                for name in link_paths:
                    transform = fk[name].copy()
                    transform[:3,3] += station.BASES[side]
                    op = ops[side, name]
                    op.Set(Gf.Matrix4d(*transform.T.ravel()))
                    expected.append((op.GetAttr().GetPrim(), transform[:3,3]))
            update_visual(base.sim.stage, RUG, data["cloth"][i], data["faces"])
            cache = UsdGeom.XformCache()
            error = max(np.linalg.norm(np.asarray(cache.GetLocalToWorldTransform(prim).ExtractTranslation()) - position)
                        for prim, position in expected)
            max_error = max(max_error, float(error))
            assert error < 1e-6
            carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
            tick = base.sim._physics_step_count
            for _ in range(8 if count == 0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            frame = Image.new("RGB", (1920,1440))
            observer = np.ascontiguousarray(rgb.get_data()[..., :3])
            assert observer.shape == (1080,1920,3) and observer.std() > 10
            frame.paste(Image.fromarray(observer))
            for j, (name, title) in enumerate(views):
                sensor = base.scene[name]
                sensor.update(1/30, force_recompute=True)
                pixels = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
                assert pixels.shape == (360,640,3) and pixels.std() > 10
                frame.paste(Image.fromarray(pixels), (j*640,1080))
            assert base.sim._physics_step_count == tick, "Rendering advanced physics"
            draw = ImageDraw.Draw(frame)
            for x, y, text in [(0,0,"External Observer View"), *[(j*640,1080,title) for j, (_,title) in enumerate(views)]]:
                draw.rectangle((x,y,x+int(font.getlength(text))+31,y+(84 if y == 0 else 52)-1), fill="black")
                draw.text((x+16,y+10), text, font=font, fill="white")
                if y == 0:
                    draw.text((16,50), "Not a robot camera", font=small, fill=(175,215,240))
            if count == 0:
                frame.save(RUN / "frames/replay-first.png")
            if i == len(times) - 1:
                frame.save(RUN / "frames/replay-final.png")
            if i in phase_frames:
                frame.save(RUN / "frames" / ("replay-" + phase_frames[i] + ".png"))
            if writer is not None:
                writer.send(np.asarray(frame))
            count += 1
            if count % 150 == 0 or PREVIEW:
                print(f"RENDER {count} time={times[i]:.3f}", flush=True)
        (RUN / ("preview-render.json" if PREVIEW else "render.json")).write_text(json.dumps({
            "frames": count, "fps": 30, "resolution": [1920,1440], "max_robot_pose_error_m": max_error,
            "source": "Measured trajectory.npz; no physics steps or policy execution",
            "observer_eye": eye, "observer_target": target, "observer_fixed": True,
            "speed": 1., "preview_only": PREVIEW}, indent=2) + "\n")
    finally:
        if writer is not None:
            writer.close()
        rgb.detach()
        product.destroy()
        env.close()


try:
    render()
except BaseException:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()

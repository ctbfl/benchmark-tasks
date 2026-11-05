"""Render a saved Dream trial at 30 FPS, without inference or resimulating actions."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dream_eval import ARMS, HERE, WaterEmbodiment, reset_rtx_history, write_json
from water import WaterTask

FPS = 30
CAMERAS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
           ("right_cam", "Right Wrist Camera"))
EYE, LOOK_AT, FOCAL = (-.75, -.55, 1.85), (.23, 0, 1.02), 34.


def annotate(pixels, title, *, observer=False):
    panel = Image.fromarray(pixels)
    draw = ImageDraw.Draw(panel)
    scale = 1
    font = ImageFont.load_default(size=32 * scale)
    subtitle_font = ImageFont.load_default(size=24)
    width = draw.textlength(title, font=font)
    if observer:
        width = max(width, draw.textlength("Not a robot camera", font=subtitle_font))
    draw.rectangle((0, 0, 32 * scale + width, (84 if observer else 52) * scale),
                   fill=(20, 26, 34))
    draw.text((16 * scale, 10 * scale), title, font=font, fill="white")
    if observer:
        draw.text((16, 50), "Not a robot camera", font=subtitle_font, fill=(175, 215, 240))
    return panel


def render(run, check_only):
    import imageio_ffmpeg
    schema = json.loads((run / "replay-schema.json").read_text())
    manifest = json.loads((run / "run.json").read_text())
    # Scoring tolerance can change without changing the recorded water dynamics.
    configs = [{key: value for key, value in runpy.run_path(str(path)).items()
                if not key.startswith("__") and key != "TOLERANCE_G"}
               for path in (HERE / "config.py", run / "source/config.py")]
    assert configs[0] == configs[1], "Physical configuration differs from the saved run"
    for name in ("water.py", "model.py", "surfaces.py", "yam_preview.py"):
        assert (HERE / name).read_bytes() == (run / "source" / name).read_bytes(), name
    states = np.fromfile(run / "physics-f64.bin", dtype=schema["dtype"]).reshape(-1, len(schema["columns"]))
    assert np.allclose(np.diff(states[:, 0]), schema["physics_dt"], atol=1e-10)
    stride = round(1 / (FPS * schema["physics_dt"]))
    assert abs(stride * schema["physics_dt"] - 1 / FPS) < 1e-12
    indices = ([0, *range(56 * 240, 58 * 240 + 1, stride), len(states) - 1]
               if check_only else range(0, len(states), stride))
    out = run / ("rtx-check-reset" if check_only else "camera-ready-rtx")
    out.mkdir()
    (out / "frames").mkdir()
    videos = run / "videos"
    env = WaterEmbodiment(out, camera_size=(640, 360), asset_dir=run / "source/assets")
    writers, product, rgb = [], None, None
    exit_code = 0
    try:
        env._ensure_sim()
        import carb.settings
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        import torch
        from pxr import Gf, UsdGeom
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        stage = omni.usd.get_context().get_stage()
        settings = carb.settings.get_settings()
        assert settings.get("/rtx/rendermode") == "RealTimePathTracing"
        assert settings.get("/rtx/post/aa/op") == 4
        rep.settings.carb_settings("/rtx/scenePartitioning/showAllPartitionsByDefault", True)
        camera = UsdGeom.Camera.Define(stage, "/World/ReplayObserver")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*EYE), Gf.Vec3d(*LOOK_AT), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(FOCAL)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
        product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        path = videos / ("realtime-check-reset.mp4" if check_only else "dream-four-camera-rtx.mp4")
        assert not path.exists(), path
        writer = imageio_ffmpeg.write_frames(str(path), (1920, 1440), fps=FPS,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-crf", "20", "-preset", "fast", "-threads", "4",
                           "-movflags", "+faststart"])
        writer.send(None)
        writers.append(writer)

        water = WaterTask(manifest["target_g"])
        previous_index = 0
        max_mass_error = 0.
        comparison = []
        tick = time.monotonic()
        saved_at = {0, 30 * 240, 60 * 240, 90 * 240, len(states) - 1}
        if check_only:
            saved_at |= {56 * 240, 57 * 240, 58 * 240}
        for frame, index in enumerate(indices):
            for j in range(previous_index + 1, index + 1):
                water.step((states[j - 1, 1] + states[j, 1]) / 2, schema["physics_dt"])
            previous_index = index
            row = states[index]
            omni.timeline.get_timeline_interface().set_current_time(float(row[0]))
            max_mass_error = max(max_mass_error, abs(water.cup_g - row[3]),
                                 abs(water.airborne_g - row[4]), abs(water.spilled_g - row[6]))
            assert max_mass_error < 1e-6
            for name in ARMS:
                robot = env._base.scene[name]
                positions = [schema["columns"].index(f"{name}.pos.{joint}") for joint in robot.joint_names]
                velocities = [schema["columns"].index(f"{name}.vel.{joint}") for joint in robot.joint_names]
                robot.write_joint_state_to_sim_index(
                    position=torch.tensor(row[positions][None, :], dtype=torch.float32),
                    velocity=torch.tensor(row[velocities][None, :], dtype=torch.float32))
            env.set_valve_for_reset(row[1])
            env.physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            env.visual.update(water)
            env._base.scene.update(1 / FPS)
            reset_rtx_history()
            # No physics step: every rendered state comes directly from the saved trajectory.
            for _ in range(4 if frame == 0 else 1):
                env._base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pictures = {"observer": np.ascontiguousarray(rgb.get_data()[..., :3])}
            for name, _ in CAMERAS:
                camera_sensor = env._base.scene[name]
                camera_sensor.update(1 / FPS, force_recompute=True)
                pictures[name] = camera_sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                assert pictures[name].shape == (360, 640, 3)
            assert pictures["observer"].shape == (1080, 1920, 3)
            composite = Image.new("RGB", (1920, 1440))
            observer = annotate(pictures["observer"], "External Observer View", observer=True)
            composite.paste(observer)
            for column, (name, title) in enumerate(CAMERAS):
                composite.paste(annotate(pictures[name], title), (column * 640, 1080))
            if index in saved_at:
                label = f"{row[0]:07.3f}s"
                for name, pixels in pictures.items():
                    Image.fromarray(pixels).save(out / "frames" / f"{label}-{name}.png")
                composite.save(out / "frames" / f"{label}-preview.png")
                original_tick = min(round(row[0] * 20), 2399)
                for name, _ in CAMERAS:
                    original = np.asarray(Image.open(run / "dream" / name / f"{original_tick:06d}.jpg"))
                    reduced = np.asarray(Image.fromarray(pictures[name]).resize((320, 176), Image.Resampling.BILINEAR))
                    comparison.append(dict(time_s=float(row[0]), camera=name,
                        resized_rgb_mae=float(np.abs(original.astype(float) - reduced).mean())))
            writers[0].send(np.asarray(composite))
            if frame % 30 == 0:
                elapsed = time.monotonic() - tick
                progress = dict(frame=frame, total_frames=len(indices), time_s=float(row[0]),
                                elapsed_s=elapsed, max_mass_error_g=max_mass_error,
                                observed_cup_g=water.cup_g)
                write_json(out / "progress.json", progress)
                print(f"Replay frame {frame + 1}/{len(indices)}; {row[0]:.1f}s; "
                      f"{elapsed:.1f}s wall time", flush=True)
        write_json(out / "render.json", dict(api_calls=0, source=str(run),
            trajectory_sha256=hashlib.sha256((run / "physics-f64.bin").read_bytes()).hexdigest(),
            fps=FPS, frame_count=len(indices), camera_sizes=dict(observer=[1920, 1080], robot=[640, 360]),
            renderer="RTX Real-Time 2.0, DLAA; reset temporal history; no frame generation",
            observer=dict(eye=EYE, look_at=LOOK_AT, focal_mm=FOCAL, fixed=True),
            max_mass_error_g=max_mass_error, comparisons=comparison,
            note="Offline replay of measured states; policy inputs were 320x176. "
                 "Overflow mass is counted but runoff is not animated."))
        print("REPLAY CHECK PASSED" if check_only else "REPLAY RENDER COMPLETE", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        for writer in writers:
            writer.close()
        if rgb is not None:
            rgb.detach()
        if product is not None:
            product.destroy()
        env.close()
        if env.app is not None:
            env.app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    assert len(sys.argv) in (2, 3) and (len(sys.argv) == 2 or sys.argv[2] == "--check"), \
        "Usage: python replay_dream.py RUN_DIR [--check]"
    sys.exit(render(Path(sys.argv[1]).resolve(), len(sys.argv) == 3))

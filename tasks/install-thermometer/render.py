"""Render measured states at 30 FPS without executing a policy or advancing physics."""
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
         ("right_cam", "Right Wrist Camera"))


def render(out):
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.physx
    import omni.replicator.core as rep
    import omni.timeline
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Sdf
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_arena.embodiment import _pinhole
    from yam_arena.station import rig
    from scene import CONFIG, ROOT, PLACEMENTS, make_env

    out = Path(out)
    schema = json.loads((out / "schema.json").read_text())
    frames = json.loads((out / "frame_states.json").read_text())[1:]
    rows = np.load(out / "trajectory.npz")["states"][8::8]
    assert len(rows) == len(frames) and np.allclose(np.diff(rows[:, 0]), 1/30)
    manifest = json.loads((out / "source_manifest.json").read_text())
    critical = ["scene.py", "dream-defaults.json", "task_runtime.py", "lab_glass.py"]
    critical += [name for name in manifest if name.startswith("assets/") and Path(name).suffix in (".usda", ".png")]
    for name in critical:
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == manifest[name], name
    for name in ("videos", "frames"):
        (out / name).mkdir(exist_ok=True)
    env = make_env()
    env.reset(seed=0)
    base = env.unwrapped
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    frozen_tick = base.sim._physics_step_count
    top = rig(CONFIG["station"]["top_camera"]).top_cam
    assert top.convention == "opengl"
    lens = _pinhole(top, 1080, 1920, near=.05)
    lens.vertical_aperture = lens.horizontal_aperture * 1080 / 1920
    camera = lens.func(ROOT + "/Observer", lens, translation=top.pos, orientation=top.rot_xyzw)
    camera.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([product])
    writer = imageio_ffmpeg.write_frames(str(out / "videos/original.mp4"), (1920, 1440),
        fps=30, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                       "-threads", "4", "-movflags", "+faststart"])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    tensor = lambda value: torch.tensor(np.asarray(value)[None], dtype=torch.float32)
    cameras, max_error = {}, 0.
    checkpoints = {0, len(rows)-1, *range(10*30, len(rows), 10*30), 17*30, 24*30, 34*30}
    try:
        for index, (row, state) in enumerate(zip(rows, frames)):
            value = lambda key: row[slice(*schema[key])]
            omni.timeline.get_timeline_interface().set_current_time(float(row[0]))
            for side in ("left", "right"):
                base.scene[side + "_robot"].write_joint_state_to_sim(tensor(value(side + "_q")), tensor(value(side + "_dq")))
            for name, *_ in PLACEMENTS:
                obj = base.scene[name]
                obj.write_root_pose_to_sim(tensor(value(name + "_pose")))
                obj.write_root_velocity_to_sim(tensor(value(name + "_velocity")))
                if name + "_q" in schema:
                    obj.write_joint_state_to_sim(tensor(value(name + "_q")), tensor(value(name + "_dq")))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1/30)
            for name in ("clamp",):
                actual = base.scene[name].data.body_link_pose_w.torch[0].cpu().numpy()
                expected = value(name + "_links").reshape(-1, 7)
                error = float(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=1).max())
                max_error = max(max_error, error)
                assert error < .0001, (name, row[0], error)
            for _ in range(12 if index == 0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pictures = {"observer": np.ascontiguousarray(rgb.get_data()[..., :3])}
            for name, _ in VIEWS:
                sensor = base.scene[name]
                sensor.update(1/30, force_recompute=True)
                pictures[name] = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                assert pictures[name].shape == (360, 640, 3)
                if index == 0:
                    cameras[name] = {"resolution": [640, 360],
                        "intrinsics": sensor.data.intrinsic_matrices.torch[0].cpu().tolist()}
            assert pictures["observer"].shape == (1080, 1920, 3)
            frame = Image.new("RGB", (1920, 1440))
            frame.paste(Image.fromarray(pictures["observer"]))
            for column, (name, _) in enumerate(VIEWS):
                frame.paste(Image.fromarray(pictures[name]), (column*640, 1080))
            draw = ImageDraw.Draw(frame)
            labels = [(0, 0, "External Observer View", True)] + [
                (i*640, 1080, title, False) for i, (_, title) in enumerate(VIEWS)]
            for x, y, title, observer in labels:
                width = int(font.getlength(title)) + 32
                draw.rectangle((x, y, x+width-1, y+(84 if observer else 52)-1), fill="black")
                draw.text((x+16, y+10), title, font=font, fill="white")
                if observer:
                    draw.text((x+16, y+50), "Not a robot camera", font=small, fill=(175, 215, 240))
            if index in checkpoints:
                frame.save(out / "frames" / f"frame-{index:05d}.png")
                for name, picture in pictures.items():
                    Image.fromarray(picture).save(out / "frames" / f"{name}-{index:05d}.png")
            writer.send(np.asarray(frame))
            if index % 150 == 0:
                print("RENDER", index, len(rows), "time", row[0], flush=True)
        assert base.sim._physics_step_count == frozen_tick
        (out / "render.json").write_text(json.dumps({"frames": len(rows), "fps": 30,
            "resolution": [1920, 1440], "duration_s": len(rows)/30, "playback_speed": 1.,
            "max_link_position_error_m": max_error, "cameras": cameras,
            "observer": {"source_camera": "top_cam", "position": top.pos,
                "rotation_xyzw": top.rot_xyzw, "intrinsics": vars(top.lens.resized(1920, 1080)),
                "resolution": [1920, 1080], "fixed": True},
            "source": "Synchronized measured states; no policy calls or physical resimulation"}, indent=2) + "\n")
    finally:
        writer.close()
        rgb.detach()
        product.destroy()
        env.close()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python render.py RUN_DIRECTORY"
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8").app
    try:
        render(sys.argv[1])
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

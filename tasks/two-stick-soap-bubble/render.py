"""Render synchronized 30 FPS RGB from the measured trajectory, with physics paused."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from geometry import ROOT
from scene import CAMERAS, SEED, make_env
from score import score
from simulation import make_body_view, render_views, source_manifest, start_app, write_json

OBSERVER_EYE = (1.09, -.69, 1.42)
OBSERVER_TARGET = (.41, .0, 1.005)
TITLES = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


def render(out):
    import imageio_ffmpeg
    import omni.physics.tensors
    import omni.physx
    import omni.replicator.core as rep
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom

    from visual import FilmMesh

    measured = np.load(out / "trajectory.npz")
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "success"
    graded = score(out)
    actual = json.loads((out / "source-manifest.json").read_text())
    present = source_manifest()
    for name in ("geometry.py", "scene.py", "simulation.py",
                 "dream-defaults.json", "assets/wand-kit.usda"):
        assert name not in actual or present[name] == actual[name], name

    videos = out / "videos"
    videos.mkdir()
    env = make_env(out / "render-isaac")
    env.reset(seed=SEED)
    base = env.unwrapped
    physics, bodies, order = make_body_view()
    robots = [base.scene[f"{side}_robot"] for side in ("left", "right")]
    frozen = base.sim._physics_step_count
    observer = UsdGeom.Camera.Define(base.sim.stage, ROOT + "/Observer")
    observer.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*OBSERVER_EYE),
        Gf.Vec3d(*OBSERVER_TARGET), Gf.Vec3d(0,0,1)).GetInverse())
    observer.CreateFocalLengthAttr(27.)
    observer.CreateHorizontalApertureAttr(36.)
    observer.CreateVerticalApertureAttr(20.25)
    observer.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
    observer.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    film = FilmMesh(base.sim.stage)

    products, channels = [], []
    def attach(path):
        product = rep.create.render_product(str(path), (1920, 1080))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        products.append(product)
        channels.append(rgb)
        return rgb

    observer_rgb = attach(observer.GetPath())
    sharp_top = attach("/World/envs/env_0/TopCam")
    writer = imageio_ffmpeg.write_frames(str(videos / "original.mp4"), (1920, 1440), fps=30,
        codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
    writer.send(None)
    header = ImageFont.load_default(size=32)
    subtitle = ImageFont.load_default(size=24)
    frames = len(measured["film"])-1
    reversed_order = np.argsort(order)
    indices = np.arange(len(order), dtype=np.int32)
    shots = {1, frames, *range(150, frames, 150)}
    shots.update(round(e["time_s"]*30) for e in graded["events"]
                 if e["event"] in {"film_formed", "bubble_released", "bubble_floating"})
    (out / "frames").mkdir()
    try:
        for i, (joint, poses, bubble) in enumerate(zip(measured["joints"], measured["objects"], measured["film"])):
            for robot, position in zip(robots, joint):
                q = torch.as_tensor(position[None], dtype=torch.float32)
                robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            bodies.set_transforms(np.ascontiguousarray(poses[reversed_order]), indices)
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1/30)
            film.update(poses, bubble)
            if i == 0:
                for _ in range(9):
                    render_views(base)
                initial = sharp_top.get_data()[...,:3]
                assert initial.shape == (1080,1920,3) and initial.std() > 8
                Image.fromarray(initial).save(out / "screenshot.png")
                sharp_top.detach()
                products[1].destroy()
                continue
            views = render_views(base)
            overview = observer_rgb.get_data()[...,:3]
            assert overview.shape == (1080,1920,3)
            canvas = Image.new("RGB", (1920,1440))
            canvas.paste(Image.fromarray(overview))
            for k, name in enumerate(CAMERAS):
                canvas.paste(Image.fromarray(views[name]), (k*640,1080))
            draw = ImageDraw.Draw(canvas)
            labels = [(0,0,"External Observer View")] + [(k*640,1080,title) for k,title in enumerate(TITLES)]
            for x, y, title in labels:
                draw.rectangle((x,y,x+int(header.getlength(title))+24,y+(79 if y==0 else 49)), fill="black")
                draw.text((x+12,y+7), title, font=header, fill="white")
            draw.text((12,48), "Not a robot camera", font=subtitle, fill=(180,220,249))
            if i in shots:
                canvas.save(out / "frames" / f"{i:06d}.jpg", quality=94)
            writer.send(np.asarray(canvas))
            if i % 150 == 0:
                print("RENDER", i, frames, flush=True)
        assert base.sim._physics_step_count == frozen
        assert abs(frames/30-result["simulation_duration_s"]) <= 1/30
        write_json(out / "render.json", {"source_run": result["gpu_run_id"], "frames": frames, "fps": 30,
                                         "observer_eye": OBSERVER_EYE, "observer_target": OBSERVER_TARGET,
                                         "resolution": [1920,1440]})
    finally:
        writer.close()
        observer_rgb.detach()
        products[0].destroy()
        env.close()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python render.py RUN_DIRECTORY"
    app = start_app()
    code = 0
    try:
        render(Path(sys.argv[1]).resolve())
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)

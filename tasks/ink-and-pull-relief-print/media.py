"""Capture initial state and synchronized recorded-state views."""
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import omni.physics.tensors as tensors
import omni.physx
import omni.replicator.core as rep
import torch
from isaaclab.sim.utils import get_current_stage_id
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom

from scene import CAMERAS, HERE, ROOT
from simulation import FilmView, Simulation, as_array, camera_refresh, write_json

EYE = (.72,.51,1.41)
TARGET = (.319,.005,.81)
VIEWS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


def screenshot(path):
    sim = Simulation(path.parent, cameras=True, screenshot=True)
    try:
        for _ in range(12):
            top = sim.capture(("top_cam",))["top_cam"]
        assert top.shape == (1080, 1920, 3) and top.std() > 15.
        Image.fromarray(top).save(path)
    finally:
        sim.close()


def observer(base, focal_mm):
    cam = UsdGeom.Camera.Define(base.sim.stage, ROOT + "/Observer")
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*EYE), Gf.Vec3d(*TARGET), Gf.Vec3d(0,0,1)).GetInverse())
    cam.CreateFocalLengthAttr(focal_mm)
    cam.CreateHorizontalApertureAttr(36.)
    cam.CreateVerticalApertureAttr(20.25)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.005, 30.))
    prim = cam.GetPrim()
    prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    product = rep.create.render_product(str(prim.GetPath()), (1920, 1080))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([product])
    return product, annotator


def render(out: Path):
    original = json.loads((out/"source_manifest.json").read_text())
    for name in ("geometry.py", "ink.py", "scoring.py", "scene.py", "simulation.py", "dream-defaults.json",
                 *(key for key in original if key.startswith("assets/"))):
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == original[name], name
    data = np.load(out/"trajectory.npz")
    trajectory = {k:data[k] for k in data.files if k.endswith(("_q","_pose")) or k.startswith("ink_")}
    time = data["time_s"]
    assert len(time) > 1 and np.allclose(np.diff(time), 1/30)
    proof = out/"frames"
    proof.mkdir(exist_ok=True)
    videos = out/"videos"
    videos.mkdir(exist_ok=True)
    sim = Simulation(out, cameras=True)
    base = sim.base
    phys = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    tick = base.sim._physics_step_count
    visual = FilmView(base.sim.stage)
    kind = json.loads((out/"result.json").read_text())["kind"]
    focal_mm = {"demo": 34., "dream": 43.}[kind]
    product, rgb = observer(base, focal_mm)
    writer = imageio_ffmpeg.write_frames(str(videos/"original.mp4"), (1920,1440),
        fps=30, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-an","-profile:v","high","-crf","17","-preset","fast",
                       "-threads","4","-movflags","+faststart"])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    snapshots = {0,len(time)-1,len(time)//3,2*len(time)//3}
    actors = dict(zip(("left","right","brayer","sheet"), (*sim.robots,*sim.props), strict=True))
    try:
        for i in range(len(time)):
            for name, obj in actors.items():
                if name in {"brayer","sheet"}:
                    root = torch.tensor(trajectory[name+"_pose"][i][None])
                    obj.write_root_pose_to_sim(root)
                q = torch.tensor(trajectory[name+"_q"][i][None])
                obj.write_joint_state_to_sim(q, torch.zeros_like(q))
            phys.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1/30)
            visual.paint([trajectory[f"ink_{j}"][i] for j in range(6)])
            for _ in range(12 if i == 0 else 1):
                camera_refresh(base)
            picture = rgb.get_data()[..., :3]
            assert picture.shape == (1080,1920,3) and picture.std() > 10.
            composite = Image.new("RGB", (1920,1440))
            composite.paste(Image.fromarray(picture), (0,0))
            for j, name in enumerate(CAMERAS):
                sensor = base.scene[name]
                sensor.update(1/30, force_recompute=True)
                image = as_array(sensor.data.output["rgb"])[..., :3]
                assert image.shape == (360,640,3)
                composite.paste(Image.fromarray(image), (640*j,1080))
            draw = ImageDraw.Draw(composite)
            labels = [(0,0,"External Observer View"), *[(640*j,1080,name) for j,name in enumerate(VIEWS)]]
            for x,y,name in labels:
                tall = 82 if y == 0 else 46
                draw.rectangle((x,y,x+int(font.getlength(name))+26,y+tall-1), fill="black")
                draw.text((x+13,y+7), name, font=font, fill="white")
                if y == 0:
                    draw.text((x+13,y+47), "Not a robot camera", font=small, fill=(171,218,247))
            if i in snapshots:
                composite.save(proof / f"frame-{i:05d}.png")
            writer.send(np.asarray(composite))
            if i % 150 == 0:
                print("RENDER", i, len(time), flush=True)
        assert base.sim._physics_step_count == tick
        write_json(out/"render.json", {"frames":len(time), "fps":30, "duration_s":len(time)/30,
                                       "viewpoint_eye":EYE, "viewpoint_target":TARGET,
                                       "observer_focal_length_mm":focal_mm,
                                       "trajectory_sha256":hashlib.sha256((out/"trajectory.npz").read_bytes()).hexdigest()})
    finally:
        writer.close()
        rgb.detach()
        product.destroy()
        sim.close()

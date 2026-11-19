"""Render measured robot poses, film and drops from a completed trial at 30 FPS."""
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from honey import Honey
from world import CAMERAS, DT, HERE, World

FPS = 30
LABELS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")
EYE = (1.025, -.63, 1.205)
LOOK = (.437, -.141, .869)
FOCAL_MM = 48.


def annotate(stage, path, size):
    import omni.replicator.core as rep

    assert stage.GetPrimAtPath(path).IsValid(), path
    product = rep.create.render_product(path, size)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([product])
    return product, annotator


def image(annotator):
    return np.ascontiguousarray(annotator.get_data()[..., :3])


def mosaic(observer, station, font, small):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (1920, 1440))
    canvas.paste(Image.fromarray(observer))
    for i, name in enumerate(CAMERAS):
        canvas.paste(Image.fromarray(station[name]), (i*640, 1080))
    draw = ImageDraw.Draw(canvas)
    labels = [(0, 0, "External Observer View"), *[(i*640, 1080, s) for i, s in enumerate(LABELS)]]
    for index, (x,y,label) in enumerate(labels):
        width = int(font.getlength(label))+29
        draw.rectangle((x, y, x+width-1, y+(83 if index == 0 else 52)-1), fill="black")
        draw.text((x+14,y+9), label, font=font, fill="white")
        if index == 0:
            draw.text((x+14,y+52), "Not a robot camera", font=small, fill=(175,215,240))
    return canvas


def render(out, screenshot):
    import carb
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.physx
    from PIL import Image, ImageFont
    from pxr import Gf, Sdf, UsdGeom, UsdUtils
    from visuals import HoneyVisual

    run = json.loads((out/"run.json").read_text())
    result = json.loads((out/"result.json").read_text())
    for name, digest in run["source_sha256"].items():
        if name not in ("render.py", "compare.py", "wet_regrasp.py"):
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name
    kind = result["kind"]
    rows = np.load(out/"trajectory.npy")
    assert np.allclose(np.diff(rows[:, 0]), DT)
    step = round(1/(DT*FPS))
    assert np.isclose(step*DT, 1/FPS)
    n_frames = (len(rows)+step-2)//step
    videos = out/"videos"
    videos.mkdir()
    frames = out/"frames"
    frames.mkdir()
    world = World(out, cameras=True)
    world.reset(run["config"]["episode"]["seed"])

    base = world.base
    stage = base.sim.stage
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    assert stage_id >= 0
    view = tensors.create_simulation_view("numpy", stage_id=stage_id)
    camera = UsdGeom.Camera.Define(stage, "/ObserverCam")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*EYE),
        Gf.Vec3d(*LOOK), Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(FOCAL_MM)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
    product, observer = annotate(stage, "/ObserverCam", (1920,1080))
    top_product = top = None
    if screenshot:
        top_product, top = annotate(stage, "/World/envs/env_0/TopCam", (1920,1080))

    vis = HoneyVisual(stage)
    honey = Honey()
    poses = rows[:, 17:24]
    axes = Rotation.from_quat(poses[:, 3:]).as_matrix()[:, :, 0]
    honey.pose(poses[0, :3], axes[0])
    tensor = lambda r: world.torch.tensor(np.asarray(r)[None], dtype=world.torch.float32)
    qv = tensor(np.zeros(8))
    writer = imageio_ffmpeg.write_frames(str(videos/f"{kind}-original.mp4"), (1920,1440),
        fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", run["config"]["renderer"]["mode"])
    settings.set("/rtx/post/aa/op", run["config"]["renderer"]["antialiasing_op"])
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    # Encode 30 Hz frame times and sample the exact terminal state in the
    # final frame if execution ends between two frame boundaries.
    key_frames = {1, n_frames, round(n_frames*.55), round(n_frames*.83), *range(450,n_frames,450)}
    writer.send(None)
    try:
        for i, (row, pose, axis) in enumerate(zip(rows, poses, axes)):
            if i:
                honey.update(DT, pose[:3], axis, row[24:27], row[27:30])
            if (i % step and i != len(rows)-1) or (i == 0 and top is None):
                continue
            for j, (robot, ids) in enumerate(zip(world.robots, world.joint_ids)):
                robot.write_joint_state_to_sim(tensor(row[1+j*8:9+j*8]), qv, joint_ids=ids)
            world.dipper.write_root_pose_to_sim(tensor(pose))
            view.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1/FPS)
            assert np.linalg.norm(world.tool()[0][:3]-pose[:3]) < .0001
            vis.update(honey, pose)
            for _ in range(4 if i <= step else 1):
                station = world.capture()
            if i == 0:
                pixels = image(top)
                assert pixels.shape == (1080,1920,3) and pixels.std() > 10.
                Image.fromarray(pixels).save(screenshot)
                continue
            obs = image(observer)
            assert obs.shape == (1080,1920,3) and obs.std() > 8
            frame = mosaic(obs, station, font, small)
            number = (i+step-1)//step
            if number in key_frames:
                frame.save(frames/f"frame-{number:05d}.jpg", quality=92)
            writer.send(np.asarray(frame))
            if number % 150 == 0:
                print(f"Rendered {number}/{n_frames} frames", flush=True)
    finally:
        writer.close()
        observer.detach()
        product.destroy()
        world.env.close()

    real = json.loads((out/"measurements.json").read_text())
    remade = honey.measurements()
    assert remade == {k: real[k] for k in remade}, "Material replay diverged from measured execution"
    (out/"render.json").write_text(json.dumps(dict(source_gpu_run_id=result["gpu_run_id"],
        frames=n_frames, fps=FPS, resolution=[1920,1440], duration_s=n_frames/FPS,
        observer_eye=EYE, observer_target=LOOK, observer_focal_mm=FOCAL_MM,
        liquid_replay="all landings and final state match exactly"), indent=2)+"\n")


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    out = Path(sys.argv[1]).resolve()
    screenshot = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 else None
    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    try:
        render(out, screenshot)
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

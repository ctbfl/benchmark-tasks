"""Render the observer panel from achieved USD poses, in a separate Kit process."""
import gzip
import json
from pathlib import Path
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from isaaclab.app import AppLauncher


def snapshots(run):
    with gzip.open(run/"measured-prefix.jsonl.gz", "rt") as journal:
        return [observation for line in journal for observation in json.loads(line)["observations"]]


def main():
    run = Path(sys.argv[1])
    probe = sys.argv[2:] == ["probe"]
    assert not sys.argv[2:] or probe
    observations = snapshots(run)
    assert [row["tick"] for row in observations] == list(range(0, 12*len(observations), 12))
    paths = json.loads((run/"render-paths.json").read_text())
    seed_file = "soak-result.json" if probe else "result.json"
    seed_key = "replay_seed" if probe else "trial_seed"
    seed = json.loads((run/seed_file).read_text())[seed_key]
    launcher = AppLauncher(headless=True, device="cpu", enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=8 --/omni/tbb/globalcontrol/maxThreadCount=8")

    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom
    from simulation import Simulation, write_json

    sim = Simulation(run/"observer-render", seed=seed, cameras=False)
    sim.base.sim.pause()
    stage = sim.base.sim.stage
    camera = UsdGeom.Camera.Define(stage, "/ObserverCamera")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.90, -.21, 1.53), Gf.Vec3d(.355, 0., .755), Gf.Vec3d(0, 0, 1)).GetInverse())
    camera.CreateFocalLengthAttr(44)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
    product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([product])
    font = ImageFont.load_default(size=32)
    small = ImageFont.load_default(size=24)
    prims = [stage.GetPrimAtPath(path) for path in paths]
    assert all(p.IsValid() and not p.IsInstanceProxy() for p in prims)
    ops = []
    for prim in prims:
        xform = UsdGeom.Xformable(prim)
        op = xform.AddTransformOp(opSuffix="achievedPose")
        xform.SetXformOpOrder([op], resetXformStack=True)
        ops.append(op)

    def set_poses(observation):
        transforms = []
        with Sdf.ChangeBlock():
            for op, (x, y, z, qx, qy, qz, qw) in zip(ops, observation["poses_xyzw"], strict=True):
                matrix = Gf.Matrix4d(1).SetRotate(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
                matrix.SetTranslateOnly(Gf.Vec3d(x, y, z))
                op.Set(matrix)
                transforms.append(matrix)
        return transforms

    videos = run/"videos"
    native = imageio_ffmpeg.read_frames(str(videos/"native-observations.mp4"), pix_fmt="rgb24")
    metadata = next(native)
    assert metadata["size"] == (1920, 360)
    if not probe:
        writer = imageio_ffmpeg.write_frames(str(videos/"original.mp4"), (1920, 1440), fps=30,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                           "-threads", "4", "-movflags", "+faststart"])
        writer.send(None)
    sampled = {0, len(observations)//2, len(observations)-1}
    set_poses(observations[0])
    for _ in range(4):
        sim.capture(names=())
    presented_ticks = []
    for i, observation in enumerate(observations):
        raw_camera_panel = next(native)
        if probe and i not in sampled:
            continue
        transforms = set_poses(observation)
        sim.capture(names=())
        if i in sampled:
            cache = UsdGeom.XformCache()
            assert all(Gf.IsClose(cache.GetLocalToWorldTransform(prim), matrix, 1e-5)
                       for prim, matrix in zip(prims, transforms, strict=True))
        panel = np.asarray(rgb.get_data()[..., :3])
        assert panel.shape == (1080, 1920, 3) and panel.std() > 5
        frame = Image.new("RGB", (1920, 1440))
        frame.paste(Image.fromarray(panel), (0, 0))
        frame.paste(Image.frombytes("RGB", (1920, 360), raw_camera_panel), (0, 1080))
        draw = ImageDraw.Draw(frame)
        for y, label, style, color in (
            (0, "External Observer View", font, "white"),
            (44, "Not a robot camera", small, "#b7daf3"),
        ):
            draw.rectangle((0, y, round(style.getlength(label))+18, y+style.size+11), fill="black")
            draw.text((9, y+3), label, fill=color, font=style)
        if probe:
            frame.save(run/"observer-render"/f"snapshot-{observation['tick']:05d}.jpg", quality=94)
        else:
            pixels = np.asarray(frame)
            end_frame = ((len(observations)-1)*3+1)//2
            while len(presented_ticks)*2//3 == i and len(presented_ticks) <= end_frame:
                writer.send(pixels)
                presented_ticks.append(observation["tick"])
    assert next(native, None) is None
    native.close()
    if not probe:
        writer.close()
    write_json(run/"observer-render"/"media-map.json", {
        "native_observation_ticks": [row["tick"] for row in observations],
        "presentation_source_ticks": presented_ticks, "probe_ticks": sorted(observations[i]["tick"] for i in sampled),
        "restored_pose_count": len(paths), "simulated_control_steps_during_render": len(sim.actions)})
    assert not sim.actions
    rgb.detach()
    product.destroy()
    sim.close()
    launcher.app.close()


if __name__ == "__main__":
    main()

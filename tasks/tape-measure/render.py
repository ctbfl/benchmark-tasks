"""Calibrated initial screenshot and fixed diagnostic views."""

import json

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom


def camera(stage, path, size, mount=None, eye=None, target=None):
    import omni.replicator.core as rep
    obj = UsdGeom.Camera.Define(stage, path)
    obj.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    obj.CreateClippingRangeAttr(Gf.Vec2f(.005, 20.))
    if mount is not None:
        x, y, z, w = mount.rot_xyzw
        obj.AddTranslateOp().Set(Gf.Vec3d(*mount.pos))
        obj.AddOrientOp().Set(Gf.Quatf(w, x, y, z))
        lens = mount.lens.resized(*size)
        obj.CreateFocalLengthAttr(lens.focal_mm)
        obj.CreateHorizontalApertureAttr(lens.width*lens.focal_mm/lens.fx)
        obj.CreateVerticalApertureAttr(lens.height*lens.focal_mm/lens.fy)
        obj.CreateHorizontalApertureOffsetAttr((lens.width/2-lens.cx)*lens.focal_mm/lens.fx)
        obj.CreateVerticalApertureOffsetAttr((lens.cy-lens.height/2)*lens.focal_mm/lens.fy)
    else:
        obj.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),
            Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        obj.CreateFocalLengthAttr(40.)
        obj.CreateHorizontalApertureAttr(36.)
    product = rep.create.render_product(path, size)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([product])
    return obj, product, rgb


def refresh(sim):
    import carb.settings
    import carb.eventdispatcher
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset", True)
    carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
    sim.instruments.update_visuals()
    sim.base.sim.render()
    ensure_isaac_rtx_render_update(force=True)


def screenshots(sim, out, initial_path):
    from yam_arena import station
    rig = station.rig("calibrated")
    targets = [
        (initial_path, {"mount": rig.top_cam}),
        (out/"overview.png", {"eye": (-.25, -.8, 1.4), "target": (.30, 0., .82)}),
        (out/"assets.png", {"eye": (.06, -.45, 1.16), "target": (.32, .0, .78)}),
    ]
    for index, (path, params) in enumerate(targets):
        obj, product, rgb = camera(sim.stage, f"/Preview_{index}", (1920, 1080), **params)
        for _ in range(20):
            refresh(sim)
        pixels = np.asarray(rgb.get_data())
        assert pixels.shape == (1080, 1920, 4) and pixels[..., :3].std() > 5
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[..., :3]).save(path)
        rgb.detach()
        product.destroy()
        print("Saved", str(path), flush=True)


def restore(sim, row, columns, physics):
    """Restore a recorded physical state for rendering only; never used by a policy."""
    import omni.physx
    import torch
    values = {}
    start = 0
    for name, width in columns.items():
        values[name] = torch.tensor(row[start:start+width][None], dtype=torch.float32, device=sim.base.device)
        start += width
    assert start == len(row)
    for side, robot in zip(("left", "right"), sim.robots):
        robot.write_joint_state_to_sim(values[side+"_q"], values[side+"_qd"])
    for name, asset in (("tape", sim.tape), ("calculator", sim.calculator), ("wood", sim.wood)):
        asset.write_root_pose_to_sim(values[name+"_pose"])
        asset.write_root_velocity_to_sim(values[name+"_velocity"])
        if name != "wood":
            asset.write_joint_state_to_sim(values[name+"_q"], values[name+"_qd"])
    physics.update_articulations_kinematic()
    omni.physx.get_physx_interface().update_transformations(False, True, True)
    # Sleeping rigid bodies may keep an old USD transform during offline replay.
    pose = values["wood_pose"][0].cpu().numpy()
    prim = sim.stage.GetPrimAtPath("/World/envs/env_0/wood/bar")
    world = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose[6]), *map(float, pose[3:6])))
    world.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
    parent = UsdGeom.XformCache().GetParentToWorldTransform(prim)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(world*parent.GetInverse())
    sim.base.scene.update(sim.dt)


def video(source):
    """Render synchronized physical records at their original speed."""
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.timeline
    from isaaclab.sim.utils import get_current_stage_id
    from PIL import ImageDraw, ImageFont
    from simulation import Simulation
    from scene import ROOT
    import hashlib
    manifest = json.loads((source/"source_manifest.json").read_text())
    for name, digest in manifest.items():
        if name.startswith("assets/") or name in ("scene.py", "model.py", "task.py", "dream-defaults.json"):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, ("Recording asset or scene changed", name)
    report = json.loads((source/"result.json").read_text())
    data = np.load(source/"trajectory.npz")
    rows = data["states"]
    hz = report["config"]["simulation"]["physics_hz"]
    assert hz % 30 == 0
    indices = np.arange(0, len(rows), hz//30)
    folder, frames = source/"videos", source/"frames"
    folder.mkdir(exist_ok=True)
    frames.mkdir(exist_ok=True)
    out = source/"render"
    (out/"logs").mkdir(parents=True, exist_ok=True)
    sim = Simulation(out, cameras=True)
    eye, target = (.85, .60, 1.42), (.31, .0, .82)
    obj, product, rgb = camera(sim.stage, "/VideoObserver", (1920, 1080), eye=eye, target=target)
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    views = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"), ("right_cam", "Right Wrist Camera"))
    writer = imageio_ffmpeg.write_frames(str(folder/"original.mp4"), (1920, 1440), fps=30,
        codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
        output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                       "-threads", "4", "-movflags", "+faststart"])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    checkpoints = {0, len(indices)-1, *range(450, len(indices), 450)}
    camera_metadata = {}
    try:
        for frame_index, index in enumerate(indices):
            row = rows[index]
            omni.timeline.get_timeline_interface().set_current_time(float(row[0]))
            restore(sim, row, report["record_columns"], physics)
            text = "0"
            for event in report["events"]:
                if event["time_s"] <= row[0]+1e-7:
                    text = event["display"]
            sim.instruments.calculator.text = text
            for _ in range(8 if frame_index == 0 else 1):
                refresh(sim)
            pictures = {"observer": np.asarray(rgb.get_data())[..., :3].copy()}
            assert pictures["observer"].shape == (1080, 1920, 3)
            for name, _ in views:
                sensor = sim.base.scene[name]
                sensor.update(1/30, force_recompute=True)
                pixels = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                assert pixels.shape == (360, 640, 3)
                pictures[name] = pixels
                if frame_index == 0:
                    camera_metadata[name] = dict(resolution=[640, 360],
                        intrinsics=sensor.data.intrinsic_matrices.torch[0].cpu().tolist())
            frame = Image.new("RGB", (1920, 1440))
            frame.paste(Image.fromarray(pictures["observer"]))
            for column, (name, _) in enumerate(views):
                frame.paste(Image.fromarray(pictures[name]), (640*column, 1080))
            draw = ImageDraw.Draw(frame)
            labels = [(0, 0, "External Observer View", True)]
            labels += [(640*i, 1080, title, False) for i, (_, title) in enumerate(views)]
            for x, y, title, observer in labels:
                width = int(font.getlength(title))+32
                draw.rectangle((x, y, x+width-1, y+(84 if observer else 52)-1), fill="black")
                draw.text((x+16, y+10), title, font=font, fill="white")
                if observer:
                    draw.text((x+16, y+50), "Not a robot camera", font=small, fill=(175, 215, 240))
            if frame_index in checkpoints:
                frame.save(frames/f"frame-{frame_index:05d}.png")
                for name, pixels in pictures.items():
                    Image.fromarray(pixels).save(frames/f"{name}-{frame_index:05d}.png")
            writer.send(np.asarray(frame))
            if frame_index % 150 == 0:
                print("RENDER", frame_index, len(indices), "time", round(float(row[0]), 3), flush=True)
        (source/"render.json").write_text(json.dumps(dict(frames=len(indices), fps=30,
            resolution=[1920, 1440], duration_s=len(indices)/30, playback_speed=1.,
            source="trajectory.npz", source_score=report["score"],
            observer=dict(eye=eye, target=target, focal_length_mm=40, fixed=True), cameras=camera_metadata,
            note="Synchronized measured-state rendering; no resimulation or edited outcome."), indent=2)+"\n")
    finally:
        writer.close()
        rgb.detach()
        product.destroy()
        sim.close()

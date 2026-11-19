"""Re-render measured robot states and outlet strokes at 30 fps."""

import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from inspect_robots import Scene
from PIL import Image, ImageDraw, ImageFont

from evaluate import write_json
from liquid import CAMERAS, DT, INSTRUCTION
from scene import ARMS, GravyStation, scene_hashes

FPS = 30
W, H = 1920, 1080
LABELS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")
EYE = (.535, .455, 1.13)
LOOK_AT = (.360, -.085, .908)


def title(panel, text, observer=False):
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=32)
    width = draw.textlength(text, font=font)
    draw.rectangle((0,0,width+12,44), fill="black")
    draw.text((6,0), text, font=font, fill="white")
    if observer:
        font = ImageFont.load_default(size=24)
        subtitle = "Not a robot camera"
        draw.rectangle((0,45,draw.textlength(subtitle,font=font)+12,78), fill="black")
        draw.text((6,45), subtitle, font=font, fill=(175,215,240))
    return panel


def pixels(sensor):
    sensor.update(1/FPS, force_recompute=True)
    return sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy().copy()


def render(run, out):
    manifest = json.loads((run / "trial.json").read_text())
    assert manifest["scene_sha256"] == scene_hashes(), "The scene differs from the recorded run"
    result = json.loads((run / "result.json").read_text())
    out.mkdir(parents=True)
    (out / "videos").mkdir()
    frames = out / "frames"
    frames.mkdir()
    scene = GravyStation(out, result["seed"], record=False, still_camera=result["kind"] == "demo")
    rgb = product = writer = None
    status = 1
    try:
        scene.reset(Scene(id="replay", instruction=INSTRUCTION), seed=result["seed"])
        import carb.eventdispatcher
        import carb.settings
        import imageio_ffmpeg
        import omni.physx
        import omni.replicator.core as rep
        import torch
        from pxr import Gf, UsdGeom

        stage = scene.stage
        base = scene._base
        sizes = [len(base.scene[arm].joint_names) for arm in ARMS]
        path = run / "physics-f64.bin"
        rows = np.fromfile(path, dtype="<f8").reshape(-1,2 + 2*sum(sizes))
        np.testing.assert_allclose(rows[:, 0], np.arange(len(rows))*DT, atol=1e-10)
        stride = round(1/(FPS*DT))
        assert stride * DT == 1/FPS
        indices = list(range(0,len(rows),stride))
        if indices[-1] != len(rows)-1:
            indices.append(len(rows)-1)

        settings = carb.settings.get_settings()
        settings.set_bool("/rtx/scenePartitioning/showAllPartitionsByDefault", True)
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        camera = UsdGeom.Camera.Define(stage, "/World/Observer")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*EYE), Gf.Vec3d(*LOOK_AT), Gf.Vec3d(0,0,1)).GetInverse())
        camera.CreateFocalLengthAttr(35.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.025, 20.))
        product = rep.create.render_product(str(camera.GetPath()), (W,H))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([product])
        writer = imageio_ffmpeg.write_frames(str(out/"videos"/"original.mp4"), (W,H+360),
            fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
        writer.send(None)
        paddle = scene.initial_paddle.copy()
        previous = 0
        checkpoints = {0, 3*FPS, len(indices)//2, max(0,len(indices)-2*FPS), len(indices)-1}
        for frame_index, index in enumerate(indices):
            for stroke in rows[previous+1:index+1, 1]:
                scene.drain.step(stroke)
            previous = index
            position = 2
            for arm, size in zip(ARMS,sizes,strict=True):
                robot = base.scene[arm]
                robot.write_joint_state_to_sim_index(
                    position=torch.tensor(rows[index,position:position+size][None],dtype=torch.float32),
                    velocity=torch.tensor(rows[index,position+size:position+2*size][None],dtype=torch.float32))
                position += 2*size
            paddle[0,2] = scene.initial_paddle[0,2] - rows[index,1]
            scene.paddle.set_transforms(paddle, np.zeros(1,dtype=np.uint32))
            scene.physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False, True, True)
            base.scene.update(1/FPS)
            scene.visual.update(scene.drain)
            settings.set_bool("/rtx-transient/post/dlss/forceParamReset", True)
            dispatcher.dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
            for _ in range(5 if frame_index == 0 else 1):
                scene.render()
            if frame_index == 0 and scene.still_camera:
                still = pixels(base.scene["still"])
                assert still.shape == (1080,1920,3) and still.std() > 8
                Image.fromarray(still).save(out/"screenshot.png")
            external = np.ascontiguousarray(rgb.get_data()[...,:3])
            assert external.shape == (1080,1920,3)
            if frame_index in checkpoints:
                assert external.std() > 8, "Observer image is blank"
            frame = Image.new("RGB", (W,H+360))
            frame.paste(title(Image.fromarray(external), "External Observer View", observer=True))
            for column, (name, label) in enumerate(zip(CAMERAS,LABELS,strict=True)):
                picture = pixels(base.scene[name])
                assert picture.shape == (360,640,3)
                frame.paste(title(Image.fromarray(picture), label), (column*640,H))
            writer.send(np.asarray(frame))
            if frame_index in checkpoints:
                frame.save(frames/f"{frame_index:05d}.png")
            if frame_index % 90 == 0:
                print(f"Rendered frame {frame_index+1}/{len(indices)}", flush=True)
        replay = scene.drain.metrics()
        for name, value in replay.items():
            np.testing.assert_allclose(value, result["measurements"][name], atol=1e-9)
        write_json(out/"render.json", dict(source=str(run), gpu_run_id=result["gpu_run_id"],
            trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(indices),
            fps=FPS, duration_s=float(rows[-1,0]), score=replay["score"],
            observer=dict(eye=EYE, look_at=LOOK_AT)))
        status = 0
    except BaseException:
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.close()
        if rgb is not None:
            rgb.detach()
            product.destroy()
        scene.close()
        if scene.app is not None:
            scene.app.close(exit_code=status)
    return status


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python render.py RUN_DIR RENDER_DIR"
    sys.exit(render(Path(sys.argv[1]),Path(sys.argv[2])))

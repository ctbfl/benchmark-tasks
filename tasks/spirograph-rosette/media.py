"""Render actual paper contact marks and synchronized views of the recorded trial."""

import json
from pathlib import Path
import sys
import traceback

import numpy as np

from scene import ASSET_ROOT, CAMERAS, DT, EPISODE, make_env
from scoring import BOARD_TOP, INK_WIDTH, RosetteScore

FPS = 30
LABELS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


class Ink:
    def __init__(self, stage):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        root = ASSET_ROOT + "/Ink"
        self.mesh = UsdGeom.Mesh.Define(stage, root)
        self.mesh.CreateSubdivisionSchemeAttr("none")
        self.mesh.CreateDoubleSidedAttr(True)
        self.mesh.CreateExtentAttr([(0., -.3, BOARD_TOP), (.6, .2, BOARD_TOP+.001)])
        fluid = UsdShade.Material.Define(stage, root + "/InkMaterial")
        shader = UsdShade.Shader.Define(stage, root + "/InkMaterial/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.011, .025, .145))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.66)
        fluid.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(fluid)
        self.points, self.indices, self.counts = [], [], []
        self.last = None
        self.i = 0

    def polygon(self, points):
        self.indices.extend(range(len(self.points), len(self.points)+len(points)))
        self.points.extend((float(x), float(y), BOARD_TOP+.00007) for x, y in points)
        self.counts.append(len(points))

    def update(self, stroke):
        from pxr import Vt

        n = len(self.points)
        for _, x, y, connected in stroke[self.i:]:
            p = np.array([x, y])
            if not connected or self.last is None:
                ring = (INK_WIDTH / 2)*np.array([[1, 0], [.7, .7], [0, 1], [-.7, .7], [-1, 0],
                                       [-.7, -.7], [0, -1], [.7, -.7]])
                self.polygon(p + ring)
            else:
                delta = p-self.last
                length = np.linalg.norm(delta)
                if length < .00022:
                    continue
                normal = np.array([-delta[1], delta[0]])*(INK_WIDTH / 2)/length
                self.polygon([self.last-normal, p-normal, p+normal, self.last+normal])
            self.last = p
        self.i = len(stroke)
        if len(self.points) != n:
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(self.points, dtype=np.float32)))
            self.mesh.GetFaceVertexIndicesAttr().Set(self.indices)
            self.mesh.GetFaceVertexCountsAttr().Set(self.counts)


def render_now(base, names, dt):
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    base.sim.render()
    ensure_isaac_rtx_render_update(force=True)
    for name in names:
        base.scene[name].update(dt, force_recompute=True)


def pixels(base, name):
    camera = base.scene[name]
    return camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()


def observer_camera(stage):
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, "/Observer")
    camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    eye, look = (.570, -.115, 1.167), (.335, -.104, .756)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1)).GetInverse())
    camera.CreateFocalLengthAttr(37)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 20))
    product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([product])
    return product, annotator


def render_trial(result_path, output, screenshot):
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.physx
    import torch
    from isaaclab.sim.utils import get_current_stage_id
    from PIL import Image, ImageDraw, ImageFont


    out = result_path.parent
    result = json.loads(result_path.read_text())
    trace_path = out / result["physics_trace"]
    trace = np.fromfile(trace_path, dtype="<f4").reshape(-1, 33)
    stride = round(1/(FPS*DT))
    assert abs(stride*DT - 1/FPS) < 1e-10
    assert trace.shape[0] == 1 + result["control_steps"]*round(1/(20*DT))
    env = make_env(True, out, top_size=(1920, 1080) if screenshot else None)
    env.reset(seed=EPISODE["seed"])
    base = env.unwrapped
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    physics.set_subspace_roots("/")
    ink = Ink(base.sim.stage)
    score = RosetteScore()
    writer = annotator = product = None

    def pose(row):
        for name, data in (("left_robot", row[:8]), ("right_robot", row[8:16])):
            q = torch.tensor(data[None], dtype=torch.float32)
            base.scene[name].write_joint_state_to_sim(q, torch.zeros_like(q))
        for name, data in (("wheel", row[16:23]), ("pen", row[23:30])):
            base.scene[name].write_root_pose_to_sim(torch.tensor(data[None], dtype=torch.float32))
        physics.update_articulations_kinematic()
        omni.physx.get_physx_interface().update_transformations(False, True, True)
        base.scene.update(1/FPS)

    try:
        if screenshot:
            pose(trace[0])
            for _ in range(7):
                render_now(base, ("top_cam",), 1/FPS)
            shot = pixels(base, "top_cam")
            assert shot.shape == (1080, 1920, 3) and shot.std() > 10
            Image.fromarray(shot).save(output)
            return

        product, annotator = observer_camera(base.sim.stage)
        font = ImageFont.load_default(size=32)
        small = ImageFont.load_default(size=24)
        writer = imageio_ffmpeg.write_frames(str(output), (1920, 1440), fps=FPS,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an", "-crf", "18", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"])
        writer.send(None)
        frame_count = 0
        for i, row in enumerate(trace):
            if i:
                score.update(row[16:23], row[23:30], row[30:33])
            if i % stride:
                continue
            pose(row)
            ink.update(score.stroke)
            for _ in range(5 if not i else 1):
                render_now(base, CAMERAS, 1/FPS)
            picture = np.ascontiguousarray(annotator.get_data()[..., :3])
            assert picture.shape == (1080, 1920, 3) and picture.std() > 8
            image = Image.new("RGB", (1920, 1440))
            image.paste(Image.fromarray(picture))
            for j, name in enumerate(CAMERAS):
                view = pixels(base, name)
                assert view.shape == (360, 640, 3) and view.std() > 8
                image.paste(Image.fromarray(view), (j*640, 1080))
            draw = ImageDraw.Draw(image)
            labels = [(0, 0, "External Observer View")] + [(j*640, 1080, label) for j, label in enumerate(LABELS)]
            for x, y, title in labels:
                width = max(font.getlength(title), small.getlength("Not a robot camera") if not y else 0) + 24
                draw.rectangle((x, y, x+width, y+(76 if not y else 44)), fill="black")
                draw.text((x+12, y+5), title, font=font, fill="white")
            draw.text((12, 44), "Not a robot camera", font=small, fill=(175, 215, 240))
            writer.send(np.asarray(image))
            frame_count += 1
            if frame_count == 1 or frame_count % 300 == 0:
                print("Rendered", frame_count, "frames", flush=True)
        assert score.result() == result["measurements"]
        output.with_suffix(".json").write_text(json.dumps(dict(fps=FPS, frames=frame_count,
            resolution=[1920, 1440], physics_trace=str(trace_path),
            gpu_run_id=result["gpu_run_id"], score=result["score"]), indent=2) + "\n")
    finally:
        if writer is not None:
            writer.close()
        if annotator is not None:
            annotator.detach()
            product.destroy()
        env.close()


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python media.py RESULT_JSON OUTPUT.mp4|screenshot.png"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True, kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app
    try:
        render_trial(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[2].endswith(".png"))
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

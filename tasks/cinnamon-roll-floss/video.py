"""Synchronous 30 Hz recording and an initial native-resolution top view."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
         ("right_cam", "Right Wrist Camera"))


class Recorder:
    def __init__(self, sim):
        import imageio_ffmpeg
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        self.sim, self.rep = sim, rep
        self.base = sim._base
        self.tick = None
        camera = UsdGeom.Camera.Define(self.base.sim.stage, "/ObserverCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(.78, -.48, 1.14), Gf.Vec3d(.270, .0, .815), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(52.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.008, 15.))
        self.product, self.rgb = self.make_product(str(camera.GetPath()))
        self.frames = []
        self.font, self.note_font = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
        (sim.out/"videos").mkdir()
        self.writer = imageio_ffmpeg.write_frames(
            str(sim.out/"videos"/"original.mp4"), (1920, 1440), fps=30, codec="libx264",
            pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                           "-threads", "4", "-movflags", "+faststart"])
        self.writer.send(None)

    def make_product(self, camera):
        product = self.rep.create.render_product(camera, (1920, 1080))
        annotator = self.rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach([product])
        return product, annotator

    def refresh(self, force=False):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        if self.tick == self.sim.tick and not force:
            return
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name, _ in VIEWS:
            self.base.scene[name].update(self.base.physics_dt, force_recompute=True)
        self.tick = self.sim.tick

    def screenshot(self):
        product, rgb = self.make_product("/World/envs/env_0/TopCam")
        for _ in range(6):
            self.refresh(force=True)
        pixels = rgb.get_data()[..., :3]
        assert pixels.shape == (1080, 1920, 3) and pixels.std() > 5
        Image.fromarray(pixels).save(self.sim.out/"screenshot.png")
        rgb.detach()
        product.destroy()

    def capture(self, tick):
        self.refresh()
        frame = Image.new("RGB", (1920, 1440))
        top = self.rgb.get_data()[..., :3]
        assert top.shape == (1080, 1920, 3)
        frame.paste(Image.fromarray(top), (0, 0))
        draw = ImageDraw.Draw(frame)

        def label(x, y, title, note=None):
            width = int(self.font.getlength(title))+22
            draw.rectangle((x, y, x+width, y+(70 if note else 40)), fill="black")
            draw.text((x+10, y+1), title, font=self.font, fill="white")
            if note:
                draw.text((x+10, y+41), note, font=self.note_font, fill=(174, 214, 237))

        label(0, 0, "External Observer View", "Not a robot camera")
        for i, (name, title) in enumerate(VIEWS):
            pixels = self.base.scene[name].data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            assert pixels.shape == (360, 640, 3)
            frame.paste(Image.fromarray(pixels), (640*i, 1080))
            label(640*i, 1080, title)
        self.writer.send(np.asarray(frame))
        self.frames.append(tick)

    def close(self):
        self.writer.close()
        self.rgb.detach()
        self.product.destroy()
        (self.sim.out/"frame-ticks.json").write_text(json.dumps(self.frames)+"\n")

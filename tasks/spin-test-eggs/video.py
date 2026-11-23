"""Record the fixed external view and the three native robot views at 30 Hz."""
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TITLE_FONT = 32
SUBTITLE_FONT = 24


class Video:
    def __init__(self, sim, out):
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        self.sim = sim
        camera = UsdGeom.Camera.Define(sim.base.sim.stage, "/ObserverCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(.90, -.21, 1.53), Gf.Vec3d(.355, 0., .755), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(44)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
        self.product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        destination = Path(out) / "videos"
        destination.mkdir()
        self.path = destination / "original.mp4"
        self.writer = imageio_ffmpeg.write_frames(str(self.path), (1920, 1440), fps=30,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast", "-threads", "4",
                           "-movflags", "+faststart"])
        self.writer.send(None)
        self.ticks = []
        self.font = ImageFont.load_default(size=TITLE_FONT)
        self.small = ImageFont.load_default(size=SUBTITLE_FONT)
        self.images = {}
        for _ in range(4):
            self.images = self.sim.capture()

    def capture(self):
        self.images = self.sim.capture()
        ext = np.asarray(self.rgb.get_data()[..., :3])
        assert ext.shape == (1080, 1920, 3) and ext.std() > 5
        frame = Image.new("RGB", (1920, 1440))
        frame.paste(Image.fromarray(ext))
        cameras = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
                   ("right_cam", "Right Wrist Camera"))
        for i, (name, _) in enumerate(cameras):
            frame.paste(Image.fromarray(self.images[name]), (i*640, 1080))
        draw = ImageDraw.Draw(frame)
        for x, y, title in [(0, 0, "External Observer View"),
                            *((640*i, 1080, title) for i, (_, title) in enumerate(cameras))]:
            width = math.ceil(self.font.getlength(title))+18
            draw.rectangle((x, y, x+width-1, y+43), fill="black")
            draw.text((x+9, y+3), title, font=self.font, fill="white")
        caption = "Not a robot camera"
        draw.rectangle((0, 44, math.ceil(self.small.getlength(caption))+18, 78), fill="black")
        draw.text((9, 49), caption, font=self.small, fill=(183, 218, 243))
        self.writer.send(np.asarray(frame))
        self.ticks.append(self.sim.ticks)

    def close(self):
        self.writer.close()
        self.rgb.detach()
        self.product.destroy()

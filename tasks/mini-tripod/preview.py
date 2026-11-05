"""Record actual simulator frames; no physics or object motion is authored here."""

import imageio_ffmpeg
import numpy as np
import omni.replicator.core as rep
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, UsdGeom, UsdLux

SIZE = (960, 720)
FPS = 15


class Preview:
    def __init__(self, stage, out):
        self.out = out
        (out / "videos").mkdir(exist_ok=True)
        dome = UsdLux.DomeLight.Define(stage, "/Lights/Fill")
        dome.CreateIntensityAttr(350)
        key = UsdLux.DistantLight.Define(stage, "/Lights/Key")
        key.CreateIntensityAttr(2200)
        key.CreateAngleAttr(15)
        key.AddRotateXYZOp().Set(Gf.Vec3f(25,-30,-25))
        panel = UsdLux.RectLight.Define(stage, "/Lights/Panel")
        panel.CreateWidthAttr(.5)
        panel.CreateHeightAttr(.7)
        panel.CreateIntensityAttr(800)
        panel.AddTranslateOp().Set(Gf.Vec3d(-.2,-.3,.6))
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        self.pose = camera.AddTransformOp()
        self.look_at((0,-.70,.38),(0,0,.145))
        camera.CreateFocalLengthAttr(43)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005,10))
        self.product = rep.create.render_product("/Camera", SIZE)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        self.font = ImageFont.load_default(size=20)
        self.writer = imageio_ffmpeg.write_frames(str(out / "videos/mini_tripod.mp4"), SIZE,
            fps=FPS, codec="libx264", quality=8, pix_fmt_out="yuv420p",
            output_params=["-movflags","+faststart"])
        self.writer.send(None)
        self.first = None
        self.max_change = 0.

    def look_at(self, eye, target):
        self.pose.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse())

    def capture(self, phase, t, angles, filename=None, video=True):
        ensure_isaac_rtx_render_update(force=True)
        pixels = self.rgb.get_data()
        assert pixels.shape[:2] == SIZE[::-1], "RTX camera has no frame"
        sample = pixels[90:-55:8,::8,:3].astype(np.int16)
        if self.first is None:
            self.first = sample.copy()
        self.max_change = max(self.max_change, float(np.abs(sample-self.first).mean()))
        frame = Image.fromarray(np.ascontiguousarray(pixels[:,:,:3]))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,76), fill=(22,27,34))
        draw.text((20,10), "MINI TRIPOD | three hinged fixed-length legs", font=self.font, fill="white")
        draw.text((20,42), phase, font=self.font, fill=(131,210,238))
        draw.rectangle((0,675,960,720), fill=(22,27,34))
        draw.text((20,688), f"t={t:4.1f}s   legs={angles[0]:4.1f}/{angles[1]:4.1f}/{angles[2]:4.1f} deg"
                  "   fixed short legs", font=self.font, fill="white")
        if filename:
            frame.save(self.out / filename)
        if video:
            self.writer.send(np.asarray(frame))

    def close(self):
        self.writer.close()
        self.rgb.detach()
        self.product.destroy()

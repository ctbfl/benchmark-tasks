"""Fixed observer and three calibrated robot cameras, captured at one physics time."""

import json
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
         ("right_cam", "Right Wrist Camera"))


class Recorder:
    def __init__(self, env, out, record, filename="expert.mp4"):
        import imageio_ffmpeg
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        self.base, self.out, self.record = env.unwrapped, out, record
        self.frames, self.writer = [], None
        self.font, self.small_font = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
        renderer = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
            ambient_light_intensity=0.0,
            carb_settings={"/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 2}))
        apply_isaac_rtx_global_settings(renderer.global_settings)
        camera = UsdGeom.Camera.Define(self.base.sim.stage, "/ExpertCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        self.observer_transform = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(-.18, -.12, 1.32), Gf.Vec3d(.30, .015, .80), Gf.Vec3d(0, 0, 1)).GetInverse()
        self.transform_op = camera.AddTransformOp()
        self.transform_op.Set(self.observer_transform)
        camera.CreateFocalLengthAttr(38)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 30))
        self.product = rep.create.render_product(str(camera.GetPath()), (1920, 1080))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        self.cameras = {}
        if record:
            for name, _ in VIEWS:
                sensor = self.base.scene.sensors[name]
                cfg = sensor.cfg
                self.cameras[name] = {
                    "prim_path": cfg.prim_path, "resolution": [cfg.width, cfg.height],
                    "offset_pos": list(cfg.offset.pos), "offset_rot_xyzw": list(cfg.offset.rot),
                    "offset_convention": cfg.offset.convention,
                    "focal_length_mm": cfg.spawn.focal_length,
                    "aperture_mm": [cfg.spawn.horizontal_aperture, cfg.spawn.vertical_aperture],
                    "aperture_offset_mm": [cfg.spawn.horizontal_aperture_offset, cfg.spawn.vertical_aperture_offset],
                }
            (out / "videos").mkdir(parents=True, exist_ok=True)
            self.writer = imageio_ffmpeg.write_frames(str(out / "videos" / filename), (1920, 1440),
                fps=30, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
                output_params=["-an", "-profile:v", "high", "-crf", "18", "-preset", "fast",
                               "-threads", "4", "-movflags", "+faststart"])
            self.writer.send(None)
        for _ in range(8):
            self.read_views()

    def read_views(self):
        import carb
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        assert carb.settings.get_settings().get("/rtx/rendermode") == "RealTimePathTracing"
        assert self.transform_op.Get() == self.observer_transform
        tick = self.base.sim._physics_step_count
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        views = {"observer": np.ascontiguousarray(self.rgb.get_data()[..., :3]).copy()}
        assert views["observer"].shape == (1080, 1920, 3)
        if self.record:
            for name, _ in VIEWS:
                sensor = self.base.scene.sensors[name]
                sensor.update(1 / 30, force_recompute=True)
                pixels = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
                assert pixels.shape == (360, 640, 3), (name, pixels.shape)
                views[name] = pixels
        assert self.base.sim._physics_step_count == tick, "Rendering advanced physics"
        return views

    def capture(self, expert, filename=None):
        views = self.read_views()
        frame = Image.new("RGB", (1920, 1440 if self.record else 1080))
        frame.paste(Image.fromarray(views["observer"]))
        labels = [(0, 0, "External Observer View", "Not a robot camera")]
        if self.record:
            for i, (name, title) in enumerate(VIEWS):
                frame.paste(Image.fromarray(views[name]), (640 * i, 1080))
                labels.append((640 * i, 1080, title, None))
        draw = ImageDraw.Draw(frame)
        for x, y, title, subtitle in labels:
            width = int(np.ceil(self.font.getlength(title))) + 32
            draw.rectangle((x, y, x + width - 1, y + (84 if subtitle else 52) - 1), fill="black")
            draw.text((x + 16, y + 10), title, font=self.font, fill="white")
            if subtitle:
                draw.text((x + 16, y + 50), subtitle, font=self.small_font, fill=(175, 215, 240))
        caption = f"{expert.phase} | {expert.step_id / 30:.1f} s | Panel: {expert.panel_angle():.1f} deg"
        draw.rectangle((0, 1036, self.small_font.getlength(caption) + 32, 1079), fill="black")
        draw.text((16, 1045), caption, font=self.small_font, fill="white")
        if filename:
            (self.out / "screenshots").mkdir(exist_ok=True)
            frame.save(self.out / "screenshots" / filename)
        else:
            self.writer.send(np.asarray(frame))
            self.frames.append({"index": len(self.frames), "time_s": expert.step_id / 30,
                "physics_step": self.base.sim._physics_step_count,
                "crc32": {name: zlib.crc32(pixels) for name, pixels in views.items()}})

    def close(self):
        if self.writer:
            self.writer.close()
        metadata = {"fps": 30, "frames": len(self.frames), "duration_s": len(self.frames) / 30,
            "resolution": [1920, 1440], "renderer": "RealTimePathTracing",
            "observer_fixed_transform": np.asarray(self.observer_transform).tolist(),
            "observer_focal_length_mm": 38, "cameras": self.cameras,
            "label_title_px": 32, "label_subtitle_px": 24,
            "encoding": {"crf": 18, "preset": "fast", "pixel_format": "yuv420p", "faststart": True}}
        (self.out / "recording.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (self.out / "frame_times.json").write_text(json.dumps(self.frames) + "\n")
        self.rgb.detach()
        self.product.destroy()

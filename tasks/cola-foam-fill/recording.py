"""Synchronized live evaluation media. The observer is never returned to a policy."""
import csv
from pathlib import Path
from zlib import crc32

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
VIEWS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


class Recording:
    def __init__(self, env, still):
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        self.env = env
        camera = UsdGeom.Camera.Define(env.stage, "/Observer")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        eye, target = (.018,.145,1.095), (.363,-.010,.888)
        self.pose = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,0,1)).GetInverse()
        self.transform = camera.AddTransformOp()
        self.transform.Set(self.pose)
        camera.CreateFocalLengthAttr(35.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20))
        self.observer = rep.create.render_product(str(camera.GetPath()), (1920,1080))
        self.observer_rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.observer_rgb.attach([self.observer])
        self.font = ImageFont.load_default(size=32)
        self.small = ImageFont.load_default(size=24)
        self.video = imageio_ffmpeg.write_frames(str(env.out/"video.mp4"), (1920,1440),
            fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an","-profile:v","high","-crf","18","-preset","fast",
                           "-threads","4","-movflags","+faststart"])
        self.video.send(None)
        self.timestamps = (env.out/"frame-times.csv").open("x", newline="")
        self.frame_log = csv.writer(self.timestamps)
        self.frame_log.writerow(["physics_step", "time_s", "observer_crc32", "left_crc32",
                                 "top_crc32", "right_crc32"])
        if still:
            top = rep.create.render_product("/World/envs/env_0/TopCam", (1920,1080))
            rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            rgb.attach([top])
        for _ in range(5):
            env.render_cameras()
        if still:
            image = rgb.get_data()[:,:,:3]
            assert image.shape == (1080,1920,3)
            Image.fromarray(image).save(env.out/"screenshot.png")
            rgb.detach([top])
            top.destroy()

    def frame(self):
        from simulation import CAMERAS

        env = self.env
        assert self.transform.Get() == self.pose
        observer = np.ascontiguousarray(self.observer_rgb.get_data()[:,:,:3])
        assert observer.shape == (1080,1920,3)
        frame = Image.new("RGB", (1920,1440))
        frame.paste(Image.fromarray(observer))
        hashes = [crc32(observer)]
        for i,name in enumerate(CAMERAS):
            raw = env._base.scene[name].data.output["rgb"].torch[0,...,:3].cpu().numpy()
            raw = np.ascontiguousarray(raw)
            assert raw.shape == (360,640,3)
            frame.paste(Image.fromarray(raw), (640*i,1080))
            hashes.append(crc32(raw))
        draw = ImageDraw.Draw(frame)
        labels = [(0,0,"External Observer View"), *[(i*640,1080,name) for i,name in enumerate(VIEWS)]]
        for x,y,label in labels:
            draw.rectangle((x,y,x+int(self.font.getlength(label))+32,y+(85 if not y else 51)),fill="black")
            draw.text((x+16,y+9),label,font=self.font,fill="white")
        draw.text((16,50),"Not a robot camera",font=self.small,fill=(175,215,240))
        self.video.send(np.asarray(frame))
        self.frame_log.writerow([env.ticks, env.pour.time, *hashes])

    def close(self):
        self.video.close()
        self.timestamps.close()
        self.observer_rgb.detach([self.observer])
        self.observer.destroy()


def save_native(out: Path, step: int, observation):
    folder = out/"native"
    folder.mkdir(exist_ok=True)
    for name, rgb in observation.images.items():
        Image.fromarray(rgb).save(folder/f"{step:04d}-{name}.png", compress_level=1)

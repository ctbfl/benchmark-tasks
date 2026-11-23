"""Record exactly the native RGB observations and achieved poses of a Dream trial."""
import gzip
import json
import os

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("left_cam", "top_cam", "right_cam")
LABELS = ("Left Wrist Camera", "Top Camera", "Right Wrist Camera")


class DreamRecording:
    def __init__(self, sim, out):
        self.sim = sim
        self.out = out
        self.ticks = []
        self.pending = []
        self.saved_controls = self.saved_rows = 0
        self.font = ImageFont.load_default(size=32)
        paths = []
        for robot in sim.robots:
            view_paths = dict(zip(robot.backend_body_names, robot.root_view.link_paths[0], strict=True))
            paths.extend(str(view_paths[name]) for name in robot.body_names)
        paths.extend(str(egg.root_view.prim_paths[0]) for egg in sim.eggs)
        self.pose_count = len(paths)
        (out/"render-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
        videos = out/"videos"
        videos.mkdir()
        self.writer = imageio_ffmpeg.write_frames(str(videos/"native-observations.mp4"), (1920, 360),
            fps=20, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-an", "-crf", "18", "-preset", "fast", "-threads", "4", "-g", "20",
                           "-movflags", "+frag_keyframe+empty_moov+default_base_moof"])
        self.writer.send(None)

    def observe(self, tracker):
        tick = self.sim.ticks
        assert tick == len(self.sim.actions) * 12
        assert not self.ticks or tick == self.ticks[-1] + 12
        images = self.sim.capture()
        state = self.sim.joints().astype(np.float64)
        frame = Image.new("RGB", (1920, 360))
        draw = ImageDraw.Draw(frame)
        for i, (name, label) in enumerate(zip(CAMERAS, LABELS)):
            rgb = images[name]
            assert rgb.shape == (360, 640, 3)
            x = 640*i
            frame.paste(Image.fromarray(rgb), (x, 0))
            draw.rectangle((x, 0, x+round(self.font.getlength(label))+17, 43), fill="black")
            draw.text((x+9, 3), label, fill="white", font=self.font)
        self.writer.send(np.asarray(frame))
        self.ticks.append(tick)
        poses = [pose for robot in self.sim.robots
                 for pose in robot.data.body_link_pose_w.torch[0].cpu().tolist()]
        poses.extend(item["pose"] for item in self.sim.rows[-1]["eggs"])
        assert len(poses) == self.pose_count
        self.pending.append({"tick": tick, "poses_xyzw": poses})
        if len(self.sim.actions) % 20 == 0:
            self.flush(tracker)
        return images, state

    def flush(self, tracker):
        if not self.pending:
            return
        segment = {"observations": self.pending, "physics": self.sim.rows[self.saved_rows:],
                   "actions": self.sim.actions[self.saved_controls:], "score": tracker.result()}
        path = self.out/"measured-prefix.jsonl.gz"
        # Each commit is a complete gzip member. Earlier members survive a truncated append.
        with path.open("ab") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3) as stream:
                stream.write(json.dumps(segment, separators=(",", ":"), allow_nan=False).encode() + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        self.saved_rows = len(self.sim.rows)
        self.saved_controls = len(self.sim.actions)
        self.pending.clear()

    def close(self):
        self.writer.close()

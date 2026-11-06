"""Compress the complete four-camera video and validate its media properties."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration:format=size,duration",
        "-of", "json", str(path)], text=True))


def pack(out):
    out = Path(out)
    source, target = out / "videos/original.mp4", out / "videos/video.mp4"
    duration = float(probe(source)["format"]["duration"])
    bitrate = math.floor(9_300_000 * 8 / duration / 1000)
    common = ["ffmpeg", "-y", "-i", str(source), "-an", "-c:v", "libx264", "-b:v", f"{bitrate}k",
              "-preset", "fast", "-threads", "4", "-pix_fmt", "yuv420p", "-passlogfile", str(out / "videos/encode")]
    with (out / "videos/encoding.log").open("w") as log:
        subprocess.run([*common, "-pass", "1", "-f", "null", "/dev/null"], check=True, stdout=log, stderr=log)
        subprocess.run([*common, "-pass", "2", "-profile:v", "high", "-movflags", "+faststart", str(target)],
                       check=True, stdout=log, stderr=log)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"],
                       check=True, stdout=log, stderr=log)
    metadata = probe(target)
    stream = metadata["streams"][0]
    render = json.loads((out / "render.json").read_text())
    assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"], stream["r_frame_rate"]) == (1920, 1440, "30/1")
    assert int(stream["nb_frames"]) == render["frames"]
    assert abs(float(stream["duration"]) - render["duration_s"]) < .001
    assert target.stat().st_size <= 10_000_000
    report = {"passed": True, "bytes": target.stat().st_size,
              "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "probe": metadata}
    (out / "videos/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python pack_video.py RUN_DIRECTORY"
    pack(sys.argv[1])

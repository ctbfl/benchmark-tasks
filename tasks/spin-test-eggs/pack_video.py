"""Compress an archived full-length 30 fps evaluation video to the delivery size."""
from pathlib import Path
import subprocess
import sys


def pack(source, target):
    source, target = Path(source), Path(target)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=nw=1:nk=1", str(source)]))
    bitrate = int(9_150_000*8/duration/1000)
    common = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-an", "-c:v", "libx264", "-b:v",
              f"{bitrate}k", "-preset", "fast", "-threads", "4", "-pix_fmt", "yuv420p",
              "-passlogfile", str(source.parent/"encode")]
    subprocess.run([*common, "-pass", "1", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run([*common, "-pass", "2", "-profile:v", "high", "-movflags", "+faststart", str(target)], check=True)
    assert target.stat().st_size <= 10_000_000
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True)


if __name__ == "__main__":
    pack(*sys.argv[1:])

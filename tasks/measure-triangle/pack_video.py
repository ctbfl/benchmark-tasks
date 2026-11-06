"""Compress and verify a complete 30 FPS four-camera recording."""

import json
from pathlib import Path
import subprocess
import sys


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
        "-show_format", "-of", "json", str(path)]))


def main(folder):
    source, output = folder/"original.mp4", folder/"video.mp4"
    original = probe(source)
    duration = float(original["format"]["duration"])
    bitrate = int(9_300_000*8/duration/1000)
    common = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-an", "-c:v", "libx264",
        "-b:v", f"{bitrate}k", "-preset", "fast", "-threads", "4", "-pix_fmt", "yuv420p",
        "-passlogfile", str(folder/"encode")]
    subprocess.run([*common, "-pass", "1", "-f", "null", "/dev/null"], check=True)
    subprocess.run([*common, "-pass", "2", "-profile:v", "high", "-movflags", "+faststart", str(output)], check=True)
    compressed = probe(output)
    assert output.stat().st_size <= 10_000_000
    for metadata, path in ((original, source), (compressed, output)):
        assert len(metadata["streams"]) == 1
        stream = metadata["streams"][0]
        assert (stream["codec_name"], stream["pix_fmt"], stream["width"], stream["height"], stream["r_frame_rate"]) == (
            "h264", "yuv420p", 1920, 1440, "30/1")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True)
    assert original["streams"][0]["nb_frames"] == compressed["streams"][0]["nb_frames"]
    assert abs(duration-float(compressed["format"]["duration"])) < .001
    (folder/"validation.json").write_text(json.dumps(dict(original=original,
        compressed=compressed, bytes=output.stat().st_size, full_decode_passed=True), indent=2)+"\n")
    print(output.resolve(), output.stat().st_size, "bytes")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python pack_video.py RUN_DIRECTORY/videos"
    main(Path(sys.argv[1]).resolve())

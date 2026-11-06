"""Create and verify a complete, under-10-MB delivery video beside the original."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def probe(path):
    return json.loads(subprocess.check_output(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(path)]))


def encode(original):
    source = probe(original)
    duration = float(source["format"]["duration"])
    bitrate = int(9_300_000 * 8 / duration / 1000)
    output = original.parent/"delivery.mp4"
    common = ["ffmpeg","-y","-v","error","-i",str(original),"-map","0:v:0","-an",
              "-c:v","libx264","-b:v",f"{bitrate}k","-preset","fast","-threads","4",
              "-pix_fmt","yuv420p","-passlogfile",str(original.parent/"encode-pass")]
    subprocess.run([*common,"-pass","1","-f","null","/dev/null"],check=True)
    subprocess.run([*common,"-pass","2","-profile:v","high","-movflags","+faststart",str(output)],check=True)
    actual = probe(output)
    assert len(actual["streams"])==1
    stream = actual["streams"][0]
    for k,v in dict(codec_name="h264",width=1920,height=1440,pix_fmt="yuv420p",r_frame_rate="30/1").items():
        assert stream[k]==v,(k,stream[k])
    assert stream["nb_frames"]==source["streams"][0]["nb_frames"]
    assert abs(float(actual["format"]["duration"])-duration)<.001
    assert output.stat().st_size<=10_000_000
    subprocess.run(["ffmpeg","-v","error","-i",str(output),"-f","null","-"],check=True)
    data = output.read_bytes()
    assert data.index(b"moov")<data.index(b"mdat"), "Missing faststart"
    summary = dict(file=str(output),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
                   duration_s=duration,frames=int(stream["nb_frames"]),width=1920,height=1440,
                   fps=30,codec="h264",pixel_format="yuv420p",audio=False,full_decode=True,faststart=True)
    (original.parent/"validation.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    assert len(sys.argv)==2,"Usage: python encode_video.py ORIGINAL_MP4"
    encode(Path(sys.argv[1]))

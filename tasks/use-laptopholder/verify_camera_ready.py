"""Check the rendered delivery, decode every frame, and extract review stills."""

import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent/"outputs/camera_ready"


def run():
    recording=json.loads((OUT/"recording.json").read_text())
    result=json.loads((OUT/"result.json").read_text())
    frames=json.loads((OUT/"frame_times.json").read_text())
    assert result["error"] is None and result["final"]["score"]["success"]
    assert not result["final"]["score"]["truncated"]
    assert len(frames)==recording["frames"]
    assert np.all(np.diff([f["physics_tick"] for f in frames])==32)
    assert np.allclose(np.diff([f["time_s"] for f in frames]),1/30,atol=1e-10)
    assert abs(recording["video_duration_s"]-recording["simulation_duration_s"])<1/30+1e-9
    assert recording["labels"]=={"title_px":32,"subtitle_px":24,"outer_margin":0}
    for name,camera in recording["cameras"].items():
        assert camera["resolution"]==([1920,1080] if name=="observer" else [640,360])
        assert len({f["crc32"][name] for f in frames})>30,name
        assert camera["renderer"]==recording["renderer"]
        if name!="observer":
            w,h=camera["resolution"]
            focal=camera["focal_length_mm"]
            ax,ay=camera["aperture_mm"]
            ox,oy=camera["aperture_offset_mm"]
            actual=[w*focal/ax,h*focal/ay,w/2-ox*w/ax,h/2+oy*h/ay]
            assert np.allclose(actual,camera["intrinsics"],atol=.001,rtol=0),(name,actual)
    for folder in (OUT/"raw_frames").iterdir():
        for name,camera in recording["cameras"].items():
            with Image.open(folder/(name+".png")) as picture:
                assert list(picture.size)==camera["resolution"]
    review=OUT/"review"
    review.mkdir(exist_ok=True)
    videos={}
    for name,resolution in (("composite",[1920,1440]),):
        path=OUT/"videos/four-camera.mp4"
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format",
                                                  "-of","json",str(path)],text=True))
        assert len(probe["streams"])==1
        stream=probe["streams"][0]
        assert [stream["width"],stream["height"]]==resolution
        assert stream["codec_name"]=="h264" and stream["pix_fmt"]=="yuv420p" and stream["profile"]=="High"
        assert stream["avg_frame_rate"]=="30/1" and int(stream["nb_frames"])==len(frames)
        assert abs(float(probe["format"]["duration"])-len(frames)/30)<.002
        atoms=[]
        with path.open("rb") as f:
            while f.tell()<path.stat().st_size:
                size,kind=struct.unpack(">I4s",f.read(8))
                assert size>=8
                atoms.append(kind.decode("ascii"))
                f.seek(size-8,1)
        assert atoms.index("moov")<atoms.index("mdat"),"MP4 must use faststart"
        subprocess.run(["ffmpeg","-v","error","-xerror","-threads","4","-i",str(path),
                        "-an","-f","null","-"],check=True)
        for mark,seconds in (("initial",0),("manual_seating",85),("final",len(frames)/30-1)):
            subprocess.run(["ffmpeg","-v","error","-y","-threads","4","-ss",str(seconds),
                "-i",str(path),"-frames:v","1","-update","1",str(review/f"{name}-{mark}.png")],check=True)
        videos[name]={"path":str(path),"bytes":path.stat().st_size,"MB":path.stat().st_size/1e6,
            "resolution":resolution,"fps":30,"frames":len(frames),"duration_s":len(frames)/30,
            "profile":stream["profile"],"level":stream["level"],"faststart":True,"full_decode_passed":True}
    report={"passed":True,"videos":videos,"synchronized_frame_clock":True,
        "calibrated_intrinsics_preserved":True,"native_checkpoint_sizes_verified":True,
        "final_score":result["final"]["score"],"recording_wall_s":recording["wall_recording_s"]}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    run()

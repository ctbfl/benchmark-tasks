"""Create and validate a full-duration H.264 delivery video below 10 MB."""
import json
from pathlib import Path
import subprocess
import sys


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],
                          check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def compress(out):
    original=out/'videos/original.mp4'
    destination=out/'videos/delivery.mp4'
    source=probe(original)
    duration=float(source['format']['duration'])
    rate=int(9_300_000*8/duration/1000)
    common=['ffmpeg','-y','-v','error','-i',str(original),'-an','-c:v','libx264','-b:v',f'{rate}k',
            '-preset','fast','-threads','4','-pix_fmt','yuv420p','-passlogfile',str(out/'videos/encode')]
    subprocess.run([*common,'-pass','1','-f','null','/dev/null'],check=True)
    subprocess.run([*common,'-pass','2','-profile:v','high','-movflags','+faststart',str(destination)],check=True)
    final=probe(destination)
    assert len(final['streams'])==1
    video=final['streams'][0]
    assert video['codec_name']=='h264' and video['pix_fmt']=='yuv420p'
    assert (video['width'],video['height'],video['r_frame_rate'])==(1920,1440,'30/1')
    assert int(video['nb_frames'])==int(source['streams'][0]['nb_frames'])
    assert abs(float(final['format']['duration'])-duration)<1e-6
    assert destination.stat().st_size<=10_000_000
    subprocess.run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],check=True)
    validation=dict(bytes=destination.stat().st_size,duration_s=duration,frames=int(video['nb_frames']),
        resolution=[1920,1440],fps=30,codec='h264',pixel_format='yuv420p',audio=False,decoded=True,
        original_bytes=original.stat().st_size)
    (out/'video-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    assert len(sys.argv)==2,'Usage: python compress.py RUN_DIRECTORY'
    compress(Path(sys.argv[1]))

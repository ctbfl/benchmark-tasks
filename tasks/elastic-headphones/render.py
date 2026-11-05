"""Render the complete measured trial with the station's three calibrated cameras."""
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
FPS=30
VIEWS=(("left_cam","Left Wrist Camera"),("top_cam","Top Camera"),("right_cam","Right Wrist Camera"))


def run(out, dream):
    import carb
    import carb.eventdispatcher
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.physx
    import omni.replicator.core as rep
    import omni.timeline
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from pxr import Gf,Sdf,UsdGeom
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from yam_scene import make_env

    OUT=Path(out)
    metrics=json.loads((OUT/"metrics.json").read_text())
    prefix="dream" if dream else "demo"
    if dream:
        assert metrics["exit_code"]==0,metrics["error"]
        for name in ("yam_scene.py","assets/headphones.usda","assets/stand.usda"):
            assert (HERE/name).read_bytes()==(OUT/"source"/name).read_bytes(),name
        schema=json.loads((OUT/"replay-schema.json").read_text())
        assert schema["columns"]==192 and schema["physics_dt"]==1/240
        states=np.fromfile(OUT/"physics-f64.bin",dtype="<f8").reshape(-1,192)
        assert np.allclose(np.diff(states[:,0]),1/240)
        rows=[dict(t=float(r[0]),robot_joint_positions={"left":r[1:9],"right":r[9:17]},
            root=r[33:40],joint_degrees=np.rad2deg(r[46:60]),stand=r[74:81],links=r[87:].reshape(15,7))
            for r in states[8::8]]
    else:
        assert metrics["completed"],metrics["error"]
        rows=json.loads((OUT/"trajectory.json").read_text())
    assert np.allclose(np.diff([r["t"] for r in rows]),1/FPS)
    for directory in ("videos","frames"):
        (OUT/directory).mkdir(exist_ok=True)
    settings=carb.settings.get_settings()
    settings.set("/rtx/rendermode","RealTimePathTracing")
    settings.set("/rtx/post/aa/op",4)
    env=make_env(enable_cameras=True,control_hz=FPS,log_dir=OUT/"logs")
    env.reset()
    base=env.unwrapped
    physics=tensors.create_simulation_view("numpy",stage_id=get_current_stage_id())
    camera=UsdGeom.Camera.Define(base.sim.stage,"/DemoObserver")
    camera.GetPrim().CreateAttribute("omni:scenePartition",Sdf.ValueTypeNames.Token).Set("env_0")
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.02,-1.03,1.65),
        Gf.Vec3d(.28,-.03,.99),Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(34)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20))
    product=rep.create.render_product(str(camera.GetPath()),(1920,1080))
    rgb=rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
    rgb.attach([product])
    writer=imageio_ffmpeg.write_frames(str(OUT/"videos"/f"{prefix}-original.mp4"),(1920,1440),fps=FPS,
        codec="libx264",pix_fmt_out="yuv420p",macro_block_size=2,
        output_params=["-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
    writer.send(None)
    font=ImageFont.load_default(size=32)
    small=ImageFont.load_default(size=24)
    tensor=lambda values:torch.tensor(np.asarray(values)[None],dtype=torch.float32)
    max_error=0.
    camera_metadata={}
    checkpoints={0,len(rows)-1,*range(15*FPS,len(rows),15*FPS)}
    try:
        for i,row in enumerate(rows):
            omni.timeline.get_timeline_interface().set_current_time(row["t"])
            for side,q in row["robot_joint_positions"].items():
                base.scene[side+"_robot"].write_joint_state_to_sim(tensor(q),torch.zeros((1,len(q))))
            hp=base.scene["headphones"]
            hp.write_root_pose_to_sim(tensor(row["root"]))
            hp.write_joint_state_to_sim(tensor(np.deg2rad(row["joint_degrees"])),torch.zeros((1,14)))
            base.scene["headphone_stand"].write_root_pose_to_sim(tensor(row["stand"]))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False,True,True)
            base.scene.update(1/FPS)
            actual=hp.data.body_link_pose_w.torch[0].cpu().numpy()
            error=float(np.linalg.norm(actual[:,:3]-np.array(row["links"])[:,:3],axis=1).max())
            max_error=max(max_error,error)
            assert error<.0001,(row["t"],error)
            settings.set("/rtx-transient/post/dlss/forceParamReset",True)
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
            for _ in range(4 if i==0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pictures={"observer":np.ascontiguousarray(rgb.get_data()[:,:,:3])}
            assert pictures["observer"].shape==(1080,1920,3)
            for name,_ in VIEWS:
                sensor=base.scene[name]
                sensor.update(1/FPS,force_recompute=True)
                pictures[name]=sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy().copy()
                assert pictures[name].shape==(360,640,3)
                if i==0:
                    camera_metadata[name]=dict(resolution=[640,360],intrinsics=sensor.data.intrinsic_matrices.torch[0].cpu().tolist())
            frame=Image.new("RGB",(1920,1440))
            frame.paste(Image.fromarray(pictures["observer"]))
            for column,(name,_) in enumerate(VIEWS):
                frame.paste(Image.fromarray(pictures[name]),(column*640,1080))
            draw=ImageDraw.Draw(frame)
            labels=[(0,0,"External Observer View",True)]+[(i*640,1080,title,False) for i,(_,title) in enumerate(VIEWS)]
            for x,y,title,observer in labels:
                width=int(font.getlength(title))+32
                draw.rectangle((x,y,x+width-1,y+(84 if observer else 52)-1),fill="black")
                draw.text((x+16,y+10),title,font=font,fill="white")
                if observer:
                    draw.text((x+16,y+50),"Not a robot camera",font=small,fill=(175,215,240))
            if i in checkpoints:
                frame.save(OUT/"frames"/f"{prefix}-{i:05d}.png")
                for name,pixels in pictures.items():
                    Image.fromarray(pixels).save(OUT/"frames"/f"{name}-{i:05d}.png")
            writer.send(np.asarray(frame))
            if i%150==0:
                print("RENDER",i,len(rows),"time",row["t"],flush=True)
        (OUT/"render.json").write_text(json.dumps(dict(frames=len(rows),fps=FPS,
            resolution=[1920,1440],duration_s=len(rows)/FPS,
            playback_speed=1.,source="physics-f64.bin" if dream else "trajectory.json",max_link_position_error_m=max_error,
            observer=dict(eye=[1.02,-1.03,1.65],target=[.28,-.03,.99],focal_length_mm=34,fixed=True),
            cameras=camera_metadata,note="Synchronized measured-state replay; no resimulation or edited outcome."),indent=2)+"\n")
    finally:
        writer.close()
        rgb.detach()
        product.destroy()
        env.close()


if __name__=="__main__":
    import traceback
    from isaaclab.app import AppLauncher
    assert len(sys.argv)==3 and sys.argv[1] in ("demo","dream"),"Usage: python render.py demo|dream RUN_DIRECTORY"
    app=AppLauncher(headless=True,enable_cameras=True,
        kit_args="--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    try:
        run(sys.argv[2],dream=sys.argv[1]=="dream")
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()

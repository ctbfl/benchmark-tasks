"""Render recorded physical states; this file never executes the expert policy."""
from pathlib import Path
import json
import hashlib
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/demo"
IS_DREAM=len(sys.argv)==3 and sys.argv[1]=="--dream"
assert IS_DREAM or sys.argv[1:] in ([],["--frames"]),"Usage: python replay.py [--frames | --dream RUN_DIRECTORY]"
if IS_DREAM:
    OUT=Path(sys.argv[2]).resolve()
    assert OUT.is_relative_to(ROOT/"output"),"Replay output must belong to this task"
FRAMES_ONLY=sys.argv[1:]==["--frames"]
metrics=json.loads((OUT/"metrics.json").read_text())
assert IS_DREAM or metrics["success"] or FRAMES_ONLY,"Expert delivery requires a successful episode"
VIDEO_NAME="dream-four-camera.mp4" if IS_DREAM else "expert-four-camera.mp4"

from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=True,kit_args=
    "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6").app

import imageio_ffmpeg
import numpy as np
import torch
import omni.physx
import omni.replicator.core as rep
import omni.usd
from PIL import Image,ImageDraw,ImageFont
from pxr import Gf,Sdf,UsdGeom
from isaaclab.sim.utils import get_current_stage_id
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
import omni.physics.tensors as tensors
import yam_scene


def run():
    trace=np.load(OUT/"trajectory.npz")
    # Validate the 30 Hz timeline within float32 timestamp precision.
    expected_time=trace["time"][0]+np.arange(len(trace["time"]))/30
    assert np.allclose(trace["time"],expected_time,rtol=0,atol=1e-5)
    changes=np.r_[0,np.flatnonzero(trace["phase"][1:]!=trace["phase"][:-1])+1,len(trace["time"])-1]
    if IS_DREAM:
        changes=np.unique(np.r_[changes,np.arange(0,len(trace["time"]),30*15)])
    indices=changes if FRAMES_ONLY else range(len(trace["time"]))
    yam_scene.OUT=OUT
    env,info=yam_scene.make_env(enable_cameras=True,control_rate=30)
    env.reset()
    base=env.unwrapped
    physics=tensors.create_simulation_view("numpy",stage_id=get_current_stage_id())
    robots=[base.scene[name+"_robot"] for name in ("left","right")]
    tripod=base.scene["mini_tripod"]
    stage=omni.usd.get_context().get_stage()
    camera=UsdGeom.Camera.Define(stage,"/ExpertObserver")
    camera.GetPrim().CreateAttribute("omni:scenePartition",Sdf.ValueTypeNames.Token).Set("env_0")
    transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.85,-.95,1.45),Gf.Vec3d(.27,0,.94),Gf.Vec3d(0,0,1)).GetInverse()
    camera.AddTransformOp().Set(transform)
    camera.CreateFocalLengthAttr(42)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20))
    product=rep.create.render_product(str(camera.GetPath()),(1920,1080))
    rgb=rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
    rgb.attach([product])
    font,small=ImageFont.load_default(size=32),ImageFont.load_default(size=24)
    views=(("left_cam","Left Wrist Camera"),("top_cam","Top Camera"),("right_cam","Right Wrist Camera"))
    (OUT/"frames").mkdir(exist_ok=True)
    writer=None
    if not FRAMES_ONLY:
        (OUT/"videos").mkdir(exist_ok=True)
        writer=imageio_ffmpeg.write_frames(str(OUT/"videos"/VIDEO_NAME),(1920,1440),
            fps=30,codec="libx264",pix_fmt_out="yuv420p",macro_block_size=2,
            output_params=["-an","-profile:v","high","-crf","18","-preset","fast","-threads","4","-movflags","+faststart"])
        writer.send(None)
    try:
        for frame,index in enumerate(indices):
            for side,robot in enumerate(robots):
                q=torch.tensor(trace["robot_q"][index,side][None],dtype=torch.float32)
                robot.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q))
            pose=torch.tensor(trace["hub_pose"][index][None],dtype=torch.float32)
            q=torch.tensor(trace["tripod_q"][index][None],dtype=torch.float32)
            tripod.write_root_pose_to_sim_index(root_pose=pose)
            tripod.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6)))
            tripod.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False,True,True)
            base.scene.update(1/30)
            before=base.sim._physics_step_count
            for _ in range(6 if frame==0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            assert base.sim._physics_step_count==before,"Replay must not advance physics"
            images={"observer":np.ascontiguousarray(rgb.get_data()[...,:3]).copy()}
            for name,_ in views:
                sensor=base.scene[name]
                sensor.update(1/30,force_recompute=True)
                images[name]=sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy().copy()
                assert images[name].shape==(360,640,3)
            assert images["observer"].shape==(1080,1920,3)
            composite=Image.new("RGB",(1920,1440))
            composite.paste(Image.fromarray(images["observer"]))
            for k,(name,_) in enumerate(views):
                composite.paste(Image.fromarray(images[name]),(640*k,1080))
            draw=ImageDraw.Draw(composite)
            labels=[(0,0,"External Observer View",True)]+[(640*k,1080,label,False) for k,(_,label) in enumerate(views)]
            for x,y,label,observer in labels:
                width=int(font.getlength(label))+32
                draw.rectangle((x,y,x+width-1,y+(84 if observer else 52)-1),fill="black")
                draw.text((x+16,y+10),label,font=font,fill="white")
                if observer:
                    draw.text((x+16,y+50),"Not a robot camera",font=small,fill=(175,215,240))
            if index in changes:
                label=f"{index:05d}-{trace['phase'][index]}"
                composite.save(OUT/"frames"/(label+".png"))
                for name,pixels in images.items():
                    Image.fromarray(pixels).save(OUT/"frames"/(label+"-"+name+".png"))
            if writer:
                writer.send(np.asarray(composite))
            if frame%30==0:
                print("render",frame,"simulation time",round(float(trace["time"][index]),2),flush=True)
        shutil.copy2(__file__,OUT/"render-source.py")
        (OUT/"recording.json").write_text(json.dumps({"source":"trajectory.npz","mode":"recorded state replay",
            "replay_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "episode_success":metrics["success"],"frames_only":FRAMES_ONLY,"fps":30,"frames":len(indices),
            "resolution":[1920,1440],"robot_camera_resolution":[640,360],"renderer":"RealTimePathTracing",
            "observer_transform":np.array(transform).tolist()},indent=2)+"\n")
    finally:
        if writer:writer.close()
        rgb.detach()
        product.destroy()
        env.close()


try:
    run()
except Exception:
    traceback.print_exc()
    app.close(exit_code=1)
else:
    app.close()

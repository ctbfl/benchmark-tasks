"""Render recorded physical states at 30 FPS without rerunning a policy."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from config import CONFIG, PHYSICS_HZ

VIEWS = (("left_cam","Left Wrist Camera"),("top_cam","Top Camera"),("right_cam","Right Wrist Camera"))


def render(out):
    import carb
    import carb.eventdispatcher
    import imageio_ffmpeg
    import omni.physx
    import omni.physics.tensors as tensors
    import omni.replicator.core as rep
    import omni.timeline
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from runtime import write_json
    from scene import make_env

    states = np.load(out/"states.npy", allow_pickle=False)
    assert states.ndim == 2 and states.shape[1] == 79 and np.isfinite(states).all()
    assert np.allclose(states[:,0],np.arange(len(states))/PHYSICS_HZ,atol=8e-6,rtol=0)
    assert json.loads((out/"config.json").read_text()) == CONFIG
    rows = states[::PHYSICS_HZ//30]
    # Include t=0 and omit the repeated endpoint to preserve the physical duration.
    rows = rows[:-1] if len(states) % (PHYSICS_HZ//30) == 1 else rows
    assert np.allclose(rows[:,0],np.arange(len(rows))/30,atol=8e-6,rtol=0)
    for name in ("videos","frames"):
        (out/name).mkdir(exist_ok=True)
    env = make_env(out, cameras=True)
    env.reset()
    base = env.unwrapped
    physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
    camera = UsdGeom.Camera.Define(base.sim.stage,"/Observer")
    camera.GetPrim().CreateAttribute("omni:scenePartition",Sdf.ValueTypeNames.Token).Set("env_0")
    eye, target = (.82,.48,1.35), (.32,0,.80)
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
    op = camera.AddTransformOp()
    op.Set(transform)
    camera.CreateFocalLengthAttr(32.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20))
    product = rep.create.render_product(str(camera.GetPath()),(1920,1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
    rgb.attach([product])
    writer = imageio_ffmpeg.write_frames(str(out/"videos/original.mp4"),(1920,1440),fps=30,
        codec="libx264",pix_fmt_out="yuv420p",macro_block_size=2,
        output_params=["-an","-crf","18","-preset","fast","-threads","4","-profile:v","high","-movflags","+faststart"])
    writer.send(None)
    font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=24)
    tensor = lambda a: torch.tensor(np.asarray(a)[None],dtype=torch.float32)
    calibration = {}
    try:
        for i,row in enumerate(rows):
            omni.timeline.get_timeline_interface().set_current_time(i/30)
            for j,side in enumerate(("left","right")):
                base.scene[side+"_robot"].write_joint_state_to_sim(
                    tensor(row[1+8*j:9+8*j]),tensor(row[17+8*j:25+8*j]))
            binder = base.scene["binder"]
            binder.write_root_pose_to_sim(tensor(row[33:40]))
            binder.write_root_velocity_to_sim(tensor(row[40:46]))
            binder.write_joint_state_to_sim(tensor(row[46:49]),tensor(row[49:52]))
            for name,start in (("card",52),("block",65)):
                base.scene[name].write_root_pose_to_sim(tensor(row[start:start+7]))
                base.scene[name].write_root_velocity_to_sim(tensor(row[start+7:start+13]))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False,True,True)
            base.scene.update(1/30)
            # Clear temporal history for this recorded instant, including digital segments.
            carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset",True)
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
            tick = base.sim._physics_step_count
            for _ in range(4 if i==0 else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            assert op.Get() == transform and base.sim._physics_step_count == tick
            frame = Image.new("RGB",(1920,1440))
            pixels = np.ascontiguousarray(rgb.get_data()[...,:3])
            assert pixels.shape == (1080,1920,3)
            frame.paste(Image.fromarray(pixels))
            for j,(name,_) in enumerate(VIEWS):
                sensor = base.scene[name]
                sensor.update(1/30,force_recompute=True)
                pixels = sensor.data.output["rgb"].torch[0,...,:3].cpu().numpy()
                assert pixels.shape == (360,640,3)
                frame.paste(Image.fromarray(pixels),(640*j,1080))
                if i==0:
                    calibration[name] = dict(resolution=[640,360],
                        intrinsics=sensor.data.intrinsic_matrices.torch[0].cpu().tolist(),
                        offset_pos=list(sensor.cfg.offset.pos), offset_rot=list(sensor.cfg.offset.rot))
            draw = ImageDraw.Draw(frame)
            for x,y,title,observer in [(0,0,"External Observer View",True)]+[
                    (640*j,1080,title,False) for j,(_,title) in enumerate(VIEWS)]:
                draw.rectangle((x,y,x+int(font.getlength(title))+31,y+(83 if observer else 51)),fill="black")
                draw.text((x+16,y+10),title,font=font,fill="white")
                if observer:
                    draw.text((x+16,y+50),"Not a robot camera",font=small,fill=(175,215,240))
            caption = f"{i/30:05.1f} s"
            draw.rectangle((0,1036,600,1079),fill="black")
            draw.text((16,1044),caption,font=small,fill="white")
            if i % 150 == 0 or i==len(rows)-1:
                frame.save(out/"frames"/f"frame-{i:05d}.png")
                print("RENDER",i,len(rows),flush=True)
            writer.send(np.asarray(frame))
        write_json(out/"render.json",dict(frames=len(rows),fps=30,resolution=[1920,1440],
            duration_s=len(rows)/30,source="states.npy",observer=dict(eye=eye,target=target,focal_mm=32,fixed=True),
            cameras=calibration,renderer=CONFIG["renderer"],title_px=32,subtitle_px=24,
            source_end_s=(len(states)-1)/PHYSICS_HZ,physics_advanced_during_render=False))
    finally:
        writer.close()
        rgb.detach()
        product.destroy()
        env.close()


if __name__ == "__main__":
    assert len(sys.argv)==2, "Usage: python render.py RUN_DIRECTORY"
    out = Path(sys.argv[1]).resolve()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True,enable_cameras=True,kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8").app
    try:
        render(out)
    except BaseException:
        traceback.print_exc()
        app.close(exit_code=1)
    else:
        app.close()

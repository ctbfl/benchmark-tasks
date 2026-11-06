"""Render synchronized measured-state replay without rerunning the policy or physics."""
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from config import CONFIG


def render(out, previews=False):
    import torch
    import imageio_ffmpeg
    import omni.physics.tensors as tensors
    import omni.physx
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, UsdGeom
    from PIL import Image, ImageDraw, ImageFont
    from isaaclab.sim.utils import get_current_stage_id
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    from scene import make_env

    rows=np.load(out/'trajectory.npz')['frames']
    assert rows.shape[1]==83 and np.allclose(np.diff(rows[:,0]),1/30)
    for name in ('videos','frames'):
        (out/name).mkdir(exist_ok=True)
    env=make_env(cameras=True,log_dir=out/'render-logs')
    env.reset()
    base=env.unwrapped
    physics=tensors.create_simulation_view('numpy',stage_id=get_current_stage_id())
    camera=UsdGeom.Camera.Define(base.sim.stage,'/ObserverCamera')
    camera.GetPrim().CreateAttribute('omni:scenePartition',Sdf.ValueTypeNames.Token).Set('env_0')
    transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.95,-.95,1.52),Gf.Vec3d(.29,-.015,.80),Gf.Vec3d(0,0,1)).GetInverse()
    camera.AddTransformOp().Set(transform)
    camera.CreateFocalLengthAttr(38.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,20.))
    product=rep.create.render_product(str(camera.GetPath()),(1920,1080))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
    rgb.attach([product])
    writer=None
    if not previews:
        writer=imageio_ffmpeg.write_frames(str(out/'videos/original.mp4'),(1920,1440),fps=30,
            codec='libx264',pix_fmt_out='yuv420p',macro_block_size=2,
            output_params=['-an','-profile:v','high','-crf','18','-preset','fast','-threads','4','-movflags','+faststart'])
        writer.send(None)
    font=ImageFont.load_default(size=32)
    small=ImageFont.load_default(size=24)
    tensor=lambda x:torch.tensor(np.asarray(x)[None],dtype=torch.float32)
    views=(('left_cam','Left Wrist Camera'),('top_cam','Top Camera'),('right_cam','Right Wrist Camera'))
    checkpoints={0,len(rows)-1,*range(90,len(rows),90)}
    max_root_error=0.
    camera_calibration={}
    try:
        for i,row in enumerate(rows):
            if previews and i not in checkpoints:
                continue
            for j,side in enumerate(('left','right')):
                base.scene[side+'_robot'].write_joint_state_to_sim(tensor(row[1+j*8:9+j*8]),torch.zeros((1,8)))
            for j in range(6):
                data=row[17+j*11:28+j*11]
                cone=base.scene[f'cone_{j}']
                cone.write_root_pose_to_sim(tensor(data[:7]))
                cone.write_joint_state_to_sim(tensor(data[7:]),torch.zeros((1,4)))
            physics.update_articulations_kinematic()
            omni.physx.get_physx_interface().update_transformations(False,True,True)
            base.scene.update(1/30)
            for j in range(6):
                actual=base.scene[f'cone_{j}'].data.root_pose_w.torch[0].cpu().numpy()
                error=float(np.linalg.norm(actual[:3]-row[17+j*11:20+j*11]))
                max_root_error=max(max_root_error,error)
                assert error<1e-4, (i,j,error)
            tick=base.sim._physics_step_count
            for _ in range(8 if i==0 or previews else 1):
                base.sim.render()
                ensure_isaac_rtx_render_update(force=True)
            pictures=[np.array(rgb.get_data()[...,:3])]
            for name,_ in views:
                sensor=base.scene[name]
                sensor.update(1/30,force_recompute=True)
                pictures.append(sensor.data.output['rgb'].torch[0,...,:3].cpu().numpy().copy())
                if i==0:
                    camera_calibration[name]=sensor.data.intrinsic_matrices.torch[0].cpu().tolist()
            assert base.sim._physics_step_count==tick
            assert pictures[0].shape==(1080,1920,3)
            frame=Image.new('RGB',(1920,1440))
            frame.paste(Image.fromarray(pictures[0]))
            for j,picture in enumerate(pictures[1:]):
                assert picture.shape==(360,640,3)
                frame.paste(Image.fromarray(picture),(j*640,1080))
            draw=ImageDraw.Draw(frame)
            for x,y,title,observer in [(0,0,'External Observer View',True)]+[(j*640,1080,title,False) for j,(_,title) in enumerate(views)]:
                width=int(font.getlength(title))+32
                draw.rectangle((x,y,x+width-1,y+(84 if observer else 52)-1),fill='black')
                draw.text((x+16,y+10),title,font=font,fill='white')
                if observer:
                    draw.text((x+16,y+50),'Not a robot camera',font=small,fill=(175,215,240))
            if i in checkpoints:
                frame.save(out/'frames'/f'{i:05}.png')
            if writer:
                writer.send(np.asarray(frame))
            if i%150==0 or previews:
                print('FRAME',i,len(rows),'time',row[0],flush=True)
        if not previews:
            (out/'render.json').write_text(json.dumps(dict(frames=len(rows),fps=30,duration_s=len(rows)/30,
                width=1920,height=1440,source='trajectory.npz',renderer=CONFIG['renderer'],
                observer_transform=np.array(transform).tolist(),native_camera_size=[640,360],
                camera_intrinsics=camera_calibration,max_root_position_error_m=max_root_error),indent=2)+'\n')
    finally:
        if writer:
            writer.close()
        rgb.detach()
        product.destroy()
        env.close()


if __name__=='__main__':
    assert len(sys.argv) in (2,3) and (len(sys.argv)==2 or sys.argv[2]=='--frames'), 'Usage: python render.py RUN_DIRECTORY [--frames]'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,kit_args='--/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8').app
    try:
        render(Path(sys.argv[1]),len(sys.argv)==3)
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    else:
        app.close()

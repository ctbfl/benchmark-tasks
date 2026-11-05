"""Synchronized 30 FPS camera-ready recording; no changes to expert actions."""

import json
import math
import time
import zlib

import numpy as np
import carb
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
    apply_isaac_rtx_global_settings, ensure_isaac_rtx_render_update,
)
from yam_arena import station
from yam_arena.robot import asset_link_paths

import model

FPS = 30
RENDER = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(
    ambient_light_intensity=0., carb_settings={
        "/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 2,
    }))
VIEWS = (("left_wrist", "Left Wrist Camera"), ("top", "Top Camera"),
         ("right_wrist", "Right Wrist Camera"))


class Recorder:
    def __init__(self, env, out, record=True):
        self.started=time.perf_counter()
        self.env,self.out=env,out
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/"camera_ready_source.py").write_text((model.ROOT/"camera_ready.py").read_text())
        self.ticks=0
        self.stride=round(1/(FPS*env.sim.cfg.dt))
        assert abs(self.stride*env.sim.cfg.dt-1/FPS)<1e-10
        self.products=[]
        self.annotators={}
        self.cameras={}
        self.frames=[]
        self.writer=None
        self.font=ImageFont.load_default(size=32)
        self.subtitle_font=ImageFont.load_default(size=24)
        apply_isaac_rtx_global_settings(RENDER.global_settings)
        # Keep the previous 4:3 preview's vertical field of view in a 16:9 frame.
        observer=rep.create.camera(position=(1.02,-.80,.70),look_at=(.44,0,.12),
                                   focal_length=30,horizontal_aperture=20.955*4/3,clipping_range=(.01,10))
        observer_prim=next(p for p in Usd.PrimRange(observer.get_output_prims()["prims"][0]) if p.IsA(UsdGeom.Camera))
        observer_path=str(observer_prim.GetPath())
        self.add_camera("observer",observer_path,(1920,1080))
        self.observer_path=observer_path
        self.observer_pose=self.camera_pose(observer_path)
        gripper=asset_link_paths()["gripper"]
        mounts={"top":(station.top_camera(),"/World/TopCamera",np.array([0,0,station.TABLE_TOP]))}
        mounts.update({s+"_wrist":(station.wrist_camera(s),f"/World/{s.title()}{gripper}/Camera",np.zeros(3))
                       for s in ("left","right")})
        for name,_ in VIEWS:
            mount,path,shift=mounts[name]
            assert mount.convention=="opengl"
            lens=mount.lens.resized(640,360)
            cfg=sim_utils.PinholeCameraCfg(focal_length=lens.focal_mm,clipping_range=(.01,20),
                horizontal_aperture=lens.width*lens.focal_mm/lens.fx,
                vertical_aperture=lens.height*lens.focal_mm/lens.fy,
                horizontal_aperture_offset=(lens.width/2-lens.cx)*lens.focal_mm/lens.fx,
                vertical_aperture_offset=(lens.cy-lens.height/2)*lens.focal_mm/lens.fy,
                distortion=sim_utils.OpenCvPinholeDistortionCfg(fx=lens.fx,fy=lens.fy,cx=lens.cx,cy=lens.cy,
                                                              image_size=(lens.width,lens.height)))
            cfg.func(path,cfg,translation=tuple(np.array(mount.pos)-shift),orientation=mount.rot_xyzw)
            self.add_camera(name,path,(lens.width,lens.height))
            self.cameras[name].update({"position_local":list(np.array(mount.pos)-shift),
                "rotation_xyzw":list(mount.rot_xyzw),"intrinsics":[lens.fx,lens.fy,lens.cx,lens.cy]})
        for _ in range(8):
            self.read_views()
        self.checkpoint("00 Camera check")
        if record:
            import imageio_ffmpeg
            (self.out/"videos").mkdir(exist_ok=True)
            self.writer=imageio_ffmpeg.write_frames(str(self.out/"videos/four-camera.mp4"),(1920,1440),
                fps=FPS,codec="libx264",pix_fmt_out="yuv420p",macro_block_size=2,
                output_params=["-an","-profile:v","high","-crf","18","-preset","fast",
                               "-threads","4","-movflags","+faststart"])
            self.writer.send(None)
        self.recording_started=time.perf_counter()

    def camera_pose(self,path):
        return np.asarray(UsdGeom.Xformable(self.env.sim.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()))

    def add_camera(self,name,path,size):
        # Apply the same backend configuration before every render product exists.
        apply_isaac_rtx_global_settings(RENDER.global_settings)
        product=rep.create.render_product(path,size)
        rgb=rep.AnnotatorRegistry.get_annotator("rgb",device="cpu")
        rgb.attach(product)
        self.products.append(product)
        self.annotators[name]=rgb
        camera=UsdGeom.Camera(self.env.sim.stage.GetPrimAtPath(path))
        self.cameras[name]={"path":path,"resolution":list(size),"renderer":"RealTimePathTracing",
            "focal_length_mm":camera.GetFocalLengthAttr().Get(),
            "aperture_mm":[camera.GetHorizontalApertureAttr().Get(),camera.GetVerticalApertureAttr().Get()],
            "aperture_offset_mm":[camera.GetHorizontalApertureOffsetAttr().Get(),camera.GetVerticalApertureOffsetAttr().Get()]}

    def read_views(self):
        settings=carb.settings.get_settings()
        assert settings.get("/rtx/rendermode")=="RealTimePathTracing"
        before=self.env.sim._physics_step_count
        self.env.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        assert self.env.sim._physics_step_count==before,"Rendering advanced physics"
        assert np.allclose(self.camera_pose(self.observer_path),self.observer_pose,atol=1e-12)
        images={}
        for name,rgb in self.annotators.items():
            pixels=np.ascontiguousarray(rgb.get_data()[...,:3]).copy()
            width,height=self.cameras[name]["resolution"]
            assert pixels.shape==(height,width,3) and pixels.std()>5,(name,pixels.shape)
            images[name]=pixels
        self.last_views=images
        return images

    def compose(self,images):
        full=Image.new("RGB",(1920,1440))
        full.paste(Image.fromarray(images["observer"]))
        for i,(name,_) in enumerate(VIEWS):
            full.paste(Image.fromarray(images[name]),(640*i,1080))
        draw=ImageDraw.Draw(full)
        labels=[(0,0,"External Observer View","Not a robot camera")]
        labels += [(i*640,1080,title,None) for i,(_,title) in enumerate(VIEWS)]
        for x,y,title,subtitle in labels:
            width=math.ceil(self.font.getlength(title))+32
            draw.rectangle((x,y,x+width-1,y+(84 if subtitle else 52)-1),fill=(0,0,0))
            draw.text((x+16,y+10),title,font=self.font,fill="white")
            if subtitle:
                draw.text((x+16,y+50),subtitle,font=self.subtitle_font,fill=(175,215,240))
        return full

    def step(self):
        self.ticks+=1
        if self.ticks%self.stride:
            return
        images=self.read_views()
        frame=self.compose(images)
        self.writer.send(np.asarray(frame))
        self.frames.append({"index":len(self.frames),"physics_tick":self.ticks,
            "time_s":self.ticks*self.env.sim.cfg.dt,
            "crc32":{name:zlib.crc32(pixels) for name,pixels in images.items()}})
        if len(self.frames)%30==0:
            print(f"Camera-ready: {len(self.frames)} frames / {self.ticks*self.env.sim.cfg.dt:.1f} s; "
                  f"wall {time.perf_counter()-self.recording_started:.1f} s",flush=True)

    def checkpoint(self,label):
        folder=self.out/"raw_frames"/label.replace(" ","_")
        folder.mkdir(parents=True,exist_ok=True)
        images=self.read_views()
        for name,pixels in images.items():
            Image.fromarray(pixels).save(folder/(name+".png"))
        preview=self.compose(images)
        (self.out/"screenshots").mkdir(exist_ok=True)
        preview.save(self.out/"screenshots"/(label.replace(" ","_")+".png"))

    def close(self):
        self.env.physics_callback=None
        if self.writer:
            self.writer.close()
        metadata={"fps":FPS,"frames":len(self.frames),"video_duration_s":len(self.frames)/FPS,
            "simulation_duration_s":self.ticks*self.env.sim.cfg.dt,
            "wall_total_s":time.perf_counter()-self.started,
            "wall_recording_s":time.perf_counter()-self.recording_started,
            "resolution":[1920,1440],"renderer":carb.settings.get_settings().get("/rtx/rendermode"),
            "observer_fixed_world_transform":self.observer_pose.tolist(),"cameras":self.cameras,
            "labels":{"title_px":32,"subtitle_px":24,"outer_margin":0},
            "encoding":{"codec":"libx264","crf":18,"preset":"fast","threads":4,
                        "pixel_format":"yuv420p","profile":"high","faststart":True}}
        (self.out/"recording.json").write_text(json.dumps(metadata,indent=2)+"\n")
        (self.out/"frame_times.json").write_text(json.dumps(self.frames)+"\n")
        for rgb in self.annotators.values():
            rgb.detach()
        for product in self.products:
            product.destroy()


def preview():
    from benchmark import LaptopHolderTask
    env=LaptopHolderTask(cameras=False)
    recorder=Recorder(env,model.ROOT/"outputs/camera_ready_preview",record=False)
    recorder.close()
    env.sim.stop()

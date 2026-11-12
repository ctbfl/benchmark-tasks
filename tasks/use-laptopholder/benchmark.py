"""Standalone YAM benchmark. Import after AppLauncher; actions are 14 joint targets."""

import json
import carb
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
from isaaclab.assets import Articulation
import isaaclab.sim as sim_utils
from yam_arena import station
from yam_arena.robot import asset_link_paths, yam_arm_cfg
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN

import model
import task
from runtime import Mechanism, simulation


class LaptopHolderTask:
    def __init__(self, cameras=False):
        self.physics_callback = None
        self.sim = simulation()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        # Use the calibrated YAM station, keeping the task's tabletop at z=0.
        stage=self.sim.stage
        offset=np.array([0,0,station.TABLE_TOP])
        stage.RemovePrim("/World/Table")
        table=model.material(stage,"/World/materials/table",(.792,.524,.273),(.65,.5))
        model.shape(stage,"/World/Table",model.block(station.TABLE_SIZE,np.array(station.TABLE_CENTER)-offset),table)
        floor=model.material(stage,"/World/materials/floor",(.45,.45,.46),(.65,.5))
        model.shape(stage,"/World/Floor",model.block((200,200,.02),(0,0,station.FLOOR_Z-station.TABLE_TOP-.01)),floor)
        dome=UsdLux.DomeLight(stage.GetPrimAtPath("/World/Dome"))
        dome.GetIntensityAttr().Set(250)
        dome.CreateColorAttr(Gf.Vec3f(*station.linear_rgb(station.SKY)))
        stage.RemovePrim("/World/Key")
        lamp=UsdLux.SphereLight.Define(stage,"/World/Key")
        lamp.CreateIntensityAttr(6e5)
        lamp.CreateRadiusAttr(.1)
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(*(np.array(station.LAMP_POS)-offset)))
        stand_mat=model.material(stage,"/World/materials/camera_stand",station.linear_rgb(station.STAND),(.5,.4))
        housing_mat=model.material(stage,"/World/materials/camera_housing",station.linear_rgb(station.HOUSING),(.5,.4))
        for name,(kind,size,pos,rot) in station.camera_stand(station.top_camera()).items():
            if kind=="box":
                mesh=model.block(size,(0,0,0))
            else:
                assert kind=="capsule",kind
                mesh=trimesh.creation.capsule(radius=size[0],height=size[1],count=[16,16])
                mesh.vertices-=mesh.bounds.mean(axis=0)
            mesh.vertices=mesh.vertices@Rotation.from_quat(rot).as_matrix().T+np.array(pos)-offset
            mat=housing_mat if name=="camera_housing" else stand_mat
            model.shape(stage,f"/World/CameraStand/{name}",mesh,mat)
        self.robots = [Articulation(yam_arm_cfg(f"/World/{side.title()}",tuple(np.array(base)-offset),
                       contact_offset=.001,dream_look=True)) for side,base in station.BASES.items()]
        model.build(self.sim.stage,"/World/Holder",origin=task.START)
        for name in model.NAMES:
            prim=self.sim.stage.GetPrimAtPath("/World/Holder/links/"+name)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        mark=model.material(self.sim.stage,"/World/materials/goal",(.55,.008,.006),(.5,.4))
        lo,hi=np.array(task.GOAL_LOW),np.array(task.GOAL_HIGH)
        mid=(lo+hi)/2
        tape=.012  # 12 mm paper tape; its inner edge is the original goal boundary.
        for i,(size,pos) in enumerate([
            ((hi[0]-lo[0]+2*tape,tape,.0002),(mid[0],lo[1]-tape/2,.0001)),
            ((hi[0]-lo[0]+2*tape,tape,.0002),(mid[0],hi[1]+tape/2,.0001)),
            ((tape,hi[1]-lo[1],.0002),(lo[0]-tape/2,mid[1],.0001)),
            ((tape,hi[1]-lo[1],.0002),(hi[0]+tape/2,mid[1],.0001))]):
            model.shape(self.sim.stage,f"/World/Goal/edge_{i}",model.block(size,pos),mark,False)
        self.rgb=None
        self.camera_views={}
        self.camera_products=[]
        if cameras:
            import omni.replicator.core as rep
            camera=rep.create.camera(position=(1.55,-1.75,1.35),look_at=(.14,0,.38),focal_length=26.5,clipping_range=(.01,20))
            self.product=rep.create.render_product(camera,(1600,1200))
            self.rgb=rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb.attach(self.product)
            gripper=asset_link_paths()["gripper"]
            mounts=[("top",station.top_camera(),"/World/TopCamera",offset)]
            mounts += [(side+"_wrist",station.wrist_camera(side),f"/World/{side.title()}{gripper}/Camera",np.zeros(3))
                       for side in ("left","right")]
            for name,mount,path,shift in mounts:
                assert mount.convention=="opengl"
                lens=mount.lens.resized(1280,720)
                cfg=sim_utils.PinholeCameraCfg(focal_length=lens.focal_mm,clipping_range=(.01,20),
                    distortion=sim_utils.OpenCvPinholeDistortionCfg(fx=lens.fx,fy=lens.fy,cx=lens.cx,cy=lens.cy,
                                                                  image_size=(lens.width,lens.height)))
                cfg.func(path,cfg,translation=tuple(np.array(mount.pos)-shift),orientation=mount.rot_xyzw)
                product=rep.create.render_product(path,(lens.width,lens.height))
                annotator=rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach(product)
                self.camera_products.append(product)
                self.camera_views[name]=annotator
        self.sim.reset()
        self.holder=Mechanism(self.sim,"/World/Holder")
        self.art=self.holder.view.create_articulation_view("/World/Holder/links/deck_L")
        self.initial_root=self.art.get_root_transforms().copy()
        self.initial_dof=self.art.get_dof_positions().copy()
        self.ids=[]
        robot_paths=[]
        for robot,side in zip(self.robots,("Left","Right")):
            self.ids.append((robot.find_joints(list(ARM_JOINTS),preserve_order=True)[0],
                             robot.find_joints(list(FINGER_JOINTS),preserve_order=True)[0]))
            robot_paths += [str(p.GetPath()) for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/'+side))
                            if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        sensor_paths=list(self.holder.bodies.prim_paths)
        self.contacts=self.holder.view.create_rigid_contact_view(sensor_paths,filter_patterns=[robot_paths]*len(sensor_paths))
        assert self.contacts.sensor_count==10 and self.contacts.filter_count==len(robot_paths)
        self.reset()

    def reset(self):
        index=np.array([0],np.uint32)
        self.art.set_root_transforms(self.initial_root,index)
        self.art.set_root_velocities(np.zeros((1,6),np.float32),index)
        self.art.set_dof_positions(self.initial_dof,index)
        self.art.set_dof_velocities(np.zeros_like(self.initial_dof),index)
        for robot in self.robots:
            q=robot.data.default_joint_pos.torch.clone()
            v=robot.data.default_joint_vel.torch.clone()
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=v)
            robot.actuators.target_command.set_position_index(value=q)
            robot.actuators.target_command.set_velocity_index(value=v)
            robot.update(self.sim.cfg.dt)
        self.scorer=task.Score()
        self.steps=0
        for _ in range(round(.5/self.sim.cfg.dt)):
            self._tick()
        return self.observe()

    def _tick(self):
        for robot,(arm_ids,_) in zip(self.robots,self.ids):
            gravity=robot.data.gravity_compensation_forces.torch[:,arm_ids]
            robot.actuators.target_command.set_effort_index(value=gravity.contiguous(),joint_ids=arm_ids)
            robot.write_data_to_sim()
        self.sim.step(render=False)
        for robot in self.robots:
            robot.update(self.sim.cfg.dt)
        if self.physics_callback is not None:
            self.physics_callback()

    def joint_positions(self):
        parts=[]
        for robot,(arm_ids,finger_ids) in zip(self.robots,self.ids):
            q=robot.data.joint_pos.torch[0].cpu().numpy()
            parts.append(np.r_[q[arm_ids],np.clip(q[finger_ids].mean()/FINGER_OPEN,0,1)])
        return np.concatenate(parts)

    def observe(self):
        result={"joint_pos":self.joint_positions(),"instruction":task.INSTRUCTION}
        if self.rgb is not None:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
            self.sim.render()
            ensure_isaac_rtx_render_update(force=True)
            pixels=self.rgb.get_data()
            assert pixels.size,"RGB camera did not render"
            result["rgb"]=np.ascontiguousarray(pixels[...,:3])
            result["camera_rgb"]={}
            for name,annotator in self.camera_views.items():
                pixels=annotator.get_data()
                assert pixels.size,f"{name} camera did not render"
                result["camera_rgb"][name]=np.ascontiguousarray(pixels[...,:3])
        return result

    def step(self,action):
        action=np.asarray(action,dtype=np.float32)
        assert action.shape==(14,) and np.isfinite(action).all()
        for i,(robot,(arm_ids,finger_ids)) in enumerate(zip(self.robots,self.ids)):
            limits=robot.data.joint_pos_limits.torch[0,arm_ids].cpu().numpy()
            q=np.clip(action[i*7:i*7+6],limits[:,0],limits[:,1])
            grip=float(np.clip(action[i*7+6],0,1))*FINGER_OPEN
            robot.actuators.target_command.set_position_index(value=torch.tensor(q[None]),joint_ids=arm_ids)
            robot.actuators.target_command.set_position_index(value=torch.full((1,2),grip),joint_ids=finger_ids)
        contact=0.0
        for _ in range(round(1/(self.sim.cfg.dt*task.CONTROL_HZ))):
            self._tick()
            matrix=self.contacts.get_contact_force_matrix(self.sim.cfg.dt)
            contact=max(contact,float(np.linalg.norm(matrix,axis=-1).max()))
        self.steps+=1
        info=self.scorer.update(self.holder.measure())
        info["robot_contact_N"]=contact
        info["truncated"]=self.steps>=task.MAX_STEPS
        return self.observe(),info


def capture_views():
    import cv2
    env=LaptopHolderTask(cameras=True)
    for _ in range(16):
        observation=env.observe()
    out=model.ROOT/"outputs/benchmark/cameras"
    out.mkdir(parents=True,exist_ok=True)
    frames={"overview":observation["rgb"],**observation["camera_rgb"]}
    for name,pixels in frames.items():
        assert cv2.imwrite(str(out/(name+".png")),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
    print("Saved camera views:",str(out),flush=True)
    env.sim.stop()


def smoke(cameras=True):
    env=LaptopHolderTask(cameras=cameras)
    q=env.joint_positions()
    for _ in range(20):
        observation,info=env.step(q)
    assert not info["success"] and info["score"]==0
    assert env.holder.measure()["width"]<.06
    reset=env.reset()
    assert env.steps==0 and env.scorer.update(env.holder.measure())["score"]==0
    out=model.ROOT/"outputs/benchmark"
    out.mkdir(exist_ok=True)
    if cameras:
        import cv2
        for _ in range(12):
            reset=env.observe()
        cv2.imwrite(str(out/"setup.png"),cv2.cvtColor(reset["rgb"],cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out/"red_frame_setup.png"),cv2.cvtColor(reset["rgb"],cv2.COLOR_RGB2BGR))
    # Sensor acceptance fixture: drop the stand over the left robot base. This
    # reset-only placement checks real robot/holder contact, not manipulation.
    root=env.initial_root.copy()
    root[0,:3]+=np.array([0,.305,1.2])-np.array(task.START)
    env.art.set_root_transforms(root,np.array([0],np.uint32))
    peak_contact=0.0
    for _ in range(round(1/env.sim.cfg.dt)):
        env._tick()
        matrix=env.contacts.get_contact_force_matrix(env.sim.cfg.dt)
        peak_contact=max(peak_contact,float(np.linalg.norm(matrix,axis=-1).max()))
    assert peak_contact>.1, f"Robot contact sensor did not detect the contact fixture: {peak_contact} N"
    env.reset()
    (out/"smoke.json").write_text(json.dumps({"passed":True,"action_shape":[14],"control_hz":task.CONTROL_HZ,
        "physics_hz":round(1/env.sim.cfg.dt),
        "robot_bodies":env.contacts.filter_count,"contact_fixture_peak_N":peak_contact,
        "initial_score":info,"holder":env.holder.measure()},indent=2)+"\n")
    env.sim.stop()

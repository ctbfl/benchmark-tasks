"""State-based expert using only the two YAM arms' joint-target actions."""

import json
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena import ik, station
from yam_arena.demo import grip_rotation
from yam_arena.runtime import link_poses, tcp_world

import model
import task
from benchmark import LaptopHolderTask
from validate import seated

TILT = Rotation.from_rotvec([0,-np.pi/4,0]).as_matrix() @ grip_rotation()


class Expert:
    def __init__(self, render, camera_ready=False):
        self.env=LaptopHolderTask(cameras=False)
        self.out=model.ROOT/"outputs/camera_ready" if camera_ready else model.ROOT/"outputs/expert"/("video" if render else "headless")
        self.out.mkdir(parents=True,exist_ok=True)
        (self.out/"expert_source.py").write_text((model.ROOT/"expert.py").read_text())
        self.render=render
        self.q=self.env.joint_positions().reshape(2,7).copy()
        self.bases=[np.asarray(station.BASES[s])-[0,0,station.TABLE_TOP] for s in ("left","right")]
        self.rows=[]
        self.actions=[]
        self.phases=[]
        self.video=None
        self.recorder=None
        if camera_ready:
            from camera_ready import Recorder
            self.recorder=Recorder(self.env,self.out)
            self.env.physics_callback=self.recorder.step
        if render:
            import omni.replicator.core as rep
            camera=rep.create.camera(position=(1.02,-.80,.70),look_at=(.44,0,.12),focal_length=30,clipping_range=(.01,10))
            self.product=rep.create.render_product(camera,(960,720))
            self.rgb=rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb.attach(self.product)
            (self.out/"videos").mkdir(exist_ok=True)
            self.video=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo",
                "-pix_fmt","rgb24","-s","960x720","-r","10","-i","-","-an","-c:v","libx264","-crf","20",
                "-pix_fmt","yuv420p","-movflags","+faststart",str(self.out/"videos/expert.mp4")],stdin=subprocess.PIPE)

    def pose(self, arm):
        robot=self.env.robots[arm]
        return tcp_world(robot),link_poses(robot)["gripper"][:3,:3]

    def tick(self, label):
        action=self.q.reshape(14).copy()
        _,info=self.env.step(action)
        m=self.env.holder.measure()
        self.actions.append(action)
        self.rows.append({"phase":label,"time_s":self.env.steps/task.CONTROL_HZ,"score":info,"holder":m,
                          "tcp":[self.pose(a)[0].tolist() for a in (0,1)],
                          "joint_pos":self.env.joint_positions().tolist()})
        assert m["joint_error"]<.003 and m["max_speed"]<2, "Mechanism lost stability"
        if self.render and self.env.steps%2==0:
            self.frame(label)

    def frame(self,label):
        import cv2
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.env.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        pixels=self.rgb.get_data()
        assert pixels.size,"Camera did not render"
        frame=np.ascontiguousarray(pixels[...,:3]).copy()
        cv2.rectangle(frame,(0,0),(960,65),(25,28,32),-1)
        cv2.putText(frame,"YAM EXPERT | "+label,(18,27),cv2.FONT_HERSHEY_SIMPLEX,.60,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(frame,f"Target: highest / slot 1    Score: {self.rows[-1]['score']['score']:.2f}",
                    (18,52),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,220,220),1,cv2.LINE_AA)
        self.video.stdin.write(frame.tobytes())
        self.last_frame=frame

    def move(self,label,seconds,goals):
        starts={a:(*self.pose(a),self.q[a,6]) for a in goals}
        def targets(u):
            result={}
            for a,(p1,r1,g1) in goals.items():
                p0,r0,g0=starts[a]
                result[a]=(p0+u*(np.asarray(p1)-p0),ik.slerp_matrix(r0,r1,u),g0+u*(g1-g0))
            return result
        self.follow(label,seconds,targets)

    def follow(self,label,seconds,targets,stop=None):
        for t in np.linspace(0,1,round(seconds*task.CONTROL_HZ)):
            for a,(p,r,g) in targets(ik.smoothstep(t)).items():
                q,pe,re=ik.solve(p,ik.matrix_to_quat_xyzw(r),self.q[a,:6],self.bases[a],iters=80,damping=.015)
                assert pe<.002 and re<.04,f"IK failed {label}: arm {a}, {pe:.4f} m / {np.rad2deg(re):.2f} deg"
                self.q[a]=np.r_[q,g]
            self.tick(label)
            if stop is not None and stop():
                break
        self.hold(label+" settle",.5)
        self.report(label)

    def hold(self,label,seconds):
        for _ in range(round(seconds*task.CONTROL_HZ)):
            self.tick(label)

    def report(self,label):
        m=self.env.holder.measure()
        entry={"name":label,"step":self.env.steps,"state":m,"tcp":[self.pose(a)[0].tolist() for a in (0,1)],
               "bodies":{n:{"position":p.tolist(),"rotation":r.tolist()} for n,(p,r,_) in self.env.holder.state().items()}}
        self.phases.append(entry)
        print(label,json.dumps(entry),flush=True)
        if self.render:
            import cv2
            cv2.imwrite(str(self.out/(label.replace(" ","_")+".png")),cv2.cvtColor(self.last_frame,cv2.COLOR_RGB2BGR))
        if self.recorder:
            self.recorder.checkpoint(label)

    def close(self,error=None):
        if self.recorder:
            self.recorder.close()
        if self.video:
            self.video.stdin.close()
            assert self.video.wait(timeout=30)==0
        np.savez_compressed(self.out/"actions.npz",actions=np.asarray(self.actions))
        result={"method":"YAM joint targets only; no object forces, pose writes or grasp constraints",
                "target_gear":task.TARGET_GEAR,"phases":self.phases,"error":error,
                "final":self.rows[-1] if self.rows else None,"steps":self.env.steps,
                "bodies":{n:{"position":p.tolist(),"rotation":r.tolist()} for n,(p,r,_) in self.env.holder.state().items()}}
        (self.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        (self.out/"trajectory.json").write_text(json.dumps(self.rows)+"\n")
        self.env.sim.stop()


def run(render=False,camera_ready=False):
    expert=Expert(render,camera_ready)
    error=None
    try:
        expert.hold("Initial folded state",.5)
        expert.report("00 Initial")
        expert.move("01 Approach far gap",4,{0:([.596,.032,.065],TILT,.64)})
        expert.move("02 Lower outside tail",3,{0:([.596,.032,.027],TILT,.64)})
        expert.move("03 Enter from tail",4,{0:([.583,.032,.027],TILT,.64)})
        expert.move("04 Lower finger into gap",3,{0:([.583,.032,.021],TILT,.64)})
        expert.move("05 Spread with one finger",5,{0:([.583,.004,.021],TILT,1.0)})
        expert.hold("Hold gap open",1)
        expert.report("06 Initial opening")
        assert expert.env.holder.measure()["width"]>.08,"Initial opening failed"
        def grasp_pose(side,x):
            p,r,_=expert.env.holder.state()["deck_"+side]
            sign=1 if side=="R" else -1
            z=np.array([0,-sign/np.sqrt(2),-1/np.sqrt(2)])
            gx=np.array([0,-1/np.sqrt(2),sign/np.sqrt(2)])
            grip_r=np.column_stack([gx,np.cross(z,gx),z])
            return p+r@np.array([x,sign*.009,.010]),grip_r
        p,r=grasp_pose("L",.11)
        expert.move("07 Right approach rail",3,{1:(p-r[:,2]*.10,r,.30)})
        expert.move("08 Right descend",3,{1:(p,r,.30)})
        expert.move("09 Right grasp",2,{1:(p,r,.05)})
        p0,r0=expert.pose(0)
        expert.move("10 Withdraw spreading finger",3,{0:(p0+[0,0,.09],r0,1.0)})
        expert.move("10b Retract left wrist",3,{0:([.35,.16,.15],r0,.30)})
        p,r=grasp_pose("R",.11)
        expert.move("10c Turn left wrist",3,{0:([.35,.16,.15],r,.30)})
        expert.move("11 Left approach rail",3,{0:(p-r[:,2]*.10,r,.30)})
        expert.move("12 Left descend",3,{0:(p,r,.30)})
        expert.move("13 Left grasp",2,{0:(p,r,.05)})
        state=expert.env.holder.state()
        goals={}
        delta=(.195-expert.env.holder.measure()["width"])/2
        for a,side in ((0,"R"),(1,"L")):
            p,r,_=state["deck_"+side]
            tp,tr=expert.pose(a)
            goals[a]=(tp+(1 if a==0 else -1)*delta*r[:,1],tr,.05)
        expert.move("14 Pull both rails apart",5,goals)
        assert expert.env.holder.measure()["width"]>.180,"Bimanual spread did not reach full width"
        def tilt(label,angle,seconds,stop=None):
            state=expert.env.holder.state()
            frames={}
            for a,side in ((0,"R"),(1,"L")):
                p,r,_=state["deck_"+side]
                tp,tr=expert.pose(a)
                base_r=state["base_"+side][1]
                frames[a]=(p,r,r.T@(tp-p),r.T@tr,base_r)
            def targets(u):
                result={}
                for a,(p,r,local_p,local_r,base_r) in frames.items():
                    rot=ik.slerp_matrix(r,base_r@model.ry(-np.deg2rad(angle)),u)
                    result[a]=(p+rot@local_p,rot@local_r,.05)
                return result
            expert.follow(label,seconds,targets,stop)
        tilt("15 Raise both decks",52,5)
        expert.hold("Let support rods swing down",2)
        expert.report("16 Support rods hanging")
        tilt("17 Lower toward highest sockets",39.5,6)
        def guide(arm,side):
            sign=1 if side=="R" else -1
            p,r=expert.pose(arm)
            expert.move("18 Free hand "+side,1,{arm:(p,r,.5)})
            expert.move("19 Withdraw hand "+side,1.5,{arm:(p-r[:,2]*.07,r,.5)})
            safe=np.array([.38,sign*.20,.15])
            expert.move("20 Retract wrist "+side,1.5,{arm:(safe,r,.1)})
            push_r=Rotation.from_euler("z",-sign*90,degrees=True).as_matrix()@Rotation.from_euler("y",-60,degrees=True).as_matrix()@grip_rotation()
            expert.move("21 Turn grasping fingers "+side,1.5,{arm:(safe,push_r,.18)})
            state=expert.env.holder.state()
            pp,pr,_=state["prop_"+side]
            br=state["base_"+side][1]
            point=pp+pr@np.array([.07,0,0])
            expert.move("22 Approach rod "+side,1.5,{arm:(point-push_r[:,2]*.06,push_r,.18)})
            expert.move("23 Reach rod "+side,1.5,{arm:(point,push_r,.18)})
            expert.move("23b Grasp rod "+side,1,{arm:(point,push_r,0)})
            start=expert.pose(arm)[0]
            end=start+br[:,0]*.014
            expert.follow("24 Guide rod "+side,2,lambda u:{arm:(start+u*(end-start),push_r,0)},
                          lambda:expert.env.holder.measure()["sides"][side]["toe_local"][0]>.1055)
            p,_=expert.pose(arm)
            expert.move("24b Release rod "+side,1,{arm:(p,push_r,.18)})
            expert.move("25 Withdraw from rod "+side,1.5,{arm:(p-push_r[:,2]*.07,push_r,.18)})
        guide(1,"L")
        holding=0
        if not seated(expert.env.holder.measure(),1):
            p,r=grasp_pose("L",.11)
            r=expert.env.holder.state()["deck_L"][1]@r
            expert.move("26 Reapproach deck L",1.5,{1:(p-r[:,2]*.07,r,.30)})
            expert.move("27 Regrasp deck L",1.5,{1:(p,r,.30)})
            expert.move("28 Hold deck L",1,{1:(p,r,.05)})
            state=expert.env.holder.state()
            p,r,_=state["deck_L"]
            tp,tr=expert.pose(1)
            down=state["base_L"][1]@model.ry(-np.deg2rad(39.5))
            expert.move("29 Bias held deck toward sockets",1.5,{1:(p+down@r.T@(tp-p),down@r.T@tr,.05)})
            guide(0,"R")
            holding=1
        expert.hold("Let pins seat",1)
        expert.report("30 Guided supports")
        assert seated(expert.env.holder.measure(),1),"Support pins missed highest sockets"
        side="R" if holding==0 else "L"
        sign=1 if holding==0 else -1
        p,r=grasp_pose(side,.11)
        r=expert.env.holder.state()["deck_"+side][1]@r
        expert.move("30b Reapproach seated deck",1.5,{holding:(p-r[:,2]*.06,r,.30)})
        expert.move("30c Reach seated deck",1.5,{holding:(p,r,.30)})
        expert.move("30d Grasp seated deck",1,{holding:(p,r,.05)})
        p,r=expert.pose(holding)
        br=expert.env.holder.state()["base_"+side][1]
        delta=.194-expert.env.holder.measure()["width"]
        expert.move("31 Restore full width",2,{holding:(p+sign*br[:,1]*delta,r,.05)})
        expert.move("32 Release both grippers",2,{a:(*expert.pose(a),.5) for a in (0,1)})
        p,r=expert.pose(holding)
        expert.move("33 Withdraw holding hand",2,{holding:(p-r[:,2]*.06,r,.5)})
        expert.hold("Released stability check",2)
        expert.report("34 Final")
        assert expert.rows[-1]["score"]["success"] and not expert.rows[-1]["score"]["truncated"],"Final benchmark conditions not met"
    except Exception as exc:
        error=str(exc)
        raise
    finally:
        expert.close(error)

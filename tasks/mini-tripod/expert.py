"""Tripod expert: robot joint commands and physical finger contact only."""

from pathlib import Path
import json
import traceback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/demo"
CONTROL_RATE = 30
MOTION_SPEED = 3.

# Start Kit before SciPy creates BLAS threads: Kit's telemetry startup forks.
if __name__=="__main__":
    from isaaclab.app import AppLauncher
    (OUT/"logs").mkdir(parents=True,exist_ok=True)
    app=AppLauncher(headless=True,enable_cameras=False,kit_args=
        "--/plugins/carb.tasking.plugin/threadCount=6 --/plugins/omni.tbb.globalcontrol/maxThreadCount=6 "
        f"--/log/file={OUT}/logs/kit.log").app

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PAD_Z = (.140, .1445)  # Left disk grasp; right tips isolate a single leg.


class Expert:
    def __init__(self, env, info):
        import torch
        import carb
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY
        from yam_arena.robot import asset_link_paths

        self.env, self.base = env, env.unwrapped
        self.info, self.airborne = info, False
        self.torch = torch
        self.action = np.array([*READY,1.,*READY,1.],np.float32)
        self.bias = np.zeros((2,6))
        self.robots = [self.base.scene[s+"_robot"] for s in ("left","right")]
        self.tripod = self.base.scene["mini_tripod"]
        self.gripper_ids = [r.body_names.index("gripper") for r in self.robots]
        self.physics = tensors.create_simulation_view("numpy",stage_id=get_current_stage_id())
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing",False)
        paths = asset_link_paths()
        self.contacts = [self.physics.create_rigid_contact_view(
            f"/World/envs/env_0/{side}Arm"+paths["tip_left"].rsplit("/",1)[0]+"/tip_*",
            [f"/World/envs/env_0/mini_tripod/{name}" for name in self.tripod.body_names])
            for side in ("Left","Right")]
        self.feet=self.physics.create_rigid_contact_view("/World/envs/env_0/mini_tripod/leg_*",
            ["/World/envs/env_0/table/geometry/mesh"])
        self.robot_contacts=self.physics.create_rigid_contact_view("/World/envs/env_0/mini_tripod/leg_*",
            [f"/World/envs/env_0/{side}Arm"+path for side in ("Left","Right") for path in paths.values()])
        right_paths=["/World/envs/env_0/RightArm"+path for path in paths.values()]
        left_paths=["/World/envs/env_0/LeftArm"+path for path in paths.values()]
        self.arm_contacts=self.physics.create_rigid_contact_view(right_paths,[left_paths]*len(right_paths))
        self.log, self.phase, self.t = [], "settle", 0.
        self.wait(2)
        self.left_park = self.hand(0)
        self.right_park = self.hand(1)

    def pose(self):
        p = self.tripod.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        return p[:3], Rotation.from_quat(p[3:])

    def hand(self, side):
        p = self.robots[side].data.body_link_pose_w.torch[0,self.gripper_ids[side]].cpu().numpy()
        r = Rotation.from_quat(p[3:])
        return p[:3]+r.apply([0,0,PAD_Z[side]]), r

    def collision_cloud(self):
        from model import ASSETS
        view=self.physics.create_articulation_view("/World/envs/env_0/mini_tripod")
        vertices=dict(np.load(ASSETS/"collision_vertices.npz"))
        return np.concatenate([Rotation.from_quat(pose[3:]).apply(vertices[name])+pose[:3]
            for name,pose in zip(view.shared_metatype.link_names,view.get_link_transforms()[0])])

    def step(self, n=1, moving_side=None, velocity=None):
        for _ in range(n):
            command=self.action.copy()
            velocity_targets=np.zeros((2,6),np.float32)
            if moving_side is not None:
                velocity_targets[moving_side]=velocity
            for side in range(2):
                command[side*7:side*7+6] += self.bias[side]
                self.robots[side].actuators.target_command.set_velocity_index(
                    value=self.torch.tensor(velocity_targets[side][None],device=self.base.device),
                    joint_ids=list(range(6)))
            self.env.step(self.torch.tensor(command[None],device=self.base.device))
            for side,robot in enumerate(self.robots):
                actual=robot.data.joint_pos.torch[0,:6].cpu().numpy()
                error=self.action[side*7:side*7+6]-actual
                self.bias[side]=np.clip(self.bias[side]+1.5*self.base.step_dt*error,-.10,.10)
            self.t += self.base.step_dt
            p,r = self.pose()
            q = self.tripod.data.joint_pos.torch[0].cpu().numpy().copy()
            forces = np.array([c.get_contact_force_matrix(self.base.physics_dt) for c in self.contacts])
            self.log.append((self.t,self.phase,command.copy(),np.r_[p,r.as_quat()],q,
                np.array([r.data.joint_pos.torch[0].cpu().numpy().copy() for r in self.robots]),forces,
                self.feet.get_contact_force_matrix(self.base.physics_dt).copy(),
                self.robot_contacts.get_contact_force_matrix(self.base.physics_dt).copy(),
                self.tripod.data.root_link_vel_w.torch[0].cpu().numpy().copy(),
                self.tripod.data.joint_vel.torch[0].cpu().numpy().copy(),
                self.arm_contacts.get_contact_force_matrix(self.base.physics_dt).copy(),velocity_targets))
            assert np.linalg.norm(self.log[-1][11],axis=-1).max()<5,"Arms collide"
            assert np.isfinite(q).all() and np.linalg.norm(p)<2, "Invalid tripod state"
            assert not self.airborne or p[2]>.79, "Tripod dropped from gripper"

    def wait(self, seconds):
        self.step(round(seconds/self.base.step_dt))

    def mark(self, phase):
        self.phase = phase
        p,r = self.pose()
        q = self.tripod.data.joint_pos.torch[0].cpu().numpy()
        print(phase, "t",round(self.t,1),"hub",p.round(4),"q",q.round(5),
            "finger_contacts",np.linalg.norm(self.log[-1][6],axis=-1).round(2),flush=True)

    def move(self, side, point, rotation, seconds=2.):
        from yam_arena import station
        from yam_arena.ik import solve, smoothstep
        p0,r0 = self.hand(side)
        interp = Slerp([0,1],Rotation.concatenate([r0,rotation]))
        offset = side*7
        q = self.action[offset:offset+6].copy()
        for s in np.linspace(0,1,max(2,round(seconds/(MOTION_SPEED*self.base.step_dt)))):
            a = smoothstep(s)
            r = interp(a)
            p = (1-a)*p0+a*np.asarray(point)
            tcp = p+r.apply([0,0,.134-PAD_Z[side]])
            q,ep,er = solve(tcp,r.as_quat(),q,station.BASES[("left","right")[side]],
                damping=.015,iters=100,tol=1e-5)
            assert ep<.002 and er<.025, ("IK unreachable",side,p.tolist(),ep,er)
            velocity=(q-self.action[offset:offset+6])/self.base.step_dt if s>0 else np.zeros(6)
            self.action[offset:offset+6] = q
            self.step(moving_side=side,velocity=velocity)
        self.wait((1 if seconds>=1 else .1)/MOTION_SPEED)

    def grip(self, side, width, seconds=1., follow=None):
        from yam_arena import station
        from yam_arena.ik import solve
        j=side*7+6
        for g in np.linspace(self.action[j],width/.0939,max(2,round(seconds/(MOTION_SPEED*self.base.step_dt)))):
            if follow:
                p,r=follow()
                q,ep,er=solve(p+r.apply([0,0,.134-PAD_Z[side]]),r.as_quat(),
                    self.action[side*7:side*7+6],station.BASES[("left","right")[side]],damping=.015,iters=100)
                assert ep<.002 and er<.025, ("Grasp tracking IK",ep,er)
                self.action[side*7:side*7+6]=q
            self.action[j]=g
            self.step()
        self.wait(.5/MOTION_SPEED)

    def move_joints(self, side, target, seconds):
        from yam_arena.ik import smoothstep
        j=7*side
        start=self.robots[side].data.joint_pos.torch[0,:6].cpu().numpy().copy()
        for s in np.linspace(0,1,max(2,round(seconds/(MOTION_SPEED*self.base.step_dt)))):
            a=smoothstep(s)
            q=(1-a)*start+a*target
            velocity=(q-self.action[j:j+6])/self.base.step_dt if s>0 else np.zeros(6)
            self.action[j:j+6]=q
            self.step(moving_side=side,velocity=velocity)
        self.wait(1/MOTION_SPEED)

    def save(self, success, error=None):
        p,r = self.pose()
        data = {"success":success,"error":error,"phase":self.phase,"time_s":self.t,
            "policy":"Privileged-state expert; robot joint position/velocity commands and physical contact only",
            "control_rate_hz":CONTROL_RATE,
            "motion_speed_multiplier":MOTION_SPEED,
            "root_position_m":p.tolist(),"root_quaternion_xyzw":r.as_quat().tolist(),
            "joint_names":self.tripod.joint_names,
            "joint_positions":self.tripod.data.joint_pos.torch[0].cpu().tolist()}
        if success:
            data["validation"]=self.validation
        (OUT/"metrics.json").write_text(json.dumps(data,indent=2)+"\n")
        np.savez_compressed(OUT/"trajectory.npz",time=[x[0] for x in self.log],phase=[x[1] for x in self.log],
            action=[x[2] for x in self.log],hub_pose=[x[3] for x in self.log],tripod_q=[x[4] for x in self.log],
            robot_q=[x[5] for x in self.log],finger_contact=[x[6] for x in self.log],
            foot_contact=[x[7] for x in self.log],robot_contact=[x[8] for x in self.log],
            hub_velocity=[x[9] for x in self.log],joint_velocity=[x[10] for x in self.log],
            arm_arm_contact=[x[11] for x in self.log],robot_velocity_target=[x[12] for x in self.log])
    def execute(self):
        from yam_arena.constants import DOWN_XYZW
        p,r = self.pose()
        down=Rotation.from_quat(DOWN_XYZW)*Rotation.from_euler("z",-90,degrees=True)
        grab = p+r.apply([0,0,.0372])+[0,0,.011]
        self.mark("approach_hub")
        self.grip(0,.012)
        self.move(0,grab+[0,0,.10],down,3)
        self.move(0,grab,down,3)
        self.mark("close_hub")
        self.grip(0,0.,3)
        h0,hr0 = self.hand(0)
        self.mark("lift_hub")
        self.move(0,h0+[0,0,.12],hr0,6)
        self.wait(1)
        self.mark("lifted")
        from yam_arena import station
        assert self.collision_cloud()[:,2].min()>station.TABLE_TOP+.005,"Object not clear of table"
        poses=np.array([x[3] for x in self.log[-30:]])
        assert np.linalg.norm(np.ptp(poses[:,:3],axis=0))<.005,"Unstable lifted grasp"
        contact=np.linalg.norm(np.array([x[6][0,:,0] for x in self.log[-30:]]),axis=-1).max(0)
        assert np.all(contact>.2),"Hub not held by both fingers"
        self.airborne=True
        self.wait(1/MOTION_SPEED)
        p,r = self.pose()
        hp,hr = self.hand(0)
        self.hold_point = r.inv().apply(hp-p)
        self.hold_rotation = r.inv()*hr
        self.mark("turn_upright")
        self.orient(np.array([.34,0,.96]),Rotation.identity(),6)
        self.mark("upright")
        self.open_leg(0)
        for leg in (1,2):
            self.move(1,*self.right_park,4)
            self.mark(f"present_leg_{leg}")
            self.present_leg(leg)
            self.open_leg(leg)
        self.place()
        self.save(True)

    def present_leg(self, leg):
        """Choose a holder pose with reachable right-hand approach and opening."""
        from scipy.optimize import least_squares
        from yam_arena import station
        from yam_arena.ik import tcp_pose, LOWER, UPPER
        p,r=self.pose()
        hp,hr=self.hand(0)
        hold_p,hold_r=r.inv().apply(hp-p),r.inv()*hr
        phi=np.deg2rad(-90+120*leg)
        radial=np.array([np.cos(phi),np.sin(phi),0.])
        tangent=np.cross([0,0,1],radial)
        initial_angle=float(self.tripod.data.joint_pos.torch[0,leg])
        def residual(v, regularize=True):
            center,rot=v[:3],Rotation.from_rotvec(v[3:6])
            targets=[(center+rot.apply(hold_p)+(rot*hold_r).apply([0,0,-.006]),rot*hold_r)]
            for theta,length,pre in ((initial_angle,.055,.06),(np.deg2rad(32),.055,.04)):
                direction=np.sin(theta)*radial-np.cos(theta)*np.array([0,0,1])
                z=-np.cos(theta)*radial-np.sin(theta)*np.array([0,0,1])
                rr=rot*Rotation.from_matrix(np.column_stack([tangent,np.cross(z,tangent),z]))
                rr=self.leg_tool_rotation(rr,leg)
                pp=center+rot.apply(.0078*radial+length*direction)
                pp+=rr.apply([0,0,-pre-.0105])
                targets.append((pp,rr))
            errors=[]
            for k,(pp,rr) in enumerate(targets):
                actual,ar=tcp_pose(v[6+6*k:12+6*k],station.BASES["left" if k==0 else "right"])
                errors.extend(100*(actual-pp))
                errors.extend(10*(rr.inv()*Rotation.from_matrix(ar)).as_rotvec())
            if regularize:
                errors.extend(.01*(center-[.30,0,.99]))
                errors.extend(.001*(rot.apply([0,0,1])-[0,0,1]))
            return errors
        low=np.r_[[.15,-.10,.95],[-6.3]*3,np.tile(LOWER+.08,3)]
        high=np.r_[[.40,.15,1.10],[6.3]*3,np.tile(UPPER-.08,3)]
        # Seed the two different wrist branches explicitly; no random IK restarts.
        angles,qs={
            1:((57,-30,-102),[[-.46,1.64,.68,1.53,1.53,2.05],
                [.41,1.80,2.00,-1.65,-.65,-2.05],[.42,1.71,1.91,-1.40,-.13,-1.94]]),
            2:((4,7,51),[[-1.22,2.05,2.00,-1.53,-1.35,-.02],
                [1.27,1.24,.49,.51,.90,.19],[1.28,1.26,.74,-.43,.65,.72]])}[leg]
        seed=np.r_[[.25,0,.99],Rotation.from_euler("xyz",angles,degrees=True).as_rotvec(),np.ravel(qs)]
        result=least_squares(residual,np.clip(seed,low+1e-5,high-1e-5),bounds=(low,high),max_nfev=400)
        error=np.abs(residual(result.x,False)).max()
        assert error<.05,("No bimanual presentation pose",leg,error)
        center,rot=result.x[:3],Rotation.from_rotvec(result.x[3:6])
        print("presentation",leg,center,rot.as_euler("xyz",degrees=True),flush=True)
        np.savez(OUT/f"presentation_{leg}.npz",position=center,quaternion=rot.as_quat(),joints=result.x[6:])
        self.move_joints(0,result.x[6:12],7)
        actual,actual_r=self.pose()
        assert np.linalg.norm(actual-center)<.01, ("Holder slipped",actual,center)
        assert (actual_r.inv()*rot).magnitude()<.15, "Holder rotated"

    def orient(self, point, rotation, seconds=4):
        p,r=self.pose()
        hp,hr=self.hand(0)
        self.hold_point=r.inv().apply(hp-p)
        self.hold_rotation=r.inv()*hr
        self.move(0,point+rotation.apply(self.hold_point),rotation*self.hold_rotation,seconds)
        p,r = self.pose()
        assert np.linalg.norm(p-point)<.01, ("Holder slipped",p,point)
        assert (r.inv()*rotation).magnitude()<.15, ("Holder rotated",r.as_quat())

    def leg_grasp(self, leg, theta, pose, length=.055):
        from model import PIVOT_RADIUS
        p,r = pose
        phi = np.deg2rad(-90+120*leg)
        radial = np.array([np.cos(phi),np.sin(phi),0.])
        tangent = np.cross([0,0,1],radial)
        direction = np.sin(theta)*radial-np.cos(theta)*np.array([0,0,1])
        approach = -np.cos(theta)*radial-np.sin(theta)*np.array([0,0,1])
        rot = r*Rotation.from_matrix(np.column_stack([tangent,np.cross(approach,tangent),approach]))
        rot=self.leg_tool_rotation(rot,leg)
        return p+r.apply(PIVOT_RADIUS*radial+length*direction),rot

    def measured_grasp(self, leg, length):
        index=self.tripod.body_names.index(f"leg_{leg}")
        pose=self.tripod.data.body_link_pose_w.torch[0,index].cpu().numpy()
        r=Rotation.from_quat(pose[3:])
        axes=r.as_matrix()
        rot=Rotation.from_matrix(np.column_stack([-axes[:,1],axes[:,2],-axes[:,0]]))
        return pose[:3]+r.apply([0,0,length]),self.leg_tool_rotation(rot,leg)

    @staticmethod
    def leg_tool_rotation(rotation,leg):
        # Approach the third tube from the front, clear of the holder's wrist.
        return rotation*Rotation.from_euler("y",-90 if leg==2 else 0,degrees=True)

    def open_leg(self, leg):
        theta=float(self.tripod.data.joint_pos.torch[0,leg])
        grasp=lambda:self.measured_grasp(leg,.055)
        point,rot = grasp()
        self.mark(f"approach_leg_{leg}")
        self.grip(1,.020)
        self.move(1,point-rot.apply([0,0,.06]),rot,4)
        self.move(1,*grasp(),3)
        self.grip(1,0.,2,follow=grasp)
        forces=np.linalg.norm(self.log[-1][6][1],axis=-1)
        assert np.all(forces[:,self.tripod.body_names.index(f"leg_{leg}")]>.2),"Outer tube not gripped"
        theta=float(self.tripod.data.joint_pos.torch[0,leg])
        opening_pose=self.pose()
        self.mark(f"open_leg_{leg}")
        for angle in np.linspace(theta,np.deg2rad(32),12):
            point,rot = self.leg_grasp(leg,angle,opening_pose)
            self.move(1,point,rot,1)
        # Bounded finger overtravel takes up compliance; the hinge stops at 32 degrees.
        for angle in (34,34,34):
            actual=float(self.tripod.data.joint_pos.torch[0,leg])
            if abs(np.rad2deg(actual)-32)<.3:
                break
            self.move(1,*self.leg_grasp(leg,np.deg2rad(angle),self.pose()),1)
        self.mark(f"opened_leg_{leg}")
        actual=float(self.tripod.data.joint_pos.torch[0,leg])
        assert abs(actual-np.deg2rad(32))<np.deg2rad(1), ("Leg failed to open",leg,actual)
        self.action[7:13]=self.robots[1].data.joint_pos.torch[0,:6].cpu().numpy()
        self.bias[1]=0
        self.grip(1,.025,2)
        point,rot=self.hand(1)
        self.move(1,point-rot.apply([0,0,.04]),rot,2)
        q=self.tripod.data.joint_pos.torch[0,:leg+1].cpu().numpy()
        assert np.abs(np.rad2deg(q)-32).max()<1,("Opened legs did not hold",q)

    def place(self):
        from model import ASSETS
        from yam_arena.constants import READY
        self.move(1,*self.right_park,4)
        point=np.array(self.info["goal_hub_position_m"])
        rotation=Rotation.from_quat(self.info["goal_hub_quaternion_xyzw"])
        self.mark("place_above_goal")
        self.orient(point+[0,0,.06],rotation,6)
        self.mark("lower_to_table")
        self.orient(point-[0,0,.0015],rotation,5)
        q=self.tripod.data.joint_pos.torch[0].cpu().numpy()
        assert np.abs(np.rad2deg(q)-32).max()<1, ("Legs retracted before release",q)
        self.mark("release_on_table")
        self.airborne=False
        self.action[:6]=self.robots[0].data.joint_pos.torch[0,:6].cpu().numpy()
        self.bias[0]=0
        self.grip(0,.006,2)
        hp,hr=self.hand(0)
        self.move(0,hp-hr.apply([0,0,.08]),hr,3)
        assert np.linalg.norm(self.log[-1][6])<.01,"Disk not released"
        self.move_joints(0,np.array(READY),4)
        self.grip(0,.0939)
        self.mark("settle_released")
        start=len(self.log)
        self.wait(5)
        p,r=self.pose()
        q=self.tripod.data.joint_pos.torch[0].cpu().numpy()
        poses=np.array([x[3] for x in self.log[start:]])
        assert np.abs(np.rad2deg(q)-32).max()<1, ("Legs not fully open",q)
        assert np.rad2deg(np.arccos(np.clip(r.apply([0,0,1])[2],-1,1)))<5
        assert np.linalg.norm(np.ptp(poses[:,:3],axis=0))<.002,"Unstable placement"
        support=np.array([x[7] for x in self.log if x[1] in ("release_on_table","settle_released")])
        assert np.all(np.linalg.norm(support,axis=-1).max(axis=(0,2))>.03),"Missing foot support"
        assert np.linalg.norm(self.log[-1][6])<.01,"Hands still touching tripod"
        view=self.physics.create_articulation_view("/World/envs/env_0/mini_tripod")
        cloud=self.collision_cloud()
        assert np.all(cloud[:,:2].min(0)>np.array(self.info["goal_low_xy_m"]))
        assert np.all(cloud[:,:2].max(0)<np.array(self.info["goal_high_xy_m"]))
        assert np.linalg.norm((cloud[:,:2].max(0)+cloud[:,:2].min(0))/2-self.info["goal_center_xy_m"])<.005
        assert np.all(view.get_dof_stiffnesses()==0) and np.all(view.get_dof_dampings()==0)
        assert np.all(view.get_dof_actuation_forces()==0)
        self.validation={"final_hinges_deg":np.rad2deg(q).tolist(),
            "settle_time_s":5,"drift_m":float(np.linalg.norm(np.ptp(poses[:,:3],axis=0))),
            "tilt_deg":float(np.rad2deg(np.arccos(np.clip(r.apply([0,0,1])[2],-1,1)))),
            "foot_support_peak_N":np.linalg.norm(support,axis=-1).max(axis=(0,2)).tolist(),
            "hand_contact_force_N":float(np.linalg.norm(self.log[-1][6])),
            "tape_clearances_m":np.r_[cloud[:,:2].min(0)-self.info["goal_low_xy_m"],
                self.info["goal_high_xy_m"]-cloud[:,:2].max(0)].tolist(),
            "passive_free_base":not view.shared_metatype.fixed_base,"no_object_drives":True}
        self.mark("success")


def run():
    import yam_scene
    yam_scene.OUT = OUT
    env,info = yam_scene.make_env(enable_cameras=False,control_rate=CONTROL_RATE)
    env.reset()
    expert = Expert(env,info)
    try:
        expert.execute()
    except Exception as e:
        expert.save(False,str(e))
        raise
    finally:
        env.close()


if __name__=="__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
    else:
        app.close()

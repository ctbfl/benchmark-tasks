"""Shared physical execution, spring latch, measurement and recording."""
import hashlib
import json
import shutil
import numpy as np
from config import (HERE,CONFIG,DT,HZ,SUBSTEPS,MAX_STEPS,OPEN_ANGLE,
                    BUTTON_FORCE,CLOSE_FORCE)
from scoring import Score
CAMERAS=('left_cam','top_cam','right_cam')


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def snapshot(out):
    files=[*HERE.glob('*.py'),HERE/'dream-defaults.json',*sorted((HERE/'assets').glob('*'))]
    if (HERE/'demo_actions.npy').exists(): files.append(HERE/'demo_actions.npy')
    for path in files:
        target=out/'source'/path.relative_to(HERE)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write_json(out/'source-sha256.json',{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


class Experiment:
    def __init__(self,out,cameras=False,screenshot=False):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY
        from yam_arena.ik import LOWER,UPPER
        from yam_arena.robot import asset_link_paths
        from scene import make_env
        snapshot(out)
        self.out,self.torch,self.cameras=out,torch,cameras
        self.env=make_env(out,cameras,screenshot)
        self.base=self.env.unwrapped
        self.env.reset(seed=CONFIG['episode']['seed'])
        self.binder=self.base.scene['binder']
        self.card,self.block=[self.base.scene[n] for n in ('card','block')]
        self.robots=[self.base.scene[s+'_robot'] for s in ('left','right')]
        self.ready=np.array([*READY,1.,*READY,1.])
        self.low,self.high=np.r_[LOWER,0.,LOWER,0.],np.r_[UPPER,1.,UPPER,1.]
        self.jids=[self.binder.joint_names.index(n) for n in ('left_hinge','right_hinge','press')]
        self.rail_id=self.binder.body_names.index('right_rail')
        assert not self.binder.actuators and not self.binder.is_fixed_base
        assert abs(self.base.physics_dt-DT)<1e-12
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing',False)
        self.physics=tensors.create_simulation_view('numpy',stage_id=get_current_stage_id())
        self.physics.set_subspace_roots('/')
        self.view=self.physics.create_articulation_view('/World/envs/env_0/binder')
        paths=asset_link_paths()
        fingers=[f'/World/envs/env_0/{side}Arm{paths[name]}' for side in ('Left','Right') for name in ('tip_left','tip_right')]
        bodies=[f'/World/envs/env_0/binder/{name}' for name in ('base','left_rail','right_rail','button')]+['/World/envs/env_0/card/body']
        self.contacts=self.physics.create_rigid_contact_view(bodies,[fingers]*5)
        self.open_latch=False
        self.events=[]
        self.score,self.steps,self.ticks=Score(),0,0
        self.rows,self.actions,self.measures=[],[],[]
        self.force_matrix=np.zeros((5,4,3))
        self.targets=self.view.get_dof_position_targets().copy()
        self.set_latch(False)
        for _ in range(HZ): self.advance(self.ready,record=False)
        self.ticks=0
        card_pose=self.card.data.root_pose_w.torch[0].cpu().numpy()
        block_pose=self.block.data.root_pose_w.torch[0].cpu().numpy()
        assert abs(card_pose[2]-block_pose[2]-.1008)<.001
        assert np.linalg.norm(card_pose[3:5])<.005
        self.save_state()
        self.measure()
        assert self.score.result()['score']==0.
        if cameras:
            for _ in range(4): self.capture()

    def angles(self):
        return self.view.get_dof_positions()[0,self.jids].copy()

    def set_latch(self,opened):
        self.open_latch=opened
        self.targets[0,self.jids[0]]=-OPEN_ANGLE if opened else 0.
        self.targets[0,self.jids[1]]=OPEN_ANGLE if opened else 0.
        self.view.set_dof_position_targets(self.targets,np.array([0],dtype=np.uint32))

    def latch(self,forces):
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(self.view.get_root_transforms()[0,3:]).as_matrix()
        f=forces.sum(axis=1)@rotation
        q=self.angles()
        if not self.open_latch and q[2]<-.0025 and -f[3,2]>BUTTON_FORCE:
            self.set_latch(True)
            self.events.append(dict(time_s=self.ticks*DT,event='open',button_force_N=float(-f[3,2])))
        elif self.open_latch and -f[1,1]>CLOSE_FORCE and f[2,1]>CLOSE_FORCE and max(abs(q[:2]))<.53:
            self.set_latch(False)
            self.events.append(dict(time_s=self.ticks*DT,event='close',forces_N=[float(-f[1,1]),float(f[2,1])]))

    def save_state(self):
        array=lambda data:data.torch[0].cpu().numpy().ravel()
        row=np.r_[self.ticks*DT,*[array(r.data.joint_pos) for r in self.robots],
            *[array(r.data.joint_vel) for r in self.robots],
            self.view.get_root_transforms()[0],self.view.get_root_velocities()[0],
            self.view.get_dof_positions()[0],self.view.get_dof_velocities()[0],
            array(self.card.data.root_pose_w),array(self.card.data.root_vel_w),
            array(self.block.data.root_pose_w),array(self.block.data.root_vel_w),float(self.open_latch)]
        assert row.shape==(79,) and np.isfinite(row).all()
        self.rows.append(row)

    def advance(self,action,record=True):
        self.base.action_manager.process_action(self.torch.tensor(action[None],dtype=self.torch.float32))
        force_norms=np.zeros((5,4))
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            # Only the prop's physically triggered latch changes its spring rest angles.
            self.view.set_dof_position_targets(self.targets,np.array([0],dtype=np.uint32))
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self.ticks+=1
            self.force_matrix=self.contacts.get_contact_force_matrix(DT).copy()
            force_norms=np.maximum(force_norms,np.linalg.norm(self.force_matrix,axis=-1))
            self.latch(self.force_matrix)
            if record: self.save_state()
        return force_norms

    def step(self,action):
        assert self.steps<MAX_STEPS
        action=np.asarray(action,dtype=float)
        assert action.shape==(14,) and np.isfinite(action).all()
        applied=np.clip(action,self.low,self.high)
        self.actions.append(applied.copy())
        forces=self.advance(applied)
        self.steps+=1
        result=self.measure(float(forces[-1].sum()))
        self.measures.append(dict(**result,angles=self.angles().tolist(),finger_force_N=forces.tolist()))

    def measure(self,card_force=0.):
        return self.score.update(self.ticks*DT,self.angles(),
            self.binder.data.body_link_pose_w.torch[0,self.rail_id].cpu().numpy(),
            self.card.data.root_pose_w.torch[0].cpu().numpy(),
            self.view.get_root_transforms()[0],card_force)

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images={}
        for name in CAMERAS:
            camera=self.base.scene[name]
            camera.update(1/HZ,force_recompute=True)
            images[name]=camera.data.output['rgb'].torch[0,...,:3].cpu().numpy().copy()
        return images

    def save(self):
        np.save(self.out/'states.npy',np.asarray(self.rows))
        np.save(self.out/'actions.npy',np.asarray(self.actions))
        write_json(self.out/'measurements.json',self.measures)
        write_json(self.out/'metrics.json',dict(**self.score.result(),control_steps=self.steps,latch_events=self.events))
        write_json(self.out/'schema.json',dict(columns=79,physics_hz=1/DT,quaternion_order='xyzw',
            binder_joint_names=self.binder.joint_names,binder_body_names=self.binder.body_names,
            robot_joint_names=[r.joint_names for r in self.robots],fields=dict(time=[0,1],
                robot_q=[1,17],robot_dq=[17,33],binder_root=[33,40],binder_velocity=[40,46],
                binder_joints=[46,49],binder_joint_velocity=[49,52],card=[52,59],
                card_velocity=[59,65],block=[65,72],block_velocity=[72,78],open_latch=[78,79])))

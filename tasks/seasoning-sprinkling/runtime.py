"""One environment and native robot action interface for all physical trials."""
import hashlib
import json
import shutil
import numpy as np
from config import HERE, CONFIG, DT, HZ, PHYSICS_HZ, SUBSTEPS, MAX_STEPS, GRAIN_COUNT
from scoring import score
CAMERAS = ('left_cam','top_cam','right_cam')


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
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY
        from yam_arena.ik import LOWER,UPPER
        from scene import make_env
        snapshot(out)
        self.out,self.torch,self.cameras=out,torch,cameras
        self.env=make_env(out,cameras,screenshot)
        self.base=self.env.unwrapped
        self.env.reset(seed=CONFIG['episode']['seed'])
        self.bottle=self.base.scene['bottle']
        self.plate,self.steak=[self.base.scene[n] for n in ('plate','steak')]
        self.robots=[self.base.scene[s+'_robot'] for s in ('left','right')]
        self.ready=np.array([*READY,1.,*READY,1.])
        self.low,self.high=np.r_[LOWER,0.,LOWER,0.],np.r_[UPPER,1.,UPPER,1.]
        self.jids=[self.bottle.joint_names.index(n) for n in ('pour_hinge','sprinkle_hinge')]
        assert not self.bottle.actuators and not self.bottle.is_fixed_base
        assert abs(self.base.physics_dt-DT)<1e-12
        self.physics=tensors.create_simulation_view('numpy',stage_id=get_current_stage_id())
        self.physics.set_subspace_roots('/')
        self.view=self.physics.create_articulation_view('/World/envs/env_0/bottle')
        self.grains=self.physics.create_rigid_body_view('/World/envs/env_0/grains/grain_*')
        assert self.grains.count==GRAIN_COUNT
        from pxr import PhysxSchema
        actual_ccd=PhysxSchema.PhysxSceneAPI(self.base.sim.stage.GetPrimAtPath('/physicsScene')).GetEnableCCDAttr().Get()
        assert actual_ccd==CONFIG['simulation']['enable_ccd']
        assert np.isclose(float(self.grains.get_masses().sum())*1000,10.,atol=1e-5)
        write_json(out/'physics-check.json',dict(scene_ccd=actual_ccd,grain_count=self.grains.count,
            total_grain_mass_g=float(self.grains.get_masses().sum())*1000,physics_hz=PHYSICS_HZ,control_hz=HZ))
        self.steps,self.ticks=0,0
        self.rows,self.grain_rows,self.actions,self.measures=[],[],[],[]
        for _ in range(2*HZ): self.advance(self.ready,record=False)
        self.ticks=0
        self.save_state()
        self.measure()
        if cameras:
            for _ in range(4): self.capture()

    def angles(self):
        return self.view.get_dof_positions()[0,self.jids].copy()

    def grain_state(self):
        return np.c_[self.grains.get_transforms(),self.grains.get_velocities()]

    @staticmethod
    def rigid_state(obj):
        return np.r_[obj.data.root_pose_w.torch[0].cpu().numpy(),obj.data.root_vel_w.torch[0].cpu().numpy()]

    def save_state(self):
        array=lambda data:data.torch[0].cpu().numpy().ravel()
        row=np.r_[self.ticks*DT,*[array(r.data.joint_pos) for r in self.robots],
            *[array(r.data.joint_vel) for r in self.robots],
            self.view.get_root_transforms()[0],self.view.get_root_velocities()[0],
            self.view.get_dof_positions()[0],self.view.get_dof_velocities()[0],
            self.rigid_state(self.plate),self.rigid_state(self.steak)]
        assert row.shape==(76,) and np.isfinite(row).all()
        self.rows.append(row.astype(np.float32))
        self.grain_rows.append(self.grain_state().astype(np.float32))

    def advance(self,action,record=True):
        self.base.action_manager.process_action(self.torch.tensor(action[None],dtype=self.torch.float32))
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self.ticks+=1
            if record and self.ticks % (PHYSICS_HZ//30)==0: self.save_state()

    def step(self,action):
        assert self.steps<MAX_STEPS
        action=np.asarray(action,dtype=float)
        assert action.shape==(14,) and np.isfinite(action).all()
        applied=np.clip(action,self.low,self.high)
        self.actions.append(applied.copy())
        self.advance(applied)
        self.steps+=1
        self.measure()
        self.measures.append(dict(**self.result,angles=self.angles().tolist(),time_s=self.steps/HZ))

    def measure(self):
        self.result=score(self.grain_state(),self.rigid_state(self.steak),self.view.get_root_transforms()[0])
        return self.result

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
        np.savez(self.out/'final-state.npz',grains=self.grain_state(),steak=self.rigid_state(self.steak),
                 bottle=self.view.get_root_transforms()[0])
        np.save(self.out/'states.npy',np.asarray(self.rows))
        np.save(self.out/'grains.npy',np.asarray(self.grain_rows))
        np.save(self.out/'actions.npy',np.asarray(self.actions))
        write_json(self.out/'measurements.json',self.measures)
        write_json(self.out/'metrics.json',dict(**self.result,control_steps=self.steps,time_s=self.steps/HZ))
        write_json(self.out/'schema.json',dict(columns=76,sample_hz=30,physics_hz=PHYSICS_HZ,
            quaternion_order='xyzw',bottle_joint_names=self.bottle.joint_names,
            grain_paths=self.grains.prim_paths,fields=dict(time=[0,1],robot_q=[1,17],robot_dq=[17,33],
                bottle_root=[33,40],bottle_velocity=[40,46],bottle_joints=[46,48],bottle_joint_velocity=[48,50],
                plate=[50,63],steak=[63,76])))

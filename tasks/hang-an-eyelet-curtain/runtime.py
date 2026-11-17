"""Shared physical execution, measurement and recording for development, replay and Dream."""
import hashlib
import json
import shutil
import numpy as np
from config import HERE, CONFIG, DT, HZ, SUBSTEPS, MAX_STEPS, RINGS
from scoring import evaluate

CAMERAS = ('left_cam', 'top_cam', 'right_cam')
CURTAIN = '/World/envs/env_0/curtain'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def snapshot(out):
    files = [*HERE.glob('*.py'), HERE/'dream-defaults.json', *sorted((HERE/'assets').glob('*'))]
    if (HERE/'demo_actions.npy').exists():
        files.append(HERE/'demo_actions.npy')
    for path in files:
        target = out/'source'/path.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(out/'source-sha256.json',
               {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


class Experiment:
    def __init__(self, out, cameras=False, screenshot=False, settle_s=1.):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY
        from yam_arena.ik import LOWER, UPPER
        from yam_arena.robot import asset_link_paths
        from scene import make_env
        from assets import curtain_topology
        snapshot(out)
        self.out, self.torch, self.cameras = out, torch, cameras
        self.env = make_env(out, cameras, screenshot)
        self.base = self.env.unwrapped
        self.env.reset(seed=CONFIG['episode']['seed'])
        self.rod, self.stand = self.base.scene['rod'], self.base.scene['stand']
        self.robots = [self.base.scene[s+'_robot'] for s in ('left', 'right')]
        self.ready = np.array([*READY, 1., *READY, 1.])
        self.low, self.high = np.r_[LOWER, 0., LOWER, 0.], np.r_[UPPER, 1., UPPER, 1.]
        assert abs(self.base.physics_dt-DT) < 1e-12
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
        self.physics = tensors.create_simulation_view('torch', stage_id=get_current_stage_id())
        self.physics.set_subspace_roots('/')
        self.rings = self.physics.create_rigid_body_view(CURTAIN+'/ring_*')
        self.cloth = self.physics.create_surface_deformable_body_view(CURTAIN+'/Cloth')
        self.links = [f'ring_{k}' for k in range(RINGS)]
        assert self.rings.prim_paths == [CURTAIN+'/'+name for name in self.links]
        self.rest, self.faces, self.edges, self.holes = curtain_topology()
        assert self.cloth.count == 1 and self.cloth.max_simulation_nodes_per_body == len(self.rest)
        assert np.array_equal(self.cloth.get_simulation_element_indices()[0].cpu().numpy(), self.faces)
        paths = asset_link_paths()
        self.finger_paths = [f'/World/envs/env_0/{side}Arm{paths[name]}'
                             for side in ('Left', 'Right') for name in ('tip_left', 'tip_right')]
        self.contact_targets = [CURTAIN+'/'+name for name in self.links]+['/World/envs/env_0/rod/body']
        self.contacts = self.physics.create_rigid_contact_view(self.finger_paths,
            [self.contact_targets]*len(self.finger_paths))
        self.support_contacts = self.physics.create_rigid_contact_view('/World/envs/env_0/rod/body',
            ['/World/envs/env_0/stand/body'])
        self.steps, self.ticks = 0, 0
        self.rows, self.actions, self.measures = [], [], []
        self.cloth_frames, self.cloth_times, self.cloth_stats = [], [], []
        self.contact_frames = []
        self.support_forces = []
        for _ in range(round(settle_s*HZ)):
            self.advance(self.ready, record=False)
        self.ticks = 0
        self.save_state()
        self.last = self.measure()
        self.observer = None
        if cameras:
            self.observer = self.add_observer((.80, -.42, 1.10), (.33, -.06, .84), (960, 540))
            for _ in range(4):
                self.capture()

    def pose(self, asset):
        return asset.data.root_pose_w.torch[0].cpu().numpy().copy()

    def link_poses(self):
        return self.rings.get_transforms().cpu().numpy().copy()

    def ring_poses(self):
        return self.link_poses()

    def cloth_positions(self):
        return self.cloth.get_simulation_nodal_positions()[0].cpu().numpy().copy()

    def save_state(self):
        array = lambda data: data.torch[0].cpu().numpy().ravel()
        row = np.r_[np.array([self.ticks*DT], dtype=np.float64), *[array(r.data.joint_pos) for r in self.robots],
                    *[array(r.data.joint_vel) for r in self.robots],
                    self.pose(self.rod), array(self.rod.data.root_vel_w),
                    self.pose(self.stand), array(self.stand.data.root_vel_w),
                    self.ring_poses().ravel()]
        if not np.isfinite(row).all():
            bad = np.where(~np.isfinite(row))[0]
            links = sorted({self.links[(i-59)//7] for i in bad if 59 <= i < 59+7*len(self.links)})
            raise FloatingPointError(f'non-finite state at t={self.ticks*DT:.4f}s: columns {bad[:12].tolist()} links {links}')
        self.rows.append(row)
        if self.ticks % round(1/(30*DT)) == 0:
            from scipy.spatial.transform import Rotation
            points = self.cloth_positions()
            assert np.isfinite(points).all(), f'Nonfinite cloth at {self.ticks*DT}'
            length = np.linalg.norm(points[self.edges[:, 1]]-points[self.edges[:, 0]], axis=1)
            rest_length = np.linalg.norm(self.rest[self.edges[:, 1]]-self.rest[self.edges[:, 0]], axis=1)
            poses = self.ring_poses()
            centers = self.rest[self.holes].mean(axis=1)
            attached = np.stack([Rotation.from_quat(p[3:]).apply(self.rest[h]-c)+p[:3]
                                 for p, h, c in zip(poses, self.holes, centers)])
            error = float(np.linalg.norm(points[self.holes]-attached, axis=2).max())
            stretch = float((length/rest_length).max())
            assert stretch < 2. and error < .03, (self.ticks*DT, stretch, error)
            self.cloth_frames.append(points)
            self.cloth_times.append(self.ticks*DT)
            self.cloth_stats.append([self.ticks*DT, stretch, error, points[:, 2].min()])

    def advance(self, action, record=True):
        self.base.action_manager.process_action(self.torch.tensor(action[None], dtype=self.torch.float32,
                                                                  device=self.base.device))
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self.ticks += 1
            if record:
                self.save_state()

    def step(self, action):
        assert self.steps < MAX_STEPS
        action = np.asarray(action, dtype=float)
        assert action.shape == (14,) and np.isfinite(action).all()
        applied = np.clip(action, self.low, self.high)
        self.actions.append(applied.copy())
        self.advance(applied)
        self.steps += 1
        self.last = self.measure()
        self.measures.append(self.last)
        self.contact_frames.append(self.contacts.get_contact_force_matrix(DT).cpu().numpy().copy())
        self.support_forces.append(self.support_contacts.get_contact_force_matrix(DT)[0, 0].cpu().numpy().copy())

    def measure(self):
        result = evaluate(self.ring_poses(), self.pose(self.rod), self.pose(self.stand))
        result['time_s'] = self.steps/HZ
        return result

    def result(self):
        return dict(self.last, control_steps=self.steps)

    def add_observer(self, eye, target, size, focal=24.):
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom
        camera = UsdGeom.Camera.Define(self.base.sim.stage, '/DevObserver')
        camera.GetPrim().CreateAttribute('omni:scenePartition', Sdf.ValueTypeNames.Token).Set('env_0')
        transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
        camera.AddTransformOp().Set(transform)
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(36.*size[1]/size[0])
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
        product = rep.create.render_product(str(camera.GetPath()), size)
        rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb.attach([product])
        return rgb

    def capture(self):
        from assets import display_cloth
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        display_cloth(self.base.sim.stage, CURTAIN, self.cloth_positions(), self.faces)
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in CAMERAS:
            camera = self.base.scene[name]
            camera.update(1/HZ, force_recompute=True)
            images[name] = camera.data.output['rgb'].torch[0, ..., :3].cpu().numpy().copy()
        if self.observer is not None:
            data = self.observer.get_data()
            if data is not None and len(data):
                images['observer'] = np.ascontiguousarray(data[..., :3])
        return images

    def save(self):
        np.save(self.out/'states.npy', np.asarray(self.rows))
        np.save(self.out/'actions.npy', np.asarray(self.actions))
        np.save(self.out/'contacts.npy', np.asarray(self.contact_frames))
        np.save(self.out/'support-forces.npy', np.asarray(self.support_forces))
        write_json(self.out/'contact-schema.json', dict(sensors=self.finger_paths, targets=self.contact_targets))
        np.savez_compressed(self.out/'cloth.npz', positions=np.asarray(self.cloth_frames),
            times=np.asarray(self.cloth_times), stats=np.asarray(self.cloth_stats),
            rest=self.rest, faces=self.faces, edges=self.edges, holes=self.holes)
        write_json(self.out/'measurements.json', self.measures)
        write_json(self.out/'metrics.json', self.result())
        write_json(self.out/'schema.json', dict(physics_hz=1/DT, quaternion_order='xyzw',
            rings=self.links, cloth_file='cloth.npz', robot_joint_names=[r.joint_names for r in self.robots],
            fields=dict(time=[0, 1], robot_q=[1, 17], robot_dq=[17, 33], rod=[33, 40],
                        rod_velocity=[40, 46], stand=[46, 53], stand_velocity=[53, 59],
                        rings=[59, 59+7*RINGS])))

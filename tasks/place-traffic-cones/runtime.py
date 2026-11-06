"""Shared physical stepping and independent measurements; no task feedback to replay."""
import numpy as np
from config import CONFIG, DT, CONTROL_HZ, SUBSTEPS, MAX_STEPS


class Runtime:
    def __init__(self, *, cameras=False, log_dir=None):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena import station
        from yam_arena.constants import READY
        from yam_arena.robot import asset_link_paths
        from scene import make_env

        self.torch = torch
        self.env = make_env(cameras=cameras, log_dir=log_dir)
        self.base = self.env.unwrapped
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
        self.env.reset(seed=CONFIG['episode']['seed'])
        self.cones = [self.base.scene[f'cone_{i}'] for i in range(6)]
        self.robots = [self.base.scene[name] for name in ('left_robot', 'right_robot')]
        self.physics = tensors.create_simulation_view('numpy', stage_id=get_current_stage_id())
        self.physics.set_subspace_roots('/')
        self.views = [self.physics.create_articulation_view(f'/World/envs/env_0/cone_{i}/body') for i in range(6)]
        robot_paths = [f'/World/envs/env_0/{side}Arm{path}' for side in ('Left', 'Right')
                       for path in asset_link_paths().values()]
        self.contacts = [self.physics.create_rigid_contact_view(
            [f'/World/envs/env_0/cone_{i}/{name}' for name in cone.body_names],
            [robot_paths] * cone.num_bodies) for i, cone in enumerate(self.cones)]
        self.support = [self.physics.create_rigid_contact_view(f'/World/envs/env_0/cone_{i}/body',
            ['/World/envs/env_0/table/geometry/mesh', '/World/envs/env_0/board/board']) for i in range(6)]
        self.table_top = station.TABLE_TOP
        self.ready = np.array([*READY, 1., *READY, 1.])
        self.steps = self.ticks = 0
        self.frames = []
        self.measurements = []
        self.actions = []
        self.robot_contact = np.zeros(6)
        self.link_contact = np.zeros((6,5))
        for _ in range(40):
            self.step(self.ready, record=False)
        self.steps = self.ticks = 0

    def step(self, action, *, record=True):
        assert self.steps < MAX_STEPS
        action = np.asarray(action, dtype=float)
        assert action.shape == (14,) and np.isfinite(action).all()
        self.base.action_manager.process_action(self.torch.tensor(action[None], dtype=self.torch.float32))
        self.robot_contact[:] = 0.
        self.support_force = np.zeros(6)
        self.link_contact[:] = 0.
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self.ticks += 1
            forces = np.array([np.linalg.norm(view.get_contact_force_matrix(DT), axis=-1).sum(axis=-1)
                               for view in self.contacts])
            self.link_contact = np.maximum(self.link_contact, forces)
            self.robot_contact = np.maximum(self.robot_contact, forces.sum(axis=-1))
            self.support_force = np.maximum(self.support_force, [np.linalg.norm(
                view.get_contact_force_matrix(DT), axis=-1).sum() for view in self.support])
            if record and self.ticks % 8 == 0:
                self.frames.append(self.snapshot())
        self.steps += 1
        if record:
            self.actions.append(action.copy())
            self.measurements.append(self.measure())
        return self.measure()

    def roots(self):
        return np.array([view.get_root_transforms()[0] for view in self.views])

    def measure(self):
        from scoring import measure
        result = measure(self.roots(), self.robot_contact, self.table_top, self.steps/CONTROL_HZ)
        result['support_N'] = self.support_force.tolist()
        result['roots'] = self.roots().tolist()
        result['robot_contact_by_link_N'] = self.link_contact.tolist()
        result['body_names'] = self.cones[0].body_names
        result['tip_compression_m'] = [view.get_dof_positions()[0].tolist() for view in self.views]
        return result

    def snapshot(self):
        return np.concatenate(([self.ticks*DT],
            *[robot.data.joint_pos.torch[0].cpu().numpy() for robot in self.robots],
            *[np.concatenate((view.get_root_transforms()[0], view.get_dof_positions()[0])) for view in self.views]))

    def save(self, out):
        import json
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/'trajectory.npz', frames=np.array(self.frames), actions=np.array(self.actions))
        (out/'measurements.json').write_text(json.dumps(self.measurements, allow_nan=False))
        (out/'score.json').write_text(json.dumps(self.measure(), indent=2, allow_nan=False)+'\n')

    def close(self):
        self.env.close()

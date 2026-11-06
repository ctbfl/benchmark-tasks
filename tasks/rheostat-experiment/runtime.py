"""Shared physical stepping and recording; no task-state inputs to the model."""
import json
import hashlib
import shutil
import numpy as np

from circuit import CONFIG, HERE, HZ, PHYSICS_HZ, MAX_STEPS, Score, voltage, intact

CAMERAS = ("left_cam", "top_cam", "right_cam")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(out):
    source = out/"source"
    source.mkdir()
    files = [*HERE.glob("*.py"), HERE/"dream-defaults.json", *sorted((HERE/"assets").glob("*"))]
    if (HERE/"demo_actions.npy").exists():
        files.append(HERE/"demo_actions.npy")
    for path in files:
        target = source/path.relative_to(HERE)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write_json(out/"source-sha256.json",{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


class Experiment:
    def __init__(self, out, cameras=True, screenshot=False):
        import carb
        import torch
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.constants import READY
        from yam_arena.ik import LOWER, UPPER
        from yam_arena.robot import asset_link_paths
        from assets import Visuals
        from scene import make_env

        snapshot(out)
        self.out, self.torch, self.cameras = out, torch, cameras
        self.env = make_env(out, cameras, screenshot)
        self.base = self.env.unwrapped
        self.env.reset(seed=CONFIG["episode"]["seed"])
        self.rheostat = self.base.scene["rheostat"]
        self.robots = [self.base.scene[s+"_robot"] for s in ("left", "right")]
        self.visuals = Visuals(self.base.sim.stage)
        self.ready = np.array([*READY,1.,*READY,1.])
        self.low, self.high = np.r_[LOWER,0.,LOWER,0.], np.r_[UPPER,1.,UPPER,1.]
        assert not self.rheostat.actuators and not self.rheostat.is_fixed_base
        assert self.rheostat.joint_names == ["slide"]
        assert abs(self.base.physics_dt - 1/PHYSICS_HZ) < 1e-10
        assert abs(self.base.step_dt - 1/HZ) < 1e-10
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        self.view = self.physics.create_articulation_view("/World/envs/env_0/rheostat")
        assert self.view.count == 1 and np.all(self.view.get_dof_stiffnesses() == 0)
        paths = asset_link_paths()
        fingers = [f"/World/envs/env_0/{side}Arm{paths[name]}"
                   for side in ("Left","Right") for name in ("tip_left","tip_right")]
        self.contacts = self.physics.create_rigid_contact_view(
            ["/World/envs/env_0/rheostat/base","/World/envs/env_0/rheostat/slider"], [fingers]*2)
        action = torch.tensor(self.ready[None], dtype=torch.float32)
        for _ in range(HZ):
            self.env.step(action)
        self.score, self.steps, self.rows, self.actions, self.measures = Score(), 0, [], [], []
        self.save_state()
        self.initial_root = self.rows[0][33:40].copy()
        self.visuals.update(voltage(self.displacement()))
        assert abs(voltage(self.displacement()) - 1.) < .01
        if cameras:
            for _ in range(4):
                self.capture()

    def displacement(self):
        return float(self.view.get_dof_positions()[0, 0])

    def save_state(self):
        array = lambda data: data.torch[0].cpu().numpy().ravel()
        self.rows.append(np.concatenate(([self.score.ticks/PHYSICS_HZ],
            *[array(r.data.joint_pos) for r in self.robots],
            *[array(r.data.joint_vel) for r in self.robots],
            self.view.get_root_transforms()[0], self.view.get_root_velocities()[0],
            self.view.get_dof_positions()[0], self.view.get_dof_velocities()[0])))
        assert len(self.rows[-1]) == 48 and np.isfinite(self.rows[-1]).all()

    def step(self, action, capture=False):
        assert self.steps < MAX_STEPS
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected 14 finite absolute joint/gripper targets")
        applied = np.clip(action, self.low, self.high)
        self.actions.append(applied.copy())
        self.base.action_manager.process_action(self.torch.tensor(applied[None], dtype=self.torch.float32))
        forces = np.zeros((2,4))
        for _ in range(PHYSICS_HZ // HZ):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(1/PHYSICS_HZ)
            self.score.update(self.displacement(), intact(self.view.get_root_transforms()[0],self.initial_root))
            self.save_state()
            forces = np.maximum(forces, np.linalg.norm(self.contacts.get_contact_force_matrix(1/PHYSICS_HZ),axis=-1))
        self.steps += 1
        self.measures.append(dict(**self.score.result(), slider_m=self.displacement(), finger_force_N=forces.tolist()))
        if capture:
            return self.capture()

    def capture(self):
        import carb
        import carb.eventdispatcher
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self.visuals.update(voltage(self.displacement()))
        carb.settings.get_settings().set_bool("/rtx-transient/post/dlss/forceParamReset",True)
        carb.eventdispatcher.get_eventdispatcher().dispatch_event(event_name="omni.rtx.ForceResetAccumulation")
        tick = self.base.sim._physics_step_count
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in CAMERAS:
            camera = self.base.scene[name]
            camera.update(1/HZ, force_recompute=True)
            images[name] = camera.data.output["rgb"].torch[0,...,:3].cpu().numpy().copy()
        assert self.base.sim._physics_step_count == tick
        return images

    def save(self):
        np.save(self.out/"states.npy", np.asarray(self.rows))
        np.save(self.out/"actions.npy", np.asarray(self.actions))
        write_json(self.out/"measurements.json", self.measures)
        write_json(self.out/"metrics.json", dict(**self.score.result(), control_steps=self.steps))
        write_json(self.out/"schema.json", dict(columns=48, dtype="float64", physics_hz=PHYSICS_HZ,
            fields=dict(time=[0,1],robot_q=[1,17],robot_dq=[17,33],root=[33,40],
                        root_velocity=[40,46],slider=[46,47],slider_velocity=[47,48]),
            robot_joint_names=[r.joint_names for r in self.robots], quaternion_order="xyzw"))

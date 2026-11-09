"""Shared native stepping and measured execution archives for both policies."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN, READY
from yam_arena.ik import LOWER, UPPER

from config import CONFIG, HERE, ROOT, DT, SUBSTEPS
from scene import PLACEMENTS, make_env
from scoring import Scorer
from visuals import update, prepare_render


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def archive_source(out):
    source = out / "source"
    source.mkdir()
    for path in [*HERE.glob("*.py"), HERE / "dream-defaults.json", *HERE.glob("demo_actions.npy")]:
        shutil.copy2(path, source / path.name)
    shutil.copytree(HERE / "assets", source / "assets")
    write_json(out / "source_manifest.json", {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*") if p.is_file()})


class Simulation:
    def __init__(self, out, cameras=True, screenshot=False):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = make_env(self.out, cameras, screenshot)
        self.base = self.env.unwrapped
        self.env.reset(seed=CONFIG["episode"]["seed"])
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        for _ in range(40):
            self.env.step(torch.tensor(self.action[None]))
        self.robots = [self.base.scene[side + "_robot"] for side in ("left", "right")]
        self.objects = {name: self.base.scene[name] for name, *_ in PLACEMENTS}
        self.ids = [(r.find_joints(ARM_JOINTS, preserve_order=True)[0],
                     r.find_joints(FINGER_JOINTS, preserve_order=True)[0]) for r in self.robots]
        self.low, self.high = np.r_[LOWER, 0., LOWER, 0.], np.r_[UPPER, 1., UPPER, 1.]
        self.control_dt = 1 / CONFIG["simulation"]["control_hz"]
        self.dt = 1 / CONFIG["simulation"]["physics_hz"]
        assert abs(self.base.physics_dt - self.dt) < 1e-10
        assert abs(self.base.step_dt - self.control_dt) < 1e-10
        import carb
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.robot import asset_link_paths
        carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
        self.physics = tensors.create_simulation_view('numpy', stage_id=get_current_stage_id())
        self.physics.set_subspace_roots('/')
        paths = asset_link_paths()
        fingers = [ROOT+f'/{side}Arm'+paths[tip] for side in ('Left','Right')
                   for tip in ('gripper','tip_left','tip_right')]
        self.contact = self.physics.create_rigid_contact_view(ROOT+'/steak', [ROOT+'/pan',ROOT+'/spatula',*fingers])
        self.scorer = Scorer(self.object_state('steak'), self.object_state('pan'))
        self.runtime = self
        self.forces = np.zeros((8,3))
        self.steps = self.ticks = 0
        self.phase = "initial"
        self.records, self.commands, self.frames = [], [], []
        self.schema = {}
        update(self.base.sim.stage, 0, False)
        self._record()
        write_json(self.out / "schema.json", self.schema)

    @property
    def time(self):
        return self.ticks * self.dt

    @property
    def score(self):
        return self.scorer.result()['score']

    def state(self):
        return self.scorer.result()

    def object_state(self, name):
        obj = self.objects[name]
        return np.r_[obj.data.root_pose_w.torch[0].cpu().numpy(),
                     obj.data.root_vel_w.torch[0].cpu().numpy()]

    def measured_action(self):
        parts = []
        for robot, (arm, fingers) in zip(self.robots, self.ids):
            q = robot.data.joint_pos.torch[0].cpu().numpy()
            parts.append(np.r_[q[arm], np.clip(q[fingers].mean() / FINGER_OPEN, 0, 1)])
        return np.concatenate(parts)

    def _record(self):
        parts = []
        offset = 0

        def add(name, value):
            nonlocal offset
            array = np.asarray(value).ravel()
            self.schema[name] = [offset, offset + len(array)]
            offset += len(array)
            parts.append(array)

        def array(value):
            return value.torch[0].cpu().numpy()

        add("time", [self.time])
        for side, robot in zip(("left", "right"), self.robots):
            add(side + "_q", array(robot.data.joint_pos))
            add(side + "_dq", array(robot.data.joint_vel))
        for name, obj in self.objects.items():
            add(name + "_pose", array(obj.data.root_pose_w))
            add(name + "_velocity", array(obj.data.root_vel_w))
            if hasattr(obj, "joint_names"):
                add(name + "_q", array(obj.data.joint_pos))
                add(name + "_dq", array(obj.data.joint_vel))
                add(name + "_links", array(obj.data.body_link_pose_w))
        add("steak_contacts", self.forces)
        row = np.concatenate(parts)
        assert np.isfinite(row).all(), "Non-finite measured state"
        self.records.append(row)
        if self.ticks % 8 == 0:
            self.frames.append({**self.runtime.state(), "phase": self.phase})

    def _after_physics(self):
        self.ticks += 1
        self.forces = self.contact.get_contact_force_matrix(self.dt)[0].copy()
        self.scorer.update(self.object_state('steak'), self.object_state('pan'),
                           bool(np.linalg.norm(self.forces[1:],axis=1).max()>.05))
        self._record()

    def step(self, action):
        assert self.steps < CONFIG["episode"]["max_control_steps"], "Trial time limit reached"
        requested = np.asarray(action, dtype=np.float32)
        assert requested.shape == (14,) and np.isfinite(requested).all()
        self.action = np.clip(requested, self.low, self.high).astype(np.float32)
        self.commands.append(self.action.copy())
        self.base.action_manager.process_action(torch.tensor(self.action[None]))
        for _ in range(SUBSTEPS):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            self._after_physics()
        update(self.base.sim.stage, self.time, self.scorer.cooked)
        self.steps += 1
        assert all(np.all(robot.actuators.target_command.velocity.torch[0].cpu().numpy() == 0)
                   for robot in self.robots), "Unexpected non-native velocity target"

    def capture(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        prepare_render()
        self.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        images = {}
        for name in CONFIG["observation"]["camera_names"]:
            sensor = self.base.scene[name]
            sensor.update(self.control_dt, force_recompute=True)
            images[name] = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
        return images

    def save(self, status, error=None):
        np.savez_compressed(self.out / "trajectory.npz", states=np.asarray(self.records),
                            actions=np.asarray(self.commands))
        write_json(self.out / "frame_states.json", self.frames)
        write_json(self.out / "result.json", {"status": status, "error": error,
            "control_steps": self.steps, "physics_ticks": self.ticks, "config": CONFIG,
            "joint_names": {name: obj.joint_names for name, obj in self.objects.items()
                            if hasattr(obj, "joint_names")}, **self.runtime.state()})

    def close(self):
        self.env.close()

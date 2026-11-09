"""Shared native physical stepping and measured execution records."""

import json
import hashlib
import shutil

import numpy as np
import torch

from scene import CONFIG, ROOT, make_env
from task import score, INITIAL, cell


def archive_source(out):
    source = out/"source"
    source.mkdir()
    for path in [*ROOT.glob("*.py"), ROOT/"dream-defaults.json", *ROOT.glob("demo_actions.npz")]:
        shutil.copy2(path, source/path.name)
    shutil.copytree(ROOT/"assets", source/"assets")
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    (out/"source_manifest.json").write_text(json.dumps(hashes, indent=2)+"\n")
    return hashes


class Simulation:
    def __init__(self, out, cameras=False):
        import carb
        import omni.usd
        from yam_arena.constants import READY
        self.out, self.cameras = out, cameras
        self.env = make_env(out, cameras)
        self.base = self.env.unwrapped
        self.env.reset()
        self.stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.robots = [self.base.scene[name+"_robot"] for name in ("left", "right")]
        self.tiles = [self.base.scene[f"tile_{i}"] for i in range(1, 9)]
        self.dt = 1/CONFIG["simulation"]["physics_hz"]
        self.substeps = CONFIG["simulation"]["physics_hz"]//CONFIG["simulation"]["control_hz"]
        self.control_dt = self.dt*self.substeps
        assert np.isclose(self.base.physics_dt, self.dt) and np.isclose(self.base.step_dt, self.control_dt)
        reference = CONFIG["robot_reference"]
        for robot in self.robots:
            for name in ("shoulder", "wrist", "gripper"):
                actuator = robot.cfg.actuators[name]
                for key, attribute in (("effort", "joint_effort_limit"), ("velocity", "joint_velocity_limit"),
                                       ("stiffness", "stiffness"), ("damping", "damping")):
                    assert np.isclose(getattr(actuator, attribute), reference[name][key]), (name, key)
            props = robot.cfg.spawn.articulation_props
            assert props.solver_position_iteration_count == reference["solver_position_iterations"]
            assert props.solver_velocity_iteration_count == reference["solver_velocity_iterations"]
            assert props.enabled_self_collisions == reference["enabled_self_collisions"]
        self.ticks, self.steps, self.phase = 0, 0, "initial"
        self.records, self.commands, self.phases = [], [], []

        for _ in range(40):
            self._advance(self.action, record=False)

        self.ticks, self.steps = 0, 0

        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        root = "/World/envs/env_0/"
        from yam_arena.robot import asset_link_paths
        filters = [root+side+"Arm"+path for side in ("Left", "Right")
                   for path in asset_link_paths().values()]
        self.contacts = [self.physics.create_rigid_contact_view(root+f"tile_{i}/body", filters)
                         for i in range(1, 9)]
        self.records.append(self.state())
        self.phases.append(self.phase)

    @property
    def time(self):
        return self.ticks*self.dt

    def _advance(self, action, record):
        command = np.asarray(action, dtype=np.float32)
        assert command.shape == (14,) and np.isfinite(command).all()
        self.base.action_manager.process_action(torch.tensor(command[None], device=self.base.device))
        for _ in range(self.substeps):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(self.dt)
            self.ticks += 1
            if record:
                self.records.append(self.state())
                self.phases.append(self.phase)

    def step(self, action):
        assert self.steps < CONFIG["episode"]["max_control_steps"], "Task time limit reached"
        self.commands.append(np.asarray(action, dtype=np.float32).copy())
        self._advance(action, record=True)
        self.action = np.asarray(action, dtype=np.float32).copy()
        self.steps += 1
        assert all(np.all(robot.actuators.target_command.velocity.torch[0].cpu().numpy() == 0)
                   for robot in self.robots), "Non-native velocity command"

    def wait(self, seconds):
        for _ in range(round(seconds/self.control_dt)):
            self.step(self.action)

    def state(self):
        def array(value):
            return value.torch[0].cpu().numpy().copy()
        parts = [[self.time]]
        for robot in self.robots:
            parts.extend([array(robot.data.joint_pos), array(robot.data.joint_vel)])
        for tile in self.tiles:
            parts.extend([array(tile.data.root_link_pose_w), array(tile.data.root_link_vel_w)])
        parts.append(self.contact_forces())
        result = np.concatenate(parts)
        assert np.isfinite(result).all(), "Non-finite physical state"
        return result

    def contact_forces(self):
        return np.array([np.linalg.norm(view.get_contact_force_matrix(dt=self.dt), axis=-1).sum()
                         for view in self.contacts])

    def result(self):
        poses = np.array([tile.data.root_link_pose_w.torch[0].cpu().numpy() for tile in self.tiles])
        return score(poses, self.contact_forces(), self.time)

    def check(self):
        self.wait(2.)
        poses = [tile.data.root_link_pose_w.torch[0].cpu().tolist() for tile in self.tiles]
        for i, pose in enumerate(poses, 1):
            assert np.linalg.norm(np.array(pose[:3])-cell(INITIAL.index(i))) < .001
        assert self.result()["score"] == 0.
        report = dict(poses=poses, config=CONFIG, checks_passed=True, result=self.result())
        (self.out/"check.json").write_text(json.dumps(report, indent=2)+"\n")
        print("CHECK PASSED", self.result(), flush=True)

    def save(self, status, error=None):
        columns = dict(time=1, left_q=8, left_qd=8, right_q=8, right_qd=8)
        for i in range(1, 9):
            columns[f"tile_{i}_pose"], columns[f"tile_{i}_velocity"] = 7, 6
        columns["robot_contact_n"] = 8
        report = dict(status=status, error=error, phase=self.phase, **self.result(),
                      config=CONFIG, record_columns=columns)
        (self.out/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(self.out/"trajectory.npz", states=np.asarray(self.records),
                            actions=np.asarray(self.commands), phases=np.asarray(self.phases))

    def close(self):
        self.env.close()

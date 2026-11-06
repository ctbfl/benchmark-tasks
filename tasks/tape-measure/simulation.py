"""Shared physical stepping, instrument updates and execution records."""

import json
import hashlib
import shutil

import numpy as np
import torch

from scene import CONFIG, ROOT, make_env
from task import Instruments


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
        self.tape = self.base.scene["tape"]
        self.calculator = self.base.scene["calculator"]
        self.wood = self.base.scene["wood"]
        self.dt = 1/CONFIG["simulation"]["physics_hz"]
        self.substeps = CONFIG["simulation"]["physics_hz"]//CONFIG["simulation"]["control_hz"]
        self.control_dt = self.dt*self.substeps
        assert np.isclose(self.base.physics_dt, self.dt) and np.isclose(self.base.step_dt, self.control_dt)
        self.ticks, self.steps, self.phase = 0, 0, "initial"
        self.records, self.commands, self.phases = [], [], []
        self.instruments = Instruments(self.base, self.stage)
        for _ in range(40):
            self._advance(self.action, record=False)
        assert not self.instruments.events, "Calculator keys fired during initial settling"
        self.ticks, self.steps = 0, 0
        self.instruments = Instruments(self.base, self.stage)
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
            self.instruments.update(self.time)
            if record:
                self.records.append(self.state())
                self.phases.append(self.phase)
        self.instruments.update_visuals()

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
        for asset in (self.tape, self.calculator):
            parts.extend([array(asset.data.root_link_pose_w), array(asset.data.root_link_vel_w),
                          array(asset.data.joint_pos), array(asset.data.joint_vel)])
        parts.extend([array(self.wood.data.root_link_pose_w), array(self.wood.data.root_link_vel_w)])
        result = np.concatenate(parts)
        assert np.isfinite(result).all(), "Non-finite physical state"
        return result

    def check(self):
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from model import KEY_STIFFNESS, KEYS, TAPE_STIFFNESS, WOOD_SIZE
        from yam_arena import station
        assert self.tape.num_joints == 1 and self.calculator.num_joints == len(KEYS)
        assert not self.tape.is_fixed_base and not self.calculator.is_fixed_base
        physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        tape = physics.create_articulation_view("/World/envs/env_0/tape")
        calculator = physics.create_articulation_view("/World/envs/env_0/calculator")
        assert np.allclose(tape.get_dof_stiffnesses(), TAPE_STIFFNESS)
        assert np.allclose(calculator.get_dof_stiffnesses(), KEY_STIFFNESS)
        assert abs(float(self.tape.data.joint_pos.torch[0, 0])) < .001
        self.wait(2.)
        positions = {name: self.base.scene[name].data.root_link_pose_w.torch[0].cpu().tolist()
                     for name in ("tape", "calculator", "wood")}
        assert positions["calculator"][1] > positions["tape"][1]
        assert positions["wood"][0] > max(positions["calculator"][0], positions["tape"][0])
        assert abs(positions["wood"][2]-station.TABLE_TOP-WOOD_SIZE[2]/2) < .002
        assert not self.instruments.events
        report = {"assets": positions, "tape_joints": self.tape.joint_names,
                  "calculator_joints": self.calculator.joint_names, "config": CONFIG,
                  "initial_score": self.instruments.result(self.time), "checks_passed": True}
        (self.out/"check.json").write_text(json.dumps(report, indent=2)+"\n")
        print("CHECK PASSED", json.dumps(positions), flush=True)

    def save(self, status, error=None):
        report = {"status": status, "error": error, "phase": self.phase,
                  **self.instruments.result(self.time), "config": CONFIG,
                  "record_columns": {"time": 1, "left_q": 8, "left_qd": 8,
                    "right_q": 8, "right_qd": 8, "tape_pose": 7, "tape_velocity": 6,
                    "tape_q": self.tape.num_joints, "tape_qd": self.tape.num_joints,
                    "calculator_pose": 7, "calculator_velocity": 6,
                    "calculator_q": self.calculator.num_joints, "calculator_qd": self.calculator.num_joints,
                    "wood_pose": 7, "wood_velocity": 6}}
        (self.out/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(self.out/"trajectory.npz", states=np.asarray(self.records),
                            actions=np.asarray(self.commands), phases=np.asarray(self.phases))

    def close(self):
        self.env.close()

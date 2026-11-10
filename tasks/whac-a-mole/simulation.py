"""Shared physical stepping, light/score updates, camera capture and execution records."""

import hashlib
import json
import shutil

import numpy as np
import torch

from scene import CONFIG, ROOT, make_env
from task import Game, Instruments

LIMIT_S = CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"]


def archive_source(out):
    source = out/"source"
    source.mkdir()
    for path in [*ROOT.glob("*.py"), ROOT/"dream-defaults.json"]:
        shutil.copy2(path, source/path.name)
    shutil.copytree(ROOT/"assets", source/"assets")
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob("*")) if p.is_file()}
    (out/"source_manifest.json").write_text(json.dumps(hashes, indent=2)+"\n")
    return hashes


class Simulation:
    def __init__(self, out, cameras=False):
        import carb
        import omni.usd
        from yam_arena.constants import READY
        from layout import BUTTONS
        self.out, self.cameras = out, cameras
        self.env = make_env(out, cameras)
        self.base = self.env.unwrapped
        self.env.reset()
        self.stage = omni.usd.get_context().get_stage()
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.robots = [self.base.scene[name+"_robot"] for name in ("left", "right")]
        self.device = self.base.scene["device"]
        sim = CONFIG["simulation"]
        self.hz = sim["physics_hz"]
        self.dt = 1/self.hz
        self.substeps = self.hz//sim["control_hz"]
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
        game = Game(CONFIG["episode"]["seed"], self.hz, LIMIT_S, BUTTONS)
        self.instruments = Instruments(self.base, self.stage, game)
        # Let the device settle on the table before the game clock starts (no lights during settling).
        for _ in range(40):
            self._advance(self.action, record=False)
        assert not self.instruments.events, "Buttons fired during initial settling"
        self.ticks, self.steps = 0, 0
        self.instruments = Instruments(self.base, self.stage, Game(CONFIG["episode"]["seed"], self.hz,
                                                                   LIMIT_S, BUTTONS))
        self.instruments.update(0)
        self.instruments.update_visuals()
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
                self.instruments.update(self.ticks)
                self.records.append(self.state())
                self.phases.append(self.phase)
        if record:
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

    @property
    def done(self):
        return self.instruments.result(LIMIT_S)["score"] == 1.

    def state(self):
        def array(value):
            return value.torch[0].cpu().numpy().copy()
        from layout import BUTTONS
        parts = [[self.time]]
        for robot in self.robots:
            parts.extend([array(robot.data.joint_pos), array(robot.data.joint_vel)])
        parts.extend([array(self.device.data.root_link_pose_w), array(self.device.data.root_link_vel_w),
                      array(self.device.data.joint_pos), array(self.device.data.joint_vel)])
        lit, hits = self.instruments.visual_state()
        parts.append([-1 if lit is None else list(BUTTONS).index(lit), hits])
        result = np.concatenate(parts)
        assert np.isfinite(result).all(), "Non-finite physical state"
        return result

    def record_columns(self):
        n = self.device.num_joints
        return {"time": 1, "left_q": 8, "left_qd": 8, "right_q": 8, "right_qd": 8,
                "device_pose": 7, "device_velocity": 6, "device_q": n, "device_qd": n, "lit_hits": 2}

    def capture(self, names=("left_cam", "top_cam", "right_cam")):
        """Render and return the robot cameras' current 640x360 RGB images."""
        from render import refresh
        refresh(self)
        images = {}
        for name in names:
            sensor = self.base.scene[name]
            sensor.update(self.control_dt, force_recompute=True)
            images[name] = sensor.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
        return images

    def check(self):
        from layout import BUTTONS
        from yam_arena import station
        assert self.device.num_joints == len(BUTTONS) and not self.device.is_fixed_base
        self.wait(2.)
        pose = self.device.data.root_link_pose_w.torch[0].cpu().numpy()
        travel = self.device.data.joint_pos.torch[0].cpu().numpy()
        assert abs(pose[2]-station.TABLE_TOP) < .003, pose
        assert np.all(np.abs(travel) < .0005), travel
        assert not self.instruments.events
        report = {"device_pose": pose.tolist(), "button_travel_m": travel.tolist(),
                  "joints": self.device.joint_names, "sequence": self.instruments.game.sequence,
                  "config": CONFIG, "checks_passed": True}
        (self.out/"check.json").write_text(json.dumps(report, indent=2)+"\n")
        print("CHECK PASSED", pose.round(4).tolist(), flush=True)

    def save(self, status, error=None):
        report = {"status": status, "error": error, "phase": self.phase,
                  **self.instruments.result(LIMIT_S), "limit_s": LIMIT_S, "control_steps": self.steps,
                  "config": CONFIG, "record_columns": self.record_columns()}
        (self.out/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        np.savez_compressed(self.out/"trajectory.npz", states=np.asarray(self.records),
                            actions=np.asarray(self.commands), phases=np.asarray(self.phases))

    def close(self):
        self.env.close()

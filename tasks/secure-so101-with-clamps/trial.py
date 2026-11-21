"""Common execution and physical evidence for scripted and model trials."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil

import numpy as np
from inspect_robots import StepResult
from isaaclab.sim.utils import get_current_stage_id
from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, READY
from yam_arena.inspect_embodiment import YamArenaEmbodiment
from yam_arena.robot import asset_link_paths

from geometry import GAP_CENTER_Z, PARK_ANGLE
from scene import ARMS, CAMERAS, CLAMPS, CONFIG, DT, HZ, INSTRUCTION, MAX_STEPS, ROOT, make_env
from score import Score

TARGET = "/".join((ROOT, "so101"))
TABLE_TOP = ROOT + "/fixture/TableTop"
READY_ACTION = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
TRAJECTORY_COLUMNS = 42


def array(value):
    return value.torch[0].cpu().numpy().copy()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


class Trial(YamArenaEmbodiment):
    def __init__(self, out, cameras=False):
        import omni.physics.tensors as tensors
        import torch

        width, height = CONFIG["observation"]["render_size_wh"]
        super().__init__(device=CONFIG["simulation"]["device"], cam_width=width, cam_height=height,
                         control_hz=HZ, scene="station", object_jitter=CONFIG["episode"]["object_jitter_m"])
        self.info = replace(self.info, name="secure_so101", environment_id="yam-secure-so101",
                            supported_target_kinds=frozenset(),
                            docs=self.info.docs.split("Simulated rig")[0] + INSTRUCTION)
        self.out, self.cameras = Path(out), cameras
        self.out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(Path(__file__).parent / "dream-defaults.json", self.out / "dream-defaults.json")
        self._env = make_env(self.out, cameras)
        self._base = self._env.unwrapped
        self._torch = torch
        self.robots = [self._base.scene[name] for name in ARMS]
        self.arm = self._base.scene["so101"]
        self.clamps = [self._base.scene[name] for name in CLAMPS]
        self._ids = {name: (r.find_joints(ARM_JOINTS, preserve_order=True)[0],
                            r.find_joints(FINGER_JOINTS, preserve_order=True)[0])
                     for name, r in zip(ARMS, self.robots)}
        self._instruction = INSTRUCTION
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        paths = asset_link_paths()
        fingers = [[f"{ROOT}/{side}Arm{paths[f'tip_{tip}']}" for tip in ("left", "right")]
                   for side in ("Left", "Right")]
        robot_paths = [f"{ROOT}/{side}Arm{path}" for side in ("Left","Right") for path in paths.values()]
        self.robot_paths = robot_paths
        self.approach_probe = None
        self.right_tips = [robot_paths.index(f) for f in fingers[1]]
        self.plate_contact = self.physics.create_rigid_contact_view(TARGET, robot_paths)
        self.left_contact = self.physics.create_rigid_contact_view(fingers[0], [[TARGET]] * 2, max_contact_data_count=128)
        self.pad_contacts = []
        self.clamp_robot = []
        for name, obj in zip(CLAMPS, self.clamps):
            assert obj.num_joints == 1 and not obj.is_fixed_base and not obj.actuators
            expression = [f"{ROOT}/{name}/{link}" for link in ("upper", "lower")]
            self.pad_contacts.append(self.physics.create_rigid_contact_view(expression, [[TARGET, TABLE_TOP]] * 2,
                                                                             max_contact_data_count=256))
            self.clamp_robot.append(self.physics.create_rigid_contact_view(expression, [robot_paths] * 2))
        self.hand_ids = [r.body_names.index("gripper") for r in self.robots]
        self.trajectory = (self.out / "trajectory-f64.bin").open("wb")
        self.actions = (self.out / "actions.jsonl").open("w")
        self.metrics = (self.out / "scorer.jsonl").open("w")

    def reset(self, scene=None, *, seed=None):
        if scene is not None:
            self._instruction = scene.instruction
        self._env.reset(seed=CONFIG["episode"]["seed"] if seed is None else seed)
        # Arena's placement event resets root pose. Restore both jaws before settling.
        for clamp in self.clamps:
            clamp.write_joint_state_to_sim(self._torch.full((1, 1), PARK_ANGLE), self._torch.zeros((1, 1)))
        self.scorer = Score()
        self.steps = self.ticks = 0
        self._t = 0.
        self.phase = "reset"
        for _ in range(10):
            self.advance(READY_ACTION, record=False)
        for clamp in self.clamps:
            position, hinge = array(clamp.data.root_pose_w)[:3], array(clamp.data.joint_pos)
            assert abs(position[2] - GAP_CENTER_Z) < .01 and .4 < hinge[0] < .6, (position, hinge)
        self.save_state()
        self.last = self.measure()
        self.metrics.write(json.dumps(dict(step=0, phase=self.phase, **self.last)) + "\n")
        return self.observe() if self.cameras else None

    def advance(self, command, record=True):
        base = self._base
        command = np.asarray(command, dtype=np.float32)
        assert command.shape == (14,) and np.isfinite(command).all()
        command = np.clip(command, self._low, self._high).astype(np.float32)
        base.action_manager.process_action(self._torch.tensor(command[None, :], device=base.device))
        for _ in range(base.cfg.decimation):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(DT)
            if record:
                self.ticks += 1
                if self.approach_probe is not None:
                    self.approach_probe()
                if self.ticks % 8 == 0:
                    self.save_state()

    def save_state(self):
        row = np.concatenate(([self.ticks * DT], *[array(robot.data.joint_pos) for robot in self.robots],
                              array(self.arm.data.root_pose_w),
                              *[np.r_[array(c.data.root_pose_w), array(c.data.joint_pos)] for c in self.clamps],
                              [0., 0.])).astype("<f8")
        assert row.shape == (TRAJECTORY_COLUMNS,) and np.isfinite(row).all()
        self.trajectory.write(row.tobytes())

    def measure(self):
        pose = array(self.arm.data.root_pose_w)
        clamp_data = {}
        robot_contacts = [self.plate_contact.get_contact_force_matrix(DT)]
        for name, clamp, pad_view, robot_view in zip(CLAMPS, self.clamps, self.pad_contacts, self.clamp_robot):
            links = array(clamp.data.body_link_pose_w)[[clamp.body_names.index("upper"), clamp.body_names.index("lower")]]
            hinge = float(array(clamp.data.joint_pos)[0])
            forces, points, normals, _, counts, starts = pad_view.get_contact_data(DT)
            assert int(counts.sum()) < pad_view.max_contact_data_count, "Contact buffer saturated"
            patches = []
            for jaw in range(2):
                start = int(starts[jaw, jaw])
                end = start + int(counts[jaw, jaw])
                patches.append(np.column_stack((points[start:end], normals[start:end], forces[start:end])))
            robot_force = robot_view.get_contact_force_matrix(DT)
            robot_contacts.append(robot_force)
            clamp_data[name] = (links, hinge, pad_view.get_contact_force_matrix(DT), patches,
                                robot_force[:, self.right_tips])
        targets = np.vstack([pose[:3] + [x, y, .005] for x in (-.057,-.017,.023,.063,.103)
                             for y in (-.149,-.109,-.069,-.029,.011,.051,.091,.125)] +
                            [links[:, :3] for links, *_ in clamp_data.values()])
        from scipy.spatial.transform import Rotation

        clearance = []
        for robot, hand_id in zip(self.robots, self.hand_ids):
            anchor = array(robot.data.body_link_pose_w)[hand_id]
            # Native finger position is a signed extension per jaw.
            extent = -array(robot.data.joint_pos)[-2:]
            pads = anchor[:3] + Rotation.from_quat(anchor[3:]).apply(
                [[extent[0], 0., .14], [-extent[1], 0., .14]])
            clearance.append(float(np.min(np.linalg.norm(pads[:, None, :] - targets[None, :, :], axis=-1))))
        return self.scorer.update(self.steps, pose, array(self.arm.data.root_vel_w), clamp_data,
                                  np.vstack(robot_contacts), self.left_contact.get_contact_force_matrix(DT)[:, 0],
                                  clearance, self.left_contact.get_contact_data(DT)[4].sum(axis=1) > 0)

    def perform(self, command):
        assert self.steps < MAX_STEPS
        self.advance(command)
        self.steps += 1
        self._t = self.steps / HZ
        self.last = self.measure()
        self.actions.write(json.dumps(dict(step=self.steps, phase=self.phase,
                                      action=np.clip(command, self._low, self._high).tolist())) + "\n")
        self.metrics.write(json.dumps(dict(step=self.steps, phase=self.phase, **self.last)) + "\n")
        return self.last

    def observe(self):
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        self._base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        for name in CAMERAS:
            self._base.scene[name].update(1 / HZ, force_recompute=True)
        return self._observation()

    def step(self, action):
        self.phase = "Dream-1"
        result = self.perform(np.asarray(action.data))
        success = result["score"] == 1.
        return StepResult(observation=self.observe(), terminated=success,
                          termination_reason="success" if success else None,
                          info={"success": success, "score": result["score"]})

    def result(self, kind, status="success"):
        for f in (self.metrics, self.trajectory, self.actions):
            f.flush()
        result = dict(kind=kind, status=status, score=self.last["score"], max_score=1,
                      steps=self.steps,
                      seconds=self._t, raw_scorer=str(self.out / "scorer.jsonl"),
                      executed_actions=str(self.out / "actions.jsonl"),
                      trajectory=str(self.out / "trajectory-f64.bin"), final=self.last)
        if "PRODUCTION_GPU_RUN_ID" in os.environ:
            result["gpu_run_id"] = os.environ["PRODUCTION_GPU_RUN_ID"]
        write_json(self.out / "result.json", result)
        return result

    def close(self):
        for f in (self.metrics, self.trajectory, self.actions):
            f.close()
        self._env.close()

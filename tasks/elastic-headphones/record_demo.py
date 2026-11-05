"""Simulate and measure fixed commands; measured states never select actions."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from demo import ACTION_FILE, HZ

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/demo"
DT = 1 / 240


class Recorder:
    def __init__(self, env):
        import carb
        import omni.physics.tensors as tensors
        import torch
        from pxr import PhysxSchema
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena.robot import asset_link_paths

        self.base, self.torch = env.unwrapped, torch
        self.hp = self.base.scene["headphones"]
        self.robots = {s: self.base.scene[s + "_robot"] for s in ("left", "right")}
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        paths = asset_link_paths()
        fingers = [f"/World/envs/env_0/{side}Arm{paths[tip]}" for side in ("Left", "Right")
                   for tip in ("tip_left", "tip_right")]
        root = "/World/envs/env_0/headphones"
        self.contacts = {n: self.physics.create_rigid_contact_view(root + "/" + n, fingers)
                         for n in ("band_*", "left_cup", "right_cup")}
        for name in self.hp.body_names:
            PhysxSchema.PhysxContactReportAPI.Apply(
                self.base.sim.stage.GetPrimAtPath(root + "/" + name)).CreateThresholdAttr(0)
        self.support = self.physics.create_rigid_contact_view(
            root + "/band_*", ["/World/envs/env_0/headphone_stand"])
        left = ["/World/envs/env_0/LeftArm" + p for p in paths.values()]
        right = ["/World/envs/env_0/RightArm" + p for p in paths.values()]
        self.arm_contacts = self.physics.create_rigid_contact_view(left, [right] * len(left))
        robot_links = left + right
        self.robot_contacts = self.physics.create_rigid_contact_view(
            [root + "/" + name for name in self.hp.body_names], [robot_links] * self.hp.num_bodies)
        self.stand_contacts = self.physics.create_rigid_contact_view(
            "/World/envs/env_0/headphone_stand", robot_links)
        self.view = self.physics.create_articulation_view(root)
        self.spring_stiffness = self.view.get_dof_stiffnesses().copy()
        self.spring_targets = self.view.get_dof_position_targets().copy()
        self.rows, self.time = [], 0.

    def step(self, action):
        self.base.action_manager.process_action(self.torch.tensor(action[None], dtype=self.torch.float32))
        robot_force, arm_force = 0., 0.
        for _ in range(8):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(DT)
            robot_force = max(robot_force, *[float(np.linalg.norm(v.get_contact_force_matrix(DT), axis=-1).sum())
                for v in (self.robot_contacts, self.stand_contacts)])
            arm_force = max(arm_force, float(np.linalg.norm(
                self.arm_contacts.get_contact_force_matrix(DT), axis=-1).sum()))
        self.time += 1 / HZ
        self.rows.append(dict(t=self.time, action=action.tolist(),
            stand=self.base.scene["headphone_stand"].data.root_pose_w.torch[0].cpu().tolist(),
            robot_contact_N=robot_force, inter_arm_force_N=arm_force,
            support_force_N=float(np.linalg.norm(self.support.get_contact_force_matrix(DT), axis=-1).sum()),
            root=self.view.get_root_transforms()[0].tolist(),
            joint_degrees=np.rad2deg(self.view.get_dof_positions()[0]).tolist(),
            force_N={n: np.linalg.norm(v.get_contact_force_matrix(DT), axis=-1).sum(axis=0).tolist()
                     for n, v in self.contacts.items()},
            robot_joint_positions={s: a.data.joint_pos.torch[0].cpu().tolist() for s, a in self.robots.items()},
            links=self.hp.data.body_link_pose_w.torch[0].cpu().tolist()))

    def save(self, error=None):
        (OUT / "trajectory.json").write_text(json.dumps(self.rows) + "\n")
        report = dict(completed=error is None, error=error, controller="recorded_open_loop",
            control_hz=HZ, physics_hz=240, steps=len(self.rows),
            action_sha256=hashlib.sha256(ACTION_FILE.read_bytes()).hexdigest(),
            headphone_joint_names=self.hp.joint_names, headphone_link_names=self.hp.body_names,
            headband_springs_unchanged=bool(np.array_equal(self.spring_stiffness, self.view.get_dof_stiffnesses())
                and np.array_equal(self.spring_targets, self.view.get_dof_position_targets())))
        (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


def run(actions):
    from pxr import Sdf
    from yam_scene import make_env, START_ANGLES

    (OUT / "source").mkdir(parents=True, exist_ok=True)
    for name in ("demo.py", "record_demo.py", "demo_actions.npy", "yam_scene.py", "scoring.py", "score.py"):
        shutil.copy2(HERE / name, OUT / "source" / name)
    env = make_env(enable_cameras=False, control_hz=HZ, log_dir=OUT / "logs")
    recorder = None
    try:
        for name in START_ANGLES:
            prim = env.unwrapped.sim.stage.GetPrimAtPath("/World/envs/env_0/headphones/joints/" + name)
            assert prim.ApplyAPI("PhysxJointAxisAPI", "angular")
            for attr, value in (("staticFrictionEffort", .07), ("dynamicFrictionEffort", .05)):
                prim.CreateAttribute("physxJointAxis:angular:" + attr, Sdf.ValueTypeNames.Float).Set(value)
        env.reset()
        recorder = Recorder(env)
        for i, action in enumerate(actions):
            recorder.step(action)
            if (i + 1) % 150 == 0:
                print("PLAYBACK", i + 1, len(actions), "time", recorder.time, flush=True)
        recorder.save()
        from score import score_demo
        result = score_demo(OUT)
        (OUT / "score.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["score"] == 1., result
        print("DEMO SCORE", result["score"], flush=True)
    except Exception as error:
        if recorder is not None:
            recorder.save(str(error))
        raise
    finally:
        env.close()

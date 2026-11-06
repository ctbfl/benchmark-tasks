"""Native robot action stepping; independent measurements never control the policy."""
from collections import deque

import numpy as np

from scene import CONFIG, ROOT
from score import STABLE_SECONDS, score


class TaskRuntime:
    def __init__(self, env):
        self.base = env.unwrapped
        self.time = 0.0
        self.ticks = 0
        self.measurements = []
        self.window = deque()
        self.result = score([])
        import carb
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        from yam_arena.robot import asset_link_paths
        paths = asset_link_paths()
        robots = [ROOT + "/" + side + paths[body] for side in ("LeftArm", "RightArm")
                  for body in self.base.scene["left_robot"].body_names]
        clamp_paths = [ROOT + "/clamp/" + name for name in self.base.scene["clamp"].body_names]
        self.contacts = {"clamp": self.physics.create_rigid_contact_view(clamp_paths, [robots] * len(clamp_paths))}
        for name in ("thermometer", "stand"):
            self.contacts[name] = self.physics.create_rigid_contact_view(ROOT + "/" + name, robots)

    def measure(self):
        sample = {"time_s": self.time}
        for name in ("clamp", "stand", "thermometer", "beaker"):
            sample[name] = self.base.scene[name].data.root_pose_w.torch[0].cpu().numpy().tolist()
        for name, contact in self.contacts.items():
            sample[name + "_robot_force_N"] = float(np.linalg.norm(
                contact.get_contact_force_matrix(1 / CONFIG["simulation"]["physics_hz"]), axis=-1).sum())
        clamp = self.base.scene["clamp"]
        q = clamp.data.joint_pos.torch[0].cpu().numpy()
        sample["screw_travel"] = float(q[clamp.joint_names.index("screw_travel")])
        sample["jaw_angle"] = float(q[clamp.joint_names.index("jaw_hinge")])
        self.measurements.append(sample)
        self.window.append(sample)
        while self.window[0]["time_s"] < self.time - STABLE_SECONDS - 1e-7:
            self.window.popleft()
        self.result = score(list(self.window))

    def step(self, action, on_physics_step=None):
        base = self.base
        dt = 1 / CONFIG["simulation"]["physics_hz"]
        base.action_manager.process_action(action)
        for _ in range(CONFIG["simulation"]["physics_hz"] // CONFIG["simulation"]["control_hz"]):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(dt)
            self.time += dt
            self.ticks += 1
            if self.ticks % (CONFIG["simulation"]["physics_hz"] // CONFIG["simulation"]["control_hz"]) == 0:
                self.measure()
            if on_physics_step is not None:
                on_physics_step()

    @property
    def score(self):
        return self.result["score"]

    def state(self):
        return {"time_s": self.time, **self.result}

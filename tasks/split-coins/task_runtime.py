"""Native stepping with per-control-step coin measurements and the binary sorting score."""
import numpy as np
from scipy.spatial.transform import Rotation

from build_assets import COINS, FRAMES
from scene import CONFIG, COIN_NAMES


def kind_of(name):
    return name.split("_")[0]


def inside_frame(pose, kind):
    """True when the coin's whole horizontal footprint lies within the tape's inner edge."""
    (cx, cy), (sx, sy) = FRAMES[kind]
    normal = Rotation.from_quat(pose[3:]).apply([0, 0, 1])
    radius = COINS[kind]["radius"]
    reach_x = radius*np.sqrt(max(0., 1-normal[0]**2))
    reach_y = radius*np.sqrt(max(0., 1-normal[1]**2))
    return bool(abs(pose[0]-cx) + reach_x <= sx/2 and abs(pose[1]-cy) + reach_y <= sy/2)


def evaluate(poses):
    """poses: {coin name: [x, y, z, qx, qy, qz, qw]} in world coordinates."""
    inside = {name: inside_frame(np.asarray(poses[name]), kind_of(name)) for name in COIN_NAMES}
    sorted_count = {kind: sum(inside[n] for n in COIN_NAMES if kind_of(n) == kind) for kind in COINS}
    return {"score": float(all(inside.values())), "inside": inside,
            "pennies_in_left_frame": sorted_count["penny"], "quarters_in_right_frame": sorted_count["quarter"]}


class TaskRuntime:
    def __init__(self, env):
        self.base = env.unwrapped
        self.ticks = 0
        self.time = 0.
        self.measurements = []
        self.result = {}
        self.measure()

    def pose(self, name):
        return self.base.scene[name].data.root_pose_w.torch[0].cpu().numpy().copy()

    def poses(self):
        return {name: self.pose(name) for name in ["jar", *COIN_NAMES]}

    def measure(self):
        poses = self.poses()
        self.result = evaluate(poses)
        self.measurements.append({"time_s": self.time, "poses": {k: v.tolist() for k, v in poses.items()},
                                  "score": self.result["score"],
                                  "pennies_in_left_frame": self.result["pennies_in_left_frame"],
                                  "quarters_in_right_frame": self.result["quarters_in_right_frame"]})

    def step(self, action, on_physics_step=None):
        base = self.base
        hz = CONFIG["simulation"]["physics_hz"]
        decimation = hz // CONFIG["simulation"]["control_hz"]
        base.action_manager.process_action(action)
        for _ in range(decimation):
            base.action_manager.apply_action()
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(1/hz)
            self.ticks += 1
            self.time = self.ticks/hz
            if on_physics_step is not None:
                on_physics_step()
        self.measure()

    def state(self):
        return {"time_s": round(self.time, 4), "score": self.result["score"],
                "pennies_in_left_frame": self.result["pennies_in_left_frame"],
                "quarters_in_right_frame": self.result["quarters_in_right_frame"]}

"""Demonstration policy: fixed joint targets triggered by Top Camera brightness regions.

The policy observes only the robot's Top Camera image. Each button has a fixed pixel
region computed from the fixed device layout and the calibrated camera model. A region
counts as lit when its red and green channels are both bright. Joint targets come from offline inverse
kinematics of the fixed layout. No simulator object state is read by the policy.
"""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from layout import BASE_SIZE, BUTTON_GAP, BUTTON_HEIGHT, BUTTONS
from scene import DEVICE_XY

PAD_Z = .140                   # gripper link to closed fingertip
TCP_Z = .134
HOVER_DZ = .060                # fingertip above the cap while waiting
ABOVE_DZ = .025                # fingertip above a near cap before pressing
PRESS_DZ = -.0075              # target below the cap's rest surface
ARM_OF = {"far_left": 0, "near_left": 0, "far_right": 1, "near_right": 1}
HOME_OF = {0: "far_left", 1: "far_right"}
YAW_DEG = -90.
ROI_HALF = 3                   # 7 x 7 pixel region
ROI_SHIFT_M = -.010            # sample the robot-side half of each cap
SETTLE_S = 2.0                 # keep acting after the fifth hit so the result is visible


def cap_top(name):
    from yam_arena import station
    x, y = BUTTONS[name]
    return np.array([DEVICE_XY[0]+x, DEVICE_XY[1]+y,
                     station.TABLE_TOP+BASE_SIZE[2]+BUTTON_GAP+BUTTON_HEIGHT])


def joint_target(side, point, q_init):
    from yam_arena import station
    from yam_arena.constants import READY
    from yam_arena.ik import LOWER, UPPER, solve
    rotation = Rotation.from_euler("z", YAW_DEG, degrees=True)*Rotation.from_euler("x", 180, degrees=True)
    tcp = np.asarray(point)+rotation.apply([0, 0, TCP_Z-PAD_Z])
    q, ep, er = solve(tcp, rotation.as_quat(), READY if q_init is None else q_init,
                      station.BASES[("left", "right")[side]], damping=.015, iters=300, tol=1e-6)
    assert ep < 1e-3 and er < .02 and np.all(q >= LOWER) and np.all(q <= UPPER), (side, point, ep, er)
    return q


def targets():
    poses = {}
    for name, side in ARM_OF.items():
        top = cap_top(name)
        poses[name] = {key: joint_target(side, top+[0, 0, dz], None)
                       for key, dz in (("hover", HOVER_DZ), ("above", ABOVE_DZ), ("press", PRESS_DZ))}
    return poses


def regions():
    """Pixel centres of the button sampling regions in the 640x360 Top Camera image."""
    from yam_arena import station
    cam = station.rig("calibrated").top_cam
    lens = cam.lens
    rotation = Rotation.from_quat(cam.rot_xyzw)
    centres = {}
    for name in BUTTONS:
        p = rotation.inv().apply(cap_top(name)+[ROI_SHIFT_M, 0, 0]-np.asarray(cam.pos))
        u = lens.cx+lens.fx*p[0]/-p[2]
        v = lens.cy-lens.fy*p[1]/-p[2]
        centres[name] = (int(round(u)), int(round(v)))
    return centres


def lit_buttons(image, centres):
    """Return the buttons whose region shows the bright lamp colour, and the per-region mean colours."""
    means = {}
    lit = []
    for name, (u, v) in centres.items():
        patch = image[v-ROI_HALF:v+ROI_HALF+1, u-ROI_HALF:u+ROI_HALF+1].reshape(-1, 3).astype(float)
        r, g, b = patch.mean(0)
        means[name] = [round(r, 1), round(g, 1), round(b, 1)]
        if r > 170 and g > 170:
            lit.append(name)
    return lit, means


class VisionPolicy:
    def __init__(self, control_hz):
        from yam_arena.constants import READY
        self.hz = control_hz
        self.poses = targets()
        self.centres = regions()
        self.action = np.array([*READY, 1., *READY, 1.], dtype=np.float32)
        self.start = self.action.copy()
        self.queue = {0: [], 1: []}
        self.steps = 0
        self.goal = np.r_[self.poses[HOME_OF[0]]["hover"], 0., self.poses[HOME_OF[1]]["hover"], 0.]

    def plan(self, name):
        side, p = ARM_OF[name], self.poses[name]
        home = self.poses[HOME_OF[side]]["hover"]
        if name.startswith("far"):
            return [p["press"]]*5+[home]*4
        return [p["above"]]*4+[p["press"]]*5+[p["above"]]*3+[home]*5

    def act(self, image):
        """One 20 Hz decision from a Top Camera RGB image."""
        from yam_arena.ik import smoothstep
        lit, means = lit_buttons(image, self.centres)
        approach = round(3.*self.hz)
        if self.steps < approach:
            t = smoothstep((self.steps+1)/approach)
            self.action = (self.start*(1-t)+self.goal*t).astype(np.float32)
        else:
            for name in lit:
                side = ARM_OF[name]
                if not self.queue[side]:
                    self.queue[side] = self.plan(name)
            for side in (0, 1):
                if self.queue[side]:
                    self.action[7*side:7*side+6] = self.queue[side].pop(0)
        self.steps += 1
        return self.action.copy(), lit, means


def run(out):
    from simulation import LIMIT_S, Simulation, archive_source
    archive_source(out)
    sim = Simulation(out, cameras=True)
    policy = VisionPolicy(round(1/sim.control_dt))
    (out/"run.json").write_text(json.dumps(dict(
        policy="top-camera brightness regions + fixed joint targets",
        policy_observations=["top_cam RGB 640x360"], regions_px=policy.centres,
        roi_size_px=2*ROI_HALF+1, control_hz=1/sim.control_dt), indent=2)+"\n")
    log = (out/"vision.jsonl").open("w")
    try:
        finish = None
        while sim.steps < round(LIMIT_S/sim.control_dt):
            image = sim.capture(("top_cam",))["top_cam"]
            action, lit, means = policy.act(image)
            log.write(json.dumps(dict(step=sim.steps, time_s=round(sim.time, 4), detected=lit,
                                      means=means))+"\n")
            sim.phase = "approach" if policy.steps <= 60 else ("press" if any(policy.queue.values()) else "wait")
            sim.step(action)
            if finish is None and sim.done:
                finish = sim.steps+round(SETTLE_S/sim.control_dt)
                print("FIFTH HIT", round(sim.time, 3), flush=True)
            if finish is not None and sim.steps >= finish:
                break
        result = sim.instruments.result(LIMIT_S)
        print("RESULT", json.dumps({k: result[k] for k in ("score", "hits", "hit_times_s")}), flush=True)
        assert result["score"] == 1., "Demonstration did not reach five hits"
        sim.save("success")
    except Exception as error:
        sim.save("error", str(error))
        raise
    finally:
        log.close()
        sim.close()

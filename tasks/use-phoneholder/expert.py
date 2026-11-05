"""State-informed YAM expert; only arm joint targets and gripper openings act on the scene."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scoring import score_placement

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/demo"
HZ = 30
STOP_AFTER = "all"
RECORD_VIDEO = True


def rotation(axis, degrees):
    return Rotation.from_rotvec(np.asarray(axis) * np.deg2rad(degrees)).as_matrix()


def pose_matrix(pose):
    return np.asarray(pose[:3]), Rotation.from_quat(pose[3:]).as_matrix()


class Expert:
    def __init__(self, env):
        import carb
        import omni.physics.tensors as tensors
        import torch
        from isaaclab.sim.utils import get_current_stage_id
        from yam_arena import ik, station
        from yam_arena.constants import READY
        from yam_arena.robot import asset_link_paths

        self.env, self.base = env, env.unwrapped
        self.ik, self.station, self.torch = ik, station, torch
        self.q = {side: np.array(READY, float) for side in ("left", "right")}
        self.grip = {side: 1.0 for side in self.q}
        self.target = {side: ik.tcp_pose(q, station.BASES[side]) for side, q in self.q.items()}
        self.holder, self.phone = self.base.scene["phoneholder"], self.base.scene["phone"]
        self.links = {n: i for i, n in enumerate(self.holder.body_names)}
        self.joints = {n: i for i, n in enumerate(self.holder.joint_names)}
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(ROOT / "assets/phoneholder.usda"))
        cache = UsdGeom.XformCache()
        self.holder_points = {}
        for name in self.links:
            body = stage.GetPrimAtPath("/PhoneHolder/" + name)
            inverse = cache.GetLocalToWorldTransform(body).GetInverse()
            points = []
            for prim in Usd.PrimRange(body):
                if prim.IsA(UsdGeom.Mesh):
                    transform = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
                    points.append(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()) @ transform[:3, :3] + transform[3, :3])
            self.holder_points[name] = np.concatenate(points)
        assert not self.holder.cfg.actuators
        # Lab submits zero efforts even for passive assets, waking them every step.
        # Leave the passive articulation entirely to native PhysX after reset.
        self.holder.actuators.submit_commands = lambda: None
        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
        physics = tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
        physics.set_subspace_roots("/")
        paths = asset_link_paths()
        fingers = [f"/World/envs/env_0/{side}Arm{paths[tip]}"
                   for side in ("Left", "Right") for tip in ("tip_left", "tip_right")]
        self.contacts = physics.create_rigid_contact_view(
            "/World/envs/env_0/phoneholder/*", fingers, max_contact_data_count=128,
        )
        self.hand_contacts = physics.create_rigid_contact_view(
            [f"/World/envs/env_0/LeftArm{paths[n]}" for n in ("gripper", "tip_left", "tip_right")],
            [[f"/World/envs/env_0/RightArm{paths[n]}" for n in ("gripper", "tip_left", "tip_right")]] * 3,
        )
        self.phone_contacts = physics.create_rigid_contact_view(
            "/World/envs/env_0/phone",
            ["/World/envs/env_0/phoneholder/cradle", "/World/envs/env_0/table/geometry/mesh", *fingers],
        )
        self.physics = physics
        self.passive_view = physics.create_articulation_view("/World/envs/env_0/phoneholder")
        print("PASSIVE FRICTION", self.passive_view.get_dof_friction_properties().cpu().numpy(), flush=True)
        print("PASSIVE STIFFNESS", self.passive_view.get_dof_stiffnesses().cpu().numpy(), flush=True)
        self.rows, self.frames, self.phase, self.step_id = [], [], "settle", 0
        self.record = None
        self.press_active = False
        self.press_side = "right"
        self.press_load = 6.0
        self.press_force = 10.0
        self.press_balance_deg = 0.0
        self.press_height = station.TABLE_TOP - .003
        self.last_force = np.zeros((4, 4, 3))

    def tcp(self, side):
        from yam_arena.constants import TCP_OFFSET

        robot = self.base.scene[side + "_robot"]
        body = robot.find_bodies("gripper")[0][0]
        p, r = pose_matrix(robot.data.body_link_pose_w.torch[0, body].cpu().numpy())
        return p + r @ TCP_OFFSET, r

    def body(self, name):
        return pose_matrix(self.holder.data.body_link_pose_w.torch[0, self.links[name]].cpu().numpy())

    def phone_pose(self):
        return pose_matrix(self.phone.data.root_link_pose_w.torch[0].cpu().numpy())

    def angles(self):
        q = self.holder.data.joint_pos.torch[0].cpu().numpy()
        return np.rad2deg(q[[self.joints[n] for n in ("yaw", "lower", "upper")]])

    def panel_angle(self):
        _, rot = self.body("cradle")
        return float(np.rad2deg(np.arctan2(rot[2, 1], np.linalg.norm(rot[:2, 1]))))

    def frame_margin(self):
        from yam_scene import TAPE_CENTER, TAPE_HALF_SIZE

        points = []
        for name, local in self.holder_points.items():
            pos, rot = self.body(name)
            points.append((local @ rot.T + pos)[:, :2])
        return float(TAPE_HALF_SIZE - np.abs(np.concatenate(points) - TAPE_CENTER).max())

    @staticmethod
    def heading_error(rot):
        return float((np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0])) + 90 + 180) % 360 - 180)

    def tick(self):
        if self.press_active:
            finger_slice = slice(0, 2) if self.press_side == "left" else slice(2, 4)
            if self.press_side == "left":
                loads = -self.last_force[:2, :2, 2].sum(axis=0)
                self.press_balance_deg = float(np.clip(self.press_balance_deg + .8 * (loads[0] - loads[1]) / HZ, -3, 3))
                self.press_rotation = rotation([1, 0, 0], self.press_balance_deg) @ self.press_nominal_rotation
            load = -self.last_force[:2, finger_slice, 2].sum()
            self.press_load += .15 * (load - self.press_load)
            self.press_height = np.clip(self.press_height + .002 * (self.press_load - self.press_force) / HZ,
                                        self.press_z_min, self.press_z_max)
            pos, rot = self.target[self.press_side]
            actual_pos, actual_rot = self.tcp(self.press_side)
            xy = pos[:2] + np.clip(.15 * (self.press_xy - actual_pos[:2]), -.0004, .0004)
            error = Rotation.from_matrix(self.press_rotation @ actual_rot.T).as_rotvec()
            correction = Rotation.from_rotvec(np.clip(.15 * error, -.008, .008)).as_matrix()
            self.solve(self.press_side, np.r_[xy, self.press_height], correction @ rot)
        action = np.r_[self.q["left"], self.grip["left"], self.q["right"], self.grip["right"]]
        self.env.step(self.torch.tensor(action[None], dtype=self.torch.float32, device=self.base.device))
        self.step_id += 1
        force = self.contacts.get_contact_force_matrix(self.base.physics_dt).cpu().numpy()
        self.last_force = force.copy()
        bp, br = self.body("base")
        cp, cr = self.body("cradle")
        pp, pr = self.phone_pose()
        self.rows.append({"time": self.step_id / HZ, "phase": self.phase,
                          "joints_deg": self.angles().tolist(), "base_pos": bp.tolist(),
                          "panel_angle_deg": self.panel_angle(),
                          "holder_frame_margin_m": self.frame_margin(),
                          "holder_heading_error_deg": self.heading_error(cr),
                          "phone_heading_error_deg": self.heading_error(pr),
                          "phone_side_tilt_deg": float(np.rad2deg(np.arcsin(np.clip(pr[2, 0], -1, 1)))),
                          "cradle_pos": cp.tolist(), "cradle_quat": Rotation.from_matrix(cr).as_quat().tolist(),
                          "base_tilt_deg": float(np.rad2deg(np.arccos(np.clip(br[2, 2], -1, 1)))),
                          "phone_pos": pp.tolist(), "phone_quat": Rotation.from_matrix(pr).as_quat().tolist(),
                          "phone_contacts_N": self.phone_contacts.get_contact_force_matrix(self.base.physics_dt)[0].cpu().tolist(),
                          "press_load_N": float(self.press_load),
                          "hand_contact_N": float(self.hand_contacts.get_contact_force_matrix(self.base.physics_dt).norm(dim=-1).max()),
                          "finger_force_N": np.linalg.norm(force, axis=-1).tolist(),
                          "action": action.tolist()})
        if self.phase.startswith("rear") or self.phase.startswith("left press"):
            f, points, normals, _, counts, starts = [x.cpu().numpy() for x in self.contacts.get_contact_data(self.base.physics_dt)]
            cp, cr = self.body("cradle")
            details = []
            for finger in range(4):
                start, count = int(starts[3, finger]), int(counts[3, finger])
                for j in range(start, start + count):
                    details.append({"finger": finger, "force_N": float(f[j, 0]),
                                    "point_cradle": ((points[j] - cp) @ cr).tolist(),
                                    "normal_cradle": (normals[j] @ cr).tolist()})
            self.rows[-1]["cradle_contacts"] = details
        if RECORD_VIDEO and self.record is not None:
            self.record.capture(self)

    def solve(self, side, pos, rot, grip=None):
        q, pe, re = self.ik.solve(pos, self.ik.matrix_to_quat_xyzw(rot), self.q[side],
                                self.station.BASES[side], iters=80, damping=.025, tol=2e-5)
        assert pe < .002 and re < .035, (self.phase, side, "IK error", pe, re)
        self.q[side], self.target[side] = q, (np.array(pos), np.array(rot))
        if grip is not None:
            self.grip[side] = grip

    def move(self, side, pos, rot, seconds, phase, grip=None):
        self.phase = phase
        p0, r0 = self.target[side]
        p0, r0 = p0.copy(), r0.copy()
        g0, g1 = self.grip[side], self.grip[side] if grip is None else grip
        for i in range(round(seconds * HZ)):
            s = self.ik.smoothstep((i + 1) / round(seconds * HZ))
            self.solve(side, p0 + s * (np.asarray(pos) - p0), self.ik.slerp_matrix(r0, rot, s),
                       g0 + s * (g1 - g0))
            self.tick()
        self.status()

    def transit(self, side, pos, rot, seconds, phase, grip=None):
        """Joint-space travel between clear approach poses."""
        self.phase = phase
        q0, g0 = self.q[side].copy(), self.grip[side]
        g1 = g0 if grip is None else grip
        self.solve(side, pos, rot)
        q1 = self.q[side].copy()
        for i in range(round(seconds * HZ)):
            s = self.ik.smoothstep((i + 1) / round(seconds * HZ))
            self.q[side] = q0 + s * (q1 - q0)
            self.grip[side] = g0 + s * (g1 - g0)
            self.tick()
        self.status()

    def approach_feature(self, side, body, local_pos, local_rot, seconds, phase, grip=None):
        """Track a measured grasp frame while approaching and closing the fingers."""
        self.phase = phase
        p0, r0 = self.target[side]
        p0, r0 = p0.copy(), r0.copy()
        g0, g1 = self.grip[side], self.grip[side] if grip is None else grip
        for i in range(round(seconds * HZ)):
            s = self.ik.smoothstep((i + 1) / round(seconds * HZ))
            bp, br = self.body(body)
            self.solve(side, p0 + s * (bp + br @ local_pos - p0),
                       self.ik.slerp_matrix(r0, br @ local_rot, s), g0 + s * (g1-g0))
            self.tick()
        self.status()

    def hold(self, seconds, phase):
        self.phase = phase
        for _ in range(round(seconds * HZ)):
            self.tick()
        self.status()

    def arc(self, side, center, axis, degrees, seconds, phase, lower_stop=None):
        self.phase = phase
        p0, r0 = self.target[side]
        p0, r0 = p0.copy(), r0.copy()
        for i in range(round(seconds * HZ)):
            s = self.ik.smoothstep((i + 1) / round(seconds * HZ))
            r = rotation(axis, s * degrees)
            self.solve(side, center + r @ (p0 - center), r @ r0)
            self.tick()
            if lower_stop is not None and self.angles()[1] <= lower_stop:
                break
        self.status()

    def status(self):
        print(self.phase, "t", round(self.step_id / HZ, 2), "joints", self.angles().round(2),
              "base", self.body("base")[0].round(4), "forces", self.rows[-1]["finger_force_N"], flush=True)
        print("press vector", self.last_force[0, 2:].sum(axis=0).round(3),
              "load", round(self.press_load, 2), "target height", round(self.press_height-self.station.TABLE_TOP, 4), flush=True)
        for side in self.q:
            robot = self.base.scene[side + "_robot"]
            ids = robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)[0]
            actual = robot.data.joint_pos.torch[0, ids].cpu().numpy()
            p, _ = self.ik.tcp_pose(actual, self.station.BASES[side])
            print(side, "TCP", p.round(4), "target", self.target[side][0].round(4), flush=True)
        if self.record:
            self.record.capture(self, self.phase.replace(" ", "_") + ".png")

    def unfold(self):
        from yam_arena.constants import READY

        self.hold(1.0, "initial")
        assert self.rows[-1]["holder_frame_margin_m"] > 0, "Initial holder is outside the tape frame"
        assert max(np.linalg.norm(r["phone_contacts_N"][1]) for r in self.rows) > .1, "Table contact filter did not detect the resting phone"
        b, rbase = self.body("base")
        contact_pose = np.array([0.314005669313277, 0.06508358311437586, 0.7793052041433329])
        down = np.array([
            [-5.746937261686308e-17, -1.0, 2.144786184852406e-16],
            [-0.25881904510252074, 2.220446049250313e-16, 0.9659258262890683],
            [-0.9659258262890683, 0.0, -0.25881904510252074],
        ])
        press = contact_pose - [0, 0, .008]
        self.transit("right", contact_pose + [0, 0, .030], down, 3, "press hover", grip=.5)
        self.move("right", press, down, 3, "press base")
        self.press_xy, self.press_rotation = contact_pose[:2].copy(), down.copy()
        self.press_height = press[2]
        self.press_z_min, self.press_z_max = contact_pose[2] - .06, contact_pose[2] + .003
        self.press_active = True
        self.hold(3, "base held")

        front = rotation([0, 1, 0], 40) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
        front_local = rotation([0, 0, 1], 90) @ front
        tp, tr = self.body("turntable")
        cp, cr = self.body("cradle")
        lower = tp + tr @ [0, .023, .0029]
        front_pos = cp + cr @ [0, -.022, .019]
        front = cr @ front_local
        self.transit("left", front_pos + [-.055, 0, .075], front, 3, "front approach", grip=1)
        self.approach_feature("left", "cradle", np.array([0, -.022, .019]), front_local,
                              3, "front insert")
        self.approach_feature("left", "cradle", np.array([0, -.022, .019]), front_local,
                              1, "front grip", grip=.52)
        assert np.linalg.norm(self.last_force[3, :2], axis=-1).min() > 2, "Front grasp lacks two-sided contact"
        self.target["left"] = self.tcp("left")
        tp, tr = self.body("turntable")
        lower = tp + tr @ [0, .023, .0029]
        self.arc("left", lower, tr[:, 0], -65, 6, "lift front", lower_stop=-55)
        self.hold(1, "front lifted")
        if STOP_AFTER == "front":
            return
        assert self.angles()[1] < -40, "Lower hinge did not open"

        pos, rot = self.target["left"]
        self.move("left", pos, rot, 1, "release front", grip=1)
        self.press_force = 1.0
        self.move("left", pos + [-.010, 0, .055], rotation([0, 1, 0], -15) @ rot, 2, "clear front")
        ready_pos, ready_rot = self.ik.tcp_pose(np.asarray(READY), self.station.BASES["left"])
        self.transit("left", ready_pos, ready_rot, 3, "left transfer clear")

        # Left arm presses the exposed front of the base before the right arm leaves.
        bp, _ = self.body("base")
        contact = bp + [-.034242, -.001517, .014727]
        down = rotation([0, 1, 0], 30) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
        self.transit("left", contact - down[:, 2] * .06 + [0, 0, .09], down, 3,
                     "left press orient", grip=.40)
        self.move("left", contact - down[:, 2] * .04, down, 2, "left press approach")
        self.move("left", contact + [0, 0, .005], down, 2, "left press above")
        self.move("left", contact - [0, 0, .012], down, 2, "left press contact")
        self.press_side = "left"
        self.press_xy, self.press_rotation = contact[:2].copy(), down.copy()
        self.press_nominal_rotation = down.copy()
        self.press_height = contact[2] - .012
        self.press_z_min, self.press_z_max = contact[2] - .07, contact[2] + .003
        self.press_force, self.press_load = 5.0, 0.0
        self.hold(3, "left press held")
        press_forces = np.array([r["finger_force_N"][0][:2] for r in self.rows[-HZ:]])
        assert press_forces.mean(axis=0).min() > 1, "Both left fingers must press the base"

        pos, rot = self.tcp("right")
        self.target["right"] = pos, rot
        self.move("right", pos - rot[:, 2] * .08 + [0, 0, .04], rot, 2, "right withdraw press")
        ready_pos, ready_rot = self.ik.tcp_pose(np.asarray(READY), self.station.BASES["right"])
        self.transit("right", ready_pos, ready_rot, 3, "right transfer clear")

        # Pinch the panel thickness from its right edge, with the right hand facing left.
        local_rot = (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0.]])
                     @ rotation([0, 1, 0], -35) @ rotation([0, 0, 1], 180))
        pad = np.array([0, 0, .0115])
        outside = np.array([.065, .030, .0035]) - local_rot @ pad
        inside = np.array([.020, .030, .0035]) - local_rot @ pad
        cp, cr = self.body("cradle")
        self.transit("right", cp + cr @ outside + [0, 0, .05], cr @ local_rot,
                     3, "rear grasp approach", grip=.30)
        self.approach_feature("right", "cradle", outside, local_rot, 2, "rear grasp outside")
        self.approach_feature("right", "cradle", inside, local_rot, 3, "rear grasp insert")
        self.phase = "rear grasp close"
        contact_steps = 0
        for i in range(2 * HZ):
            self.grip["right"] = .30 * (1 - self.ik.smoothstep((i + 1) / (2 * HZ)))
            self.tick()
            paired = np.linalg.norm(self.last_force[3, 2:], axis=-1).min() > 1.5
            contact_steps = contact_steps + 1 if paired else 0
            if contact_steps >= 3:
                break
        self.status()
        assert contact_steps >= 3, "Both fingers did not establish panel contact"
        self.hold(3, "rear panel held")
        rows = self.rows[-2 * HZ:]
        forces = np.array([r["finger_force_N"][3][2:] for r in rows])
        checks = {"two_fingers_contact_panel": bool(forces.mean(axis=0).min() > 1),
                  "base_stays_upright": max(r["base_tilt_deg"] for r in rows) < 5,
                  "lower_hinge_remains_open": bool(self.angles()[1] < -38)}
        review = {"scope": "grasp only; no upper-hinge lift or phone placement",
                  "checks": checks, "mean_right_panel_contact_N": forces.mean(axis=0).tolist(),
                  "final_holder_angles_deg": self.angles().tolist()}
        (OUT / "grasp-review.json").write_text(json.dumps(review, indent=2) + "\n")
        print(json.dumps(review, indent=2), flush=True)
        assert all(checks.values()), checks
        if STOP_AFTER == "grasp":
            return
        self.lift_panel()

    def lift_panel(self):
        self.press_force = 10.0
        self.hold(2, "rear support preload")
        center, cr = self.body("cradle")
        p0, r0 = self.tcp("right")
        start_grip = self.grip["right"]
        self.phase = "rear panel lift"
        for i in range(8 * HZ):
            turn = rotation(cr[:, 0], 127 * self.ik.smoothstep(min(1, (i + 1) / (6 * HZ))))
            self.solve("right", center + turn @ (p0 - center), turn @ r0)
            contact = np.linalg.norm(self.last_force[3, 2:], axis=-1).min()
            self.grip["right"] = np.clip(self.grip["right"] + .0005 * (contact - 1) / HZ,
                                         start_grip, start_grip + .04)
            self.tick()
            assert self.body("base")[0][2] < self.station.TABLE_TOP + .008, "Base lifted during panel rotation"
            if self.panel_angle() >= 55:
                break
        self.status()
        self.straighten_holder()
        pos, rot = self.target["right"]
        self.move("right", pos, rot, 1, "rear release panel", grip=.30)
        pos, rot = self.tcp("right")
        self.target["right"] = pos, rot
        _, cr = self.body("cradle")
        self.move("right", pos - rot[:, 2] * .06, rotation(cr[:, 0], -25) @ rot, 2, "rear hand clear")
        self.hold(1, "panel opened")
        assert self.panel_angle() >= 47, "Panel is below 47 degrees above the table"

    def place_phone(self):
        from yam_arena.constants import READY
        from yam_arena.demo import grip_rotation

        self.press_active = False
        pos, rot = self.tcp("left")
        self.target["left"] = pos, rot
        self.move("left", pos - rot[:, 2] * .08, rot, 3, "release base")
        ready_pos, ready_rot = self.ik.tcp_pose(np.asarray(READY), self.station.BASES["left"])
        self.transit("left", ready_pos, ready_rot, 3, "left clear")
        self.hold(2, "stand free")
        assert self.angles()[1] < -38 and self.panel_angle() >= 47, self.angles()

        ready_pos, ready_rot = self.ik.tcp_pose(np.asarray(READY), self.station.BASES["right"])
        self.transit("right", ready_pos, ready_rot, 3, "right prepare phone")

        phone_pos, _ = self.phone_pose()
        hover = np.r_[phone_pos[:2], self.station.TABLE_TOP + .12]
        down = grip_rotation()
        self.transit("right", hover, down, 3, "phone approach", grip=1)
        pick = np.r_[phone_pos[:2], self.station.TABLE_TOP + .017]
        self.move("right", pick, down, 3, "phone descend")
        self.move("right", pick, down, 1.5, "phone grip", grip=.65)
        self.move("right", hover, down, 3, "phone lift")
        pp, pr = self.phone_pose()
        assert pp[2] > self.station.TABLE_TOP + .08, "Phone was not lifted"
        tp, tr = self.tcp("right")
        grasp_pos, grasp_rot = pr.T @ (tp - pp), pr.T @ tr

        cp, cr = self.body("cradle")
        goal = np.array([0, .0574, .0099])

        def place_at(local, seconds, phase):
            desired_phone = cp + cr @ local
            self.move("right", desired_phone + cr @ grasp_pos, cr @ grasp_rot, seconds, phase)

        place_at(goal + [0, .045, .025], 5, "phone align")
        place_at(goal + [0, .045, 0], 2, "phone above tray")
        place_at(goal, 4, "phone into tray")
        pos, rot = self.target["right"]
        self.move("right", pos, rot, 1.5, "phone release", grip=1)
        self.move("right", pos - rot[:, 2] * .08, rot, 3, "phone clear")
        ready_pos, ready_rot = self.ik.tcp_pose(np.asarray(READY), self.station.BASES["right"])
        self.transit("right", ready_pos, ready_rot, 3, "right clear")
        self.release_time = self.step_id / HZ
        self.hold(10, "hands off")
        self.validate()

    def straighten_holder(self):
        center, _ = self.body("turntable")
        pos, rot = self.tcp("right")
        self.target["right"] = pos, rot
        yaw_command, aligned_steps = 0.0, 0
        # Allow for the measured yaw relaxation during release and phone seating.
        release_yaw_allowance = 2.5
        self.phase = "holder straighten"
        for _ in range(6 * HZ):
            _, cr = self.body("cradle")
            error = self.heading_error(cr) - release_yaw_allowance
            yaw_command += np.clip(-1.5 * error / HZ, -.4, .4)
            assert abs(yaw_command) < 30, "Gripper rotated without bringing the holder straight"
            turn = rotation([0, 0, 1], yaw_command)
            self.solve("right", center + turn @ (pos - center), turn @ rot)
            self.tick()
            aligned_steps = aligned_steps + 1 if abs(error) < .5 else 0
            if aligned_steps >= 10:
                break
        self.status()
        assert abs(self.heading_error(self.body("cradle")[1]) - release_yaw_allowance) < .5, "Holder did not straighten"

    def validate(self):
        surface = np.load(ROOT / "assets/iphone17pro/surface_points.npy")
        result = score_placement(self.rows[-1], surface, self.station.TABLE_TOP, self.step_id / HZ)
        (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        assert result["success"], result["checks"]


def run():
    from camera_ready import Recorder
    from yam_scene import make_env

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("expert.py", "yam_scene.py", "camera_ready.py", "scoring.py"):
        (OUT / name).write_text((ROOT / name).read_text())
    env = make_env(control_hz=HZ, enable_cameras=RECORD_VIDEO, log_dir=OUT / "logs", use_fabric=True,
                   device="cpu", physics_dt=1 / 480)
    expert = None
    try:
        env.reset()
        expert = Expert(env)
        expert.record = Recorder(env, OUT, RECORD_VIDEO)
        expert.unfold()
        if STOP_AFTER == "all":
            expert.place_phone()
    finally:
        if expert:
            (OUT / "trajectory.json").write_text(json.dumps(expert.rows) + "\n")
            if expert.record:
                expert.record.close()
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        app.close(exit_code=1)
        raise
    app.close()

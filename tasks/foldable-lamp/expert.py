"""State-informed dual-arm policy. Only the 14-D robot action moves scene objects."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
HZ = 30
RECORD_VIDEO = True
ROBOT_CAMERAS = (("left_cam", "Left Wrist Camera"), ("top_cam", "Top Camera"),
                 ("right_cam", "Right Wrist Camera"))


def rot(axis, degrees):
    return Rotation.from_euler(axis, degrees, degrees=True).as_matrix()


def pose(value):
    return value[:3].copy(), Rotation.from_quat(value[3:]).as_matrix()


class Expert:
    def __init__(self, env, out):
        import carb
        import omni.physics.tensors as tensors
        import torch
        from isaaclab.sim.utils import get_current_stage_id
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from pxr import PhysxSchema, UsdGeom
        from yam_arena import ik, station
        from yam_arena.constants import READY
        from yam_arena.robot import asset_link_paths
        from touch import TouchSwitch

        self.env, self.base, self.out = env, env.unwrapped, out
        self.ik, self.station, self.torch = ik, station, torch
        self.q = {s: np.array(READY, float) for s in ("left", "right")}
        self.grip = {s: 1. for s in self.q}
        self.target = {s: ik.tcp_pose(q, station.BASES[s]) for s, q in self.q.items()}
        self.lamp = self.base.scene["foldable_lamp"]
        self.links = {n: i for i, n in enumerate(self.lamp.body_names)}
        self.joints = {n: i for i, n in enumerate(self.lamp.joint_names)}
        self.head_vertices = np.concatenate([np.asarray(UsdGeom.Mesh(self.base.sim.stage.GetPrimAtPath(
            f"/World/envs/env_0/foldable_lamp/head/{name}")).GetPointsAttr().Get(), dtype=float)
            for name in ("shell", "diffuser")])
        self.rows, self.phase, self.step = [], "settle", 0
        self.record = None
        self.base_anchor = None
        settings = carb.settings.get_settings()
        settings.set_bool("/physics/disableContactProcessing", False)
        settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
        assert not settings.get("/rtx/post/histogram/enabled"), "Brightness comparison requires fixed exposure"
        self.switch = TouchSwitch(self.base.sim.stage, "/World/envs/env_0/foldable_lamp")
        self.subscription = get_physx_interface().subscribe_physics_step_events(self.switch.update)
        self.physics = tensors.create_simulation_view("torch", stage_id=get_current_stage_id())
        self.physics.set_subspace_roots("/")
        paths = asset_link_paths()
        self.arm_contacts = []
        self.head_body_contacts = {}
        self.arm_report_headers = 0
        for side in ("Left", "Right"):
            for path in paths.values():
                prim = self.base.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{side}Arm{path}")
                assert prim.HasAPI(PhysxSchema.PhysxContactReportAPI), str(prim.GetPath())
                assert PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr().Get() == 0
        self.arm_subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._arm_contact)
        fingers = [f"/World/envs/env_0/{s}Arm{paths[t]}"
                   for s in ("Left", "Right") for t in ("tip_left", "tip_right")]
        self.contacts = {n: self.physics.create_rigid_contact_view(
            f"/World/envs/env_0/foldable_lamp/{n}", fingers) for n in self.links}

    def _arm_contact(self, headers, data):
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools

        left, right = "/World/envs/env_0/LeftArm/", "/World/envs/env_0/RightArm/"
        for h in headers:
            if h.type == ContactEventType.CONTACT_LOST:
                continue
            a, b = (str(PhysicsSchemaTools.intToSdfPath(p)) for p in (h.actor0, h.actor1))
            if a.startswith((left, right)) or b.startswith((left, right)):
                self.arm_report_headers += 1
            for head, robot in ((a, b), (b, a)):
                if head.endswith("/foldable_lamp/head") and robot.startswith((left, right)) and "/tip_" not in robot:
                    points = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
                    force = sum(np.linalg.norm(p.impulse) for p in points) / self.base.physics_dt
                    if force > .1:
                        key = self.phase + " | " + robot
                        if key not in self.head_body_contacts:
                            print("HEAD BODY CONTACT", key, force, flush=True)
                        self.head_body_contacts[key] = max(self.head_body_contacts.get(key, 0.), float(force))
            if not ((a.startswith(left) and b.startswith(right)) or
                    (a.startswith(right) and b.startswith(left))):
                continue
            points = [data[i] for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data)]
            if points:
                self.arm_contacts.append({"time_s": self.switch.time, "phase": self.phase,
                    "bodies": [a, b], "contact_points": len(points),
                    "force_N": sum(np.linalg.norm(p.impulse) for p in points) / self.base.physics_dt,
                    "min_separation_m": min(p.separation for p in points)})

    def body(self, name):
        return pose(self.lamp.data.body_link_pose_w.torch[0, self.links[name]].cpu().numpy())

    def tcp(self, side):
        from yam_arena.constants import TCP_OFFSET
        robot = self.base.scene[side + "_robot"]
        index = robot.find_bodies("gripper")[0][0]
        p, r = pose(robot.data.body_link_pose_w.torch[0, index].cpu().numpy())
        return p + r @ TCP_OFFSET, r

    def angles(self):
        values = self.lamp.data.joint_pos.torch[0].cpu().numpy()
        return np.rad2deg(values[[self.joints[n] for n in ("lower", "middle", "upper")]])

    def tick(self):
        if self.base_anchor is not None:
            p, r = self.body("base")
            goal_p, goal_r = self.base_anchor
            correction = Rotation.from_rotvec(.15 * Rotation.from_matrix(goal_r @ r.T).as_rotvec()).as_matrix()
            tp, tr = self.target["right"]
            self.solve("right", p + .15*(goal_p-p) + correction @ (tp-p), correction @ tr)
        action = np.r_[self.q["left"], self.grip["left"], self.q["right"], self.grip["right"]]
        self.base.action_manager.process_action(self.torch.tensor(action[None], dtype=self.torch.float32, device=self.base.device))
        for _ in range(self.base.cfg.decimation):
            self.base.action_manager.apply_action()
            self.base.scene.write_data_to_sim()
            self.base.sim.step(render=False)
            self.base.scene.update(self.base.physics_dt)
        self.step += 1
        p, r = self.body("base")
        q = self.angles()
        assert np.isfinite(np.r_[p, q]).all(), "Non-finite lamp state"
        self.rows.append({"time_s": self.step / HZ, "phase": self.phase, "action": action.tolist(),
            "joints_deg": q.tolist(), "base_pos": p.tolist(),
            "base_quat": Rotation.from_matrix(r).as_quat().tolist(),
            "book_pos": self.base.scene["book"].data.root_link_pose_w.torch[0, :3].cpu().tolist(),
            "base_tilt_deg": float(np.rad2deg(np.arccos(np.clip(r[2, 2], -1, 1)))),
            "lamp_on": self.switch.state.Get(),
            "tcp": {s: self.tcp(s)[0].tolist() for s in self.q},
            "tcp_quat": {s: Rotation.from_matrix(self.tcp(s)[1]).as_quat().tolist() for s in self.q},
            "finger_force_N": {n: np.linalg.norm(v.get_contact_force_matrix(self.base.physics_dt)
                .cpu().numpy()[0], axis=-1).tolist() for n, v in self.contacts.items()}})
        if self.record:
            self.record.capture(self)
        if self.step % HZ == 0:
            print("FRAME", self.step, self.phase, flush=True)
        if self.arm_contacts:
            if self.record:
                self.record.capture(self, "arm_collision.png")
            raise AssertionError(("Inter-arm contact", self.arm_contacts[0]))
        assert p[2] - self.station.TABLE_TOP < .0025, (self.phase, "Base lifted", p[2])
        assert self.rows[-1]["base_tilt_deg"] < 2., (self.phase, "Base tilted")

    def solve(self, side, p, r, seed=None):
        q, pe, re = self.ik.solve(p, self.ik.matrix_to_quat_xyzw(r), self.q[side] if seed is None else seed,
            self.station.BASES[side], iters=100, damping=.02, tol=2e-5)
        assert pe < .002 and re < .035, (self.phase, side, "IK unreachable", pe, re)
        self.q[side], self.target[side] = q, (np.array(p), np.array(r))

    def move(self, side, p, r, seconds, phase, grip=None, transit=False, seed=None):
        self.phase = phase
        print("START", phase, flush=True)
        p0, r0 = self.target[side]
        p0, r0 = p0.copy(), r0.copy()
        q0, g0 = self.q[side].copy(), self.grip[side]
        g1 = g0 if grip is None else grip
        if transit:
            self.solve(side, p, r, seed)
            q1 = self.q[side].copy()
        count = round(seconds * HZ)
        for i in range(count):
            s = self.ik.smoothstep((i+1) / count)
            if transit:
                self.q[side] = q0 + s * (q1-q0)
            else:
                self.solve(side, p0 + s * (np.asarray(p)-p0), self.ik.slerp_matrix(r0, r, s))
            self.grip[side] = g0 + s * (g1-g0)
            self.tick()
        self.status()

    def hold(self, seconds, phase):
        self.phase = phase
        for _ in range(round(seconds * HZ)):
            self.tick()
        self.status()

    def status(self):
        row = self.rows[-1]
        print(self.phase, "t", row["time_s"], "joints", np.round(row["joints_deg"], 2),
              "base", np.round(row["base_pos"], 4), "forces", row["finger_force_N"], flush=True)
        for s in self.q:
            print(s, "TCP", self.tcp(s)[0].round(4), "target", self.target[s][0].round(4), flush=True)
        if self.record:
            self.record.capture(self, self.phase.replace(" ", "_") + ".png")

    def head_pose(self, angles, bp, br):
        a, b, c = angles
        p = bp + br @ (np.array([0, .053, .028])
            + rot("x", 180 + a) @ [0, .106, -.006]
            + rot("x", a + b) @ [0, .106, 0])
        return p, br @ rot("x", 180 + a + b + c)

    def articulate(self, goal, seconds, phase):
        self.phase = phase
        print("START", phase, flush=True)
        start = self.angles()
        bp, br = self.body("base")
        hp, hr = self.body("head")
        tp, tr = self.target["left"]
        offset, relative_grip = hr.T @ (tp-hp), hr.T @ tr
        count = round(seconds * HZ)
        for i in range(count):
            s = self.ik.smoothstep((i + 1) / count)
            p, r = self.head_pose(start + s * (np.asarray(goal) - start), bp, br)
            self.solve("left", p + r @ offset, r @ relative_grip)
            self.tick()
        self.hold(.8, phase + " held")

    def park(self, side):
        from yam_arena.constants import READY
        q = np.array(READY, float)
        p, r = self.ik.tcp_pose(q, self.station.BASES[side])
        self.move(side, p, r, 3., side + " retreat", grip=1., transit=True, seed=q)

    def tap_with_right(self):
        from yam_arena.demo import grip_rotation

        bp, br = self.body("base")
        button = bp + br @ [0, -.035, .0189]
        tap_r = br @ rot("z", 90) @ rot("y", -65) @ grip_rotation()
        tap_p = button - tap_r @ [0, .01369, .01284]
        self.move("right", tap_p - tap_r[:, 2] * .08, tap_r, 3., "right touch approach",
                  grip=0., transit=True, seed=[.7, 2.4, 1.7, .1, 0, 0])
        self.move("right", tap_p + [0, 0, .006], tap_r, 2., "right above switch")
        assert not self.switch.state.Get(), "Unexpected switch contact before tap"
        self.record.capture(self, "before_touch.png")
        self.phase = "right touch switch"
        for i in range(40):
            self.solve("right", tap_p + [0, 0, .006 - (i+1) * .0004], tap_r)
            self.tick()
            if self.switch.state.Get():
                break
        self.status()
        assert self.switch.state.Get(), "Right touch did not turn the lamp on"
        self.move("right", tap_p + [0, 0, .006], tap_r, .5, "right lift off switch")
        self.record.capture(self, "after_touch.png")
        p, r = self.target["right"]
        self.move("right", p - r[:, 2] * .08, r, 2., "right release switch")

    def placement(self):
        from yam_scene import BASE_HALF_SIZE, BOOK_XY, TARGET_MARGIN, TARGET_XY, TARGET_YAW

        p, r = self.body("base")
        corners = np.array([[x, y, .01] for x in (-BASE_HALF_SIZE[0], BASE_HALF_SIZE[0])
                            for y in (-BASE_HALF_SIZE[1], BASE_HALF_SIZE[1])])
        local = (corners @ r.T + p - [*TARGET_XY, self.station.TABLE_TOP]) @ rot("z", TARGET_YAW)
        clearance = np.array(BASE_HALF_SIZE) + TARGET_MARGIN - np.max(np.abs(local[:, :2]), axis=0)
        yaw = np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))
        yaw_error = (yaw - TARGET_YAW + 180) % 360 - 180
        return {"target_xy_m": list(TARGET_XY), "target_yaw_deg": TARGET_YAW,
                "margin_m": TARGET_MARGIN, "base_xy_m": p[:2].tolist(),
                "offset_from_book_xy_m": (p[:2]-BOOK_XY).tolist(),
                "footprint_inside": bool(np.all(clearance >= 0)),
                "edge_clearance_xy_m": clearance.tolist(), "yaw_error_deg": float(yaw_error)}

    def alignment(self):
        return {"second_segment_from_vertical_deg": float(np.rad2deg(np.arccos(np.clip(
                    self.body("upper_arm")[1][2, 1], -1, 1)))),
                "lamp_face_from_horizontal_deg": float(np.rad2deg(np.arccos(np.clip(
                    -self.body("head")[1][2, 2], -1, 1))))}

    def head_height(self):
        p, r = self.body("head")
        return float(np.min((self.head_vertices @ r.T + p)[:, 2]) - self.station.TABLE_TOP)

    def beam(self):
        p, r = self.body("head")
        source, direction = p + r @ [0, .054, -.006], r[:, 2]
        book = self.base.scene["book"].data.root_link_pose_w.torch[0].cpu().numpy()
        assert direction[2] < -.2, "Lamp faces away from table"
        return source + direction * (book[2] + .020 - source[2]) / direction[2]

    def execute(self):
        from yam_arena.demo import grip_rotation
        from yam_scene import LAMP_MAX_TILT, LAMP_MIN_HEIGHT, TARGET_XY, TASK_SECONDS

        self.hold(1., "initial")
        initial_book = self.base.scene["book"].data.root_link_pose_w.torch[0].cpu().numpy().copy()
        p, r = self.body("base")
        grasp = p + r @ np.array([0, -.030, .024])
        approach = rot("y", -60) @ grip_rotation()
        self.move("right", [.370, .18, .8115], approach, 3., "right low approach", transit=True,
                  seed=[1, 2.4, 1.7, .1, 0, 0])
        self.move("right", grasp, approach, 3., "right descend")
        self.hold(.6, "right at base")
        self.move("right", grasp, approach, 1.5, "right close base", grip=.50)
        self.hold(1., "base grasp")
        assert min(self.rows[-1]["finger_force_N"]["base"][2:]) > 2, "Base grasp missing"

        # Orient the still-folded lamp on the table before bringing it closer.
        final_xy = np.array(TARGET_XY)
        bp, br = self.body("base")
        direction, front = initial_book[:2] - final_xy, -br[:2, 1]
        delta = np.rad2deg(np.arctan2(direction[1], direction[0]) - np.arctan2(front[1], front[0]))
        delta = (delta + 180) % 360 - 180
        p0, r0 = self.target["right"]
        self.phase = "orient folded lamp"
        print("START", self.phase, "yaw", delta, flush=True)
        for i in range(120):
            turn = rot("z", delta * self.ik.smoothstep((i + 1) / 120))
            self.solve("right", bp + turn @ (p0-bp), turn @ r0)
            self.tick()
        self.hold(.5, "folded lamp oriented")
        assert np.max(np.abs(self.angles())) < 2., "Lamp unfolded during orientation"
        self.folded_yaw_deg = float(delta)
        bp, br = self.body("base")
        p, r = self.target["right"]
        self.move("right", p + np.r_[final_xy-bp[:2], 0.], r, 4., "pull folded lamp closer")
        self.hold(.7, "folded lamp positioned")
        p, r = self.target["right"]
        self.move("right", p + [0, 0, -.003], r, 1., "steady base on table")
        from yam_arena.constants import READY
        assert all(np.array_equal(row["action"][:6], READY) for row in self.rows), "Left arm moved before receiving lamp"
        self.left_start_time = self.step / HZ

        # Descend vertically from above the head, then lift before setting its pitch.
        head_offset = np.array([0, .060, -.012])
        hp, hr = self.body("head")
        head_grasp = hp + hr @ head_offset
        br = self.body("base")[1]
        head_approach = rot("z", np.rad2deg(np.arctan2(br[1, 0], br[0, 0]))) @ rot("x", 180) @ rot("y", 10)
        self.move("left", head_grasp + [0, 0, .08], head_approach, 3., "left above head", transit=True,
                  seed=[-.3, 1.8, 1.3, -.3, 0, -.3])
        self.move("left", head_grasp, head_approach, 3., "left vertical descend")
        self.hold(.6, "left at head")
        self.move("left", head_grasp, head_approach, 1.5, "left close head", grip=.70)
        self.hold(.7, "head grasp")
        assert min(self.rows[-1]["finger_force_N"]["head"][:2]) > 2, "Head grasp missing"

        p, r = self.target["right"]
        self.move("right", p, r, 1., "clear head while holding base", grip=.62)
        assert min(self.rows[-1]["finger_force_N"]["base"][2:]) > 2, "Base grasp missing"
        bp, br = self.body("base")
        self.base_anchor = (np.r_[bp[:2], self.station.TABLE_TOP],
                            rot("z", np.rad2deg(np.arctan2(br[1, 0], br[0, 0]))))

        # Open the folded links by lifting straight up at the forward grasp point.
        p, r = self.target["left"]
        self.lift_start = self.tcp("left")[0]
        self.move("left", p + [0, 0, .15], r, 7., "lift head vertically")
        self.hold(.5, "head raised")
        assert min(self.rows[-1]["finger_force_N"]["head"][:2]) > 2, "Head slipped during lift"
        # The right hand keeps the base throughout both final hinge adjustments.
        p, r = self.target["left"]
        self.move("left", p, r, 1., "release thin edge grip", grip=1.)
        self.move("left", p-r[:, 2]*.01, r, 1., "clear lamp face")
        self.move("left", [.31, .23, 1.0], r, 2., "raise hand for face grip")
        hp, hr = self.body("head")
        # Opposed contact on the broad faces transmits the head-hinge torque.
        r = hr @ np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        self.move("left", [.31, .23, 1.0], r, 2., "turn hand for face grip", grip=.20)
        hp, hr = self.body("head")
        r = hr @ np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        p = hp + hr @ head_offset
        self.move("left", p-r[:, 2]*.10, r, 2., "approach broad faces")
        self.move("left", p, r, 2., "straddle lamp face")
        self.move("left", p, r, 1.5, "grasp broad faces", grip=0.)
        self.hold(.5, "face grasp")
        assert min(self.rows[-1]["finger_force_N"]["head"][:2]) > 2, "Face grasp missing"
        angles = self.angles()
        self.articulate([-65, 155, sum(angles)-90], 4., "stand second segment upright")
        lower = self.angles()[0]
        self.articulate([lower, 90-lower, -90], 4., "level lamp face")
        self.hold(1., "lamp aligned")
        p, r = self.target["left"]
        self.move("left", p, r, 1., "release head", grip=1.)
        self.move("left", p - r[:, 2] * .06, r, 1., "left clear")
        self.park("left")

        assert min(self.rows[-1]["finger_force_N"]["base"][2:]) > 2, "Right hand lost base before switch"
        self.right_release_time = self.step / HZ
        self.base_anchor = None
        p, r = self.tcp("right")
        self.move("right", p, r, 1., "release base", grip=1.)
        self.move("right", p - r[:, 2] * .07, r, 2., "right clear for switch")
        self.tap_with_right()
        self.park("right")
        self.hold(3., "book illuminated")
        self.record.capture(self, "final.png")
        beam = self.beam()
        book = self.base.scene["book"].data.root_link_pose_w.torch[0].cpu().numpy()
        placement, alignment = self.placement(), self.alignment()
        assert placement["footprint_inside"], placement
        height = self.head_height()
        assert height >= LAMP_MIN_HEIGHT, ("Lamp shade too low", height)
        assert alignment["lamp_face_from_horizontal_deg"] <= LAMP_MAX_TILT, alignment
        assert self.step / HZ <= TASK_SECONDS, "Task exceeded time limit"
        assert self.switch.state.Get(), "Lamp is off"
        recent = self.rows[-60:]
        brightness = [self.record.book_luminance(name, book) for name in ("before_touch.png", "after_touch.png")]
        lift = np.array([row["tcp"]["left"] for row in self.rows if row["phase"] == "lift head vertically"])
        lift_xy = float(np.max(np.linalg.norm(lift[:, :2] - self.lift_start[:2], axis=1)))
        lift_z = float(lift[-1, 2] - self.lift_start[2])
        hold_force = min(min(row["finger_force_N"]["base"][2:]) for row in self.rows
                         if self.left_start_time < row["time_s"] <= self.right_release_time)
        self.metrics = {"score": 1, "beam_intersection_m": beam.tolist(), "book_pose": book.tolist(),
                        "lamp_bottom_height_m": height,
                        "left_grasp_forward_shift_m": .025, "left_lift_actual_z_m": lift_z,
                        "left_lift_max_xy_deviation_m": lift_xy,
                        "right_base_min_finger_force_until_release_N": hold_force,
                        "right_release_time_s": self.right_release_time,
                        "placement": placement, "alignment": alignment,
                        "book_displacement_m": float(np.linalg.norm(book[:3] - initial_book[:3])),
                        "final_joints_deg": self.angles().tolist(), "diagnostic_window_seconds": 2.,
                        "left_default_until_s": self.left_start_time, "folded_yaw_deg": self.folded_yaw_deg,
                        "max_base_lift_m": max(row["base_pos"][2] for row in self.rows) - self.station.TABLE_TOP,
                        "max_base_tilt_deg": max(row["base_tilt_deg"] for row in self.rows),
                        "book_luminance_off_on": brightness,
                        "book_brightness_increase_percent": 100 * (brightness[1] / brightness[0] - 1),
                        "brightness_metric": "Mean weighted 8-bit display RGB; not illuminance in lux",
                        "joint_drift_deg": np.ptp([row["joints_deg"] for row in recent], axis=0).tolist(),
                        "base_drift_m": np.ptp([row["base_pos"] for row in recent], axis=0).tolist()}


class Recorder:
    def __init__(self, stage, out):
        import imageio_ffmpeg
        import omni.replicator.core as rep
        from PIL import ImageFont
        from pxr import Gf, Sdf, UsdGeom

        self.out, self.size, self.focal = out, (1920, 1080), 35.
        self.eye, self.lookat = np.array([1.10, -.90, 1.55]), np.array([.25, .04, .97])
        camera = UsdGeom.Camera.Define(stage, "/ExpertCamera")
        camera.GetPrim().CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*self.eye), Gf.Vec3d(*self.lookat), Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.CreateFocalLengthAttr(self.focal)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(36 * self.size[1] / self.size[0])
        camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 30))
        self.product = rep.create.render_product(str(camera.GetPath()), self.size)
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb.attach([self.product])
        self.font = ImageFont.load_default(size=32)
        self.subtitle_font = ImageFont.load_default(size=24)
        self.phase_font = ImageFont.load_default(size=16)
        (out / "videos").mkdir()
        self.writer = None
        if RECORD_VIDEO:
            self.writer = imageio_ffmpeg.write_frames(str(out / "videos/expert.mp4"), (1920, 1440),
                fps=HZ, codec="libx264", quality=None, pix_fmt_out="yuv420p", macro_block_size=2,
                output_params=["-crf", "18", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"])
            self.writer.send(None)

    def capture(self, expert, filename=None):
        if filename is None and not RECORD_VIDEO:
            return
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        from PIL import Image, ImageDraw

        expert.base.sim.render()
        ensure_isaac_rtx_render_update(force=True)
        pixels = self.rgb.get_data()
        assert pixels.shape == (1080, 1920, 4), pixels.shape
        observer = Image.fromarray(np.ascontiguousarray(pixels[..., :3]))
        frame = Image.new("RGB", (1920, 1440))
        frame.paste(observer)
        robot_frames = {}
        for i, (name, _) in enumerate(ROBOT_CAMERAS):
            camera = expert.base.scene[name]
            camera.update(1/HZ, force_recompute=True)
            pixels = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy()
            assert pixels.shape == (360, 640, 3), (name, pixels.shape)
            robot_frames[name] = Image.fromarray(pixels)
            frame.paste(robot_frames[name], (640*i, 1080))
        draw = ImageDraw.Draw(frame)
        labels = [(0, 0, "External Observer View"),
                  *((640*i, 1080, title) for i, (_, title) in enumerate(ROBOT_CAMERAS))]
        for x, y, title in labels:
            width = draw.textlength(title, font=self.font) + 32
            draw.rectangle((x, y, x+width, y+(84 if y == 0 else 52)), fill=(20, 25, 32))
            draw.text((x+16, y+10), title, font=self.font, fill="white")
        draw.text((16, 50), "Not a robot camera", font=self.subtitle_font, fill=(175, 215, 240))
        draw.rectangle((0, 1050, 1920, 1079), fill=(20, 25, 32))
        draw.text((12, 1055), f"{expert.step / HZ:.1f}s | {expert.phase} | Lamp {'ON' if expert.switch.state.Get() else 'OFF'}",
                  font=self.phase_font, fill="white")
        if filename:
            observer.save(self.out / filename)
            stem = Path(filename).stem
            frame.save(self.out / (stem + "_views.png"))
            for name, picture in robot_frames.items():
                picture.save(self.out / (stem + "_" + name + ".png"))
        else:
            self.writer.send(np.asarray(frame))

    def close(self):
        if self.writer:
            self.writer.close()
        self.rgb.detach()
        self.product.destroy()

    def book_luminance(self, filename, book_pose):
        """Compare the same inset cover region in the two actual rendered frames."""
        from PIL import Image, ImageDraw

        forward = self.lookat - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        p, r = pose(book_pose)
        pixels = []
        for x, y in ((-.06, -.03), (.06, -.03), (.06, .03), (-.06, .03)):
            d = p + r @ [x, y, .020] - self.eye
            pixels.append((self.size[0]/2 + self.focal / 36 * self.size[0] * (d @ right) / (d @ forward),
                           self.size[1]/2 - self.focal / 36 * self.size[0] * (d @ up) / (d @ forward)))
        mask = Image.new("L", self.size)
        ImageDraw.Draw(mask).polygon(pixels, fill=255)
        pixels = np.asarray(Image.open(self.out / filename))
        rgb = pixels[np.asarray(mask) > 0, :3].astype(np.float64)
        return float(np.mean(rgb @ [.2126, .7152, .0722]))


def run():
    import carb
    import shutil
    from pxr import UsdLux
    from yam_scene import ENVIRONMENT_LIGHT_SCALE, make_env

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / f"demo-{stamp}"
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / "expert.py")
    shutil.copy2(ROOT / "yam_scene.py", out / "yam_scene.py")
    print("OUTPUT", out, flush=True)
    env = make_env(control_hz=HZ, enable_cameras=True, log_dir=out / "logs")
    expert = None
    result = {"success": False}
    try:
        env.reset()
        lights = {str(p.GetPath()): {"type": p.GetTypeName(),
                  "intensity": UsdLux.LightAPI(p).GetIntensityAttr().Get(),
                  "exposure": UsdLux.LightAPI(p).GetExposureAttr().Get()}
                  for p in env.unwrapped.sim.stage.Traverse() if p.HasAPI(UsdLux.LightAPI)}
        assert sorted(v["intensity"] for v in lights.values() if v["type"] != "RectLight") == [125., 300000.], lights
        settings = carb.settings.get_settings()
        result["lighting"] = {"environment_scale": ENVIRONMENT_LIGHT_SCALE, "lights_initial": lights,
            "lamp_on_intensity": 50000.,
            "renderer_settings": {p: settings.get(p) for p in (
                "/rtx/post/histogram/enabled", "/rtx/post/tonemap/op", "/rtx/post/tonemap/filmIso",
                "/rtx/post/tonemap/exposureTime", "/rtx/post/tonemap/fNumber",
                "/rtx/post/tonemap/enableSrgbToGamma", "/rtx/post/tonemap/colorMode")}}
        print("LIGHTING", json.dumps(result["lighting"]), flush=True)
        expert = Expert(env, out)
        expert.record = Recorder(expert.base.sim.stage, out)
        expert.execute()
        result["completed_phase"] = expert.phase
        result["success"] = True
        result.update(expert.metrics)
        result["recording"] = {"fps": HZ, "frames": expert.step, "render_mode": settings.get("/rtx/rendermode"),
            "video_recorded": RECORD_VIDEO,
            "observer_native_size": [1920, 1080], "observer_is_robot_camera": False,
            "robot_native_size": [640, 360], "video_size": [1920, 1440],
            "robot_intrinsics": {name: expert.base.scene[name].data.intrinsic_matrices.torch[0].cpu().tolist()
                                 for name, _ in ROBOT_CAMERAS}}
    except Exception as exc:
        result["error"] = repr(exc)
        if expert and expert.record:
            expert.record.capture(expert, "failure.png")
        raise
    finally:
        if expert:
            (out / "trajectory.json").write_text(json.dumps(expert.rows) + "\n")
            result["touch_events"] = expert.switch.events
            result["inter_arm_contact_count"] = len(expert.arm_contacts)
            result["robot_contact_report_headers"] = expert.arm_report_headers
            result["contact_check_hz"] = 1 / expert.base.physics_dt
            (out / "arm_contacts.json").write_text(json.dumps(expert.arm_contacts, indent=2) + "\n")
            (out / "head_body_contacts.json").write_text(json.dumps(expert.head_body_contacts, indent=2) + "\n")
            expert.arm_subscription = None
            expert.subscription = None
            expert.switch.close()
            if expert.record:
                expert.record.close()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    import traceback
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True,
        kit_args="--/carb/tasking/plugin/threadCount=12 --/omni/tbb/globalcontrol/maxThreadCount=12").app
    try:
        run()
    except Exception:
        traceback.print_exc()
        app.close(exit_code=1)
    else:
        app.close()

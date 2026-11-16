"""Physical key events, tape measurement progress and calculator scoring."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


class Calculator:
    def __init__(self):
        self.text = "0"
        self.lhs = None
        self.operator = None
        self.fresh = True
        self.submitted = None
        self.ended = False

    def show(self, number):
        if not math.isfinite(number) or abs(number) >= 1e8:
            self.text = "Error"
            return
        self.text = format(number, ".8g")
        if "e" in self.text or len(self.text.replace(".", "")) > 8:
            self.text = "Error"

    def apply(self):
        rhs = float(self.text)
        if self.operator == "plus":
            self.show(self.lhs+rhs)
        elif self.operator == "minus":
            self.show(self.lhs-rhs)
        elif self.operator == "multiply":
            self.show(self.lhs*rhs)
        elif self.operator == "divide":
            if rhs == 0:
                self.text = "Error"
            else:
                self.show(self.lhs/rhs)
        self.operator = None

    def press(self, key):
        if self.ended:
            return
        if key in ("C", "CE"):
            self.text, self.fresh = "0", True
            if key == "C":
                self.lhs, self.operator = None, None
        elif key.isdigit() or key == "dot":
            char = "." if key == "dot" else key
            if self.fresh or self.text == "Error":
                self.text, self.fresh = "0", False
            if char == "." and "." in self.text:
                return
            candidate = char if self.text == "0" and char != "." else self.text+char
            if len(candidate.replace(".", "").replace("-", "")) <= 8:
                self.text = candidate
        elif key == "equals":
            if self.text != "Error":
                self.apply()
            self.submitted = self.text
            self.ended = True
        elif self.text != "Error":
            if key in ("plus", "minus", "multiply", "divide"):
                self.apply()
                if self.text != "Error":
                    self.lhs, self.operator, self.fresh = float(self.text), key, True
            elif key == "sign":
                self.show(-float(self.text))
            elif key == "percent":
                self.show(float(self.text)/100)
            elif key == "sqrt":
                if float(self.text) < 0:
                    self.text = "Error"
                else:
                    self.show(math.sqrt(float(self.text)))


def measurement(zero, mouth, wood_pose, length_mm, hook_force):
    """Check the tape against either end of the bar in the bar's own frame."""
    points = Rotation.from_quat(wood_pose[3:]).inv().apply(np.array([zero, mouth])-wood_pose[:3])
    end = 1 if points[0, 0] >= 0 else -1
    error = abs(points[0, 0]-end*length_mm/2000)
    direction = points[1]-points[0]
    cosine = -end*direction[0]/np.linalg.norm(direction)
    overrun = -end*points[1, 0]-length_mm/2000
    hooked = bool(error <= .001 and hook_force > 0)
    return dict(hooked=hooked, measured=bool(hooked and overrun > 0 and cosine >= math.cos(math.radians(10))),
                zero_error_mm=float(error*1000), angle_deg=float(math.degrees(math.acos(np.clip(cosine, -1, 1)))),
                overrun_mm=float(overrun*1000), hook_force_N=float(hook_force))


def score(submitted, seconds, length_mm, limit_seconds, hooked_at, measured_at):
    cutoff = min(seconds, limit_seconds)
    progress = .5 if measured_at is not None and measured_at <= cutoff else (
        .25 if hooked_at is not None and hooked_at <= cutoff else 0.)
    if (progress == .5 and seconds <= limit_seconds and submitted is not None
            and submitted.isdigit() and int(submitted) == length_mm):
        return 1.
    return progress


class Instruments:
    def __init__(self, base, stage):
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils import get_current_stage_id
        from pxr import UsdGeom
        from model import KEYS
        self.base, self.stage = base, stage
        self.calculator = Calculator()
        self.keys = list(KEYS)
        self.key_ids = [base.scene["calculator"].joint_names.index("press_"+key) for key in self.keys]
        self.down = np.zeros(len(self.keys), dtype=bool)
        self.events = []
        self.submission_time = None
        self.hooked_at = self.measured_at = None
        self.measurement_events = []
        self.physics = tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())
        self.contact = self.physics.create_rigid_contact_view("/World/envs/env_0/tape/hook",
            ["/World/envs/env_0/wood/bar"])
        root = "/World/envs/env_0"
        self.blade = UsdGeom.Mesh.Get(stage, root+"/tape/case/blade")
        self.uv = UsdGeom.PrimvarsAPI(self.blade).GetPrimvar("st")
        self.digits = [[UsdGeom.Imageable(stage.GetPrimAtPath(
            root+f"/calculator/base/display/digit_{digit}_{segment}"))
            for segment in "abcdefg"] for digit in range(8)]
        self.points = [UsdGeom.Imageable(stage.GetPrimAtPath(
            root+f"/calculator/base/display/digit_{digit}_point")) for digit in range(8)]
        self.displayed = None
        self.update_measurement(0.)
        self.update_visuals()

    def update(self, seconds):
        self.update_measurement(seconds)
        positions = self.base.scene["calculator"].data.joint_pos.torch[0, self.key_ids].cpu().numpy()
        pressed = positions < -.0008
        for index in np.flatnonzero(pressed & ~self.down):
            if not self.calculator.ended:
                key = self.keys[index]
                self.calculator.press(key)
                self.events.append({"time_s": seconds, "key": key, "display": self.calculator.text})
                if self.calculator.ended:
                    self.submission_time = seconds
        self.down |= pressed
        self.down[positions > -.00025] = False

    def update_measurement(self, seconds):
        from model import BLADE_Z, HOOK_HOME, TAPE_MOUTH
        from scene import BAR_LENGTH_MM, CONFIG
        tape = self.base.scene["tape"].data
        pose = tape.root_link_pose_w.torch[0].cpu().numpy()
        q = float(tape.joint_pos.torch[0, 0])
        self.tape_points = pose[:3]+Rotation.from_quat(pose[3:]).apply(
            [[HOOK_HOME-q, 0, BLADE_Z], [TAPE_MOUTH, 0, BLADE_Z]])
        self.hook_force = float(np.linalg.norm(self.contact.get_contact_force_matrix(
            1/CONFIG["simulation"]["physics_hz"])))
        if self.calculator.ended or seconds > CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"]:
            return
        wood = self.base.scene["wood"].data.root_link_pose_w.torch[0].cpu().numpy()
        state = measurement(*self.tape_points, wood, BAR_LENGTH_MM, self.hook_force)
        if state["hooked"] and self.hooked_at is None:
            self.hooked_at = seconds
            self.measurement_events.append(dict(time_s=seconds, stage="hooked", **state))
        if state["measured"] and self.measured_at is None:
            self.measured_at = seconds
            self.measurement_events.append(dict(time_s=seconds, stage="measured", **state))

    def update_visuals(self):
        from pxr import UsdGeom
        from model import BLADE_Z, HOOK_HOME, SEGMENTS, TAPE_MOUTH
        q = float(self.base.scene["tape"].data.joint_pos.torch[0, 0])
        start = HOOK_HOME-q
        self.blade.GetPointsAttr().Set([(start, -.011, BLADE_Z+.0004),
            (TAPE_MOUTH, -.011, BLADE_Z+.0004), (TAPE_MOUTH, .011, BLADE_Z+.0004),
            (start, .011, BLADE_Z+.0004)])
        u = (TAPE_MOUTH-start)/.330
        self.uv.Set([(0, 0), (u, 0), (u, 1), (0, 1)])
        text = self.calculator.text
        if text == self.displayed:
            return
        plain = text.replace(".", "").rjust(8)
        decimal = text.find(".")
        dot_digit = 8-len(text.replace(".", ""))+decimal-1 if decimal >= 0 else -1
        for index, char in enumerate(plain):
            lit = SEGMENTS[char]
            for name, segment in zip("abcdefg", self.digits[index]):
                segment.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if name in lit else UsdGeom.Tokens.invisible)
            self.points[index].GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if index == dot_digit else UsdGeom.Tokens.invisible)
        self.displayed = text

    def result(self, seconds):
        from scene import BAR_LENGTH_MM, CONFIG
        truth = BAR_LENGTH_MM
        limit = CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"]
        return {"score": score(self.calculator.submitted,
                    seconds if self.submission_time is None else self.submission_time, truth, limit,
                    self.hooked_at, self.measured_at),
                "length_mm": truth, "submitted": self.calculator.submitted,
                "hooked_at_s": self.hooked_at, "measured_at_s": self.measured_at,
                "measurement_events": self.measurement_events,
                "submission_time_s": self.submission_time, "time_s": seconds, "events": self.events}

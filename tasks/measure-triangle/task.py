"""Physical key events, visible instrument state and result-only scoring."""

import math

import numpy as np

AREA_CM2 = 50.


def score(submitted, submission_time_s, limit_seconds):
    """Only an equal-key submission of the correct area within the limit scores."""
    if submitted is None or submission_time_s is None or not 0 <= submission_time_s <= limit_seconds:
        return 0.
    try:
        return float(float(submitted) == AREA_CM2)
    except (TypeError, ValueError):
        return 0.


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


class Instruments:
    def __init__(self, base, stage):
        from pxr import UsdGeom
        from model import KEYS
        self.base, self.stage = base, stage
        self.calculator = Calculator()
        self.keys = list(KEYS)
        self.key_ids = [base.scene["calculator"].joint_names.index("press_"+key) for key in self.keys]
        self.down = np.zeros(len(self.keys), dtype=bool)
        self.events = []
        self.submission_time = None
        root = "/World/envs/env_0"
        self.digits = [[UsdGeom.Imageable(stage.GetPrimAtPath(
            root+f"/calculator/base/display/digit_{digit}_{segment}"))
            for segment in "abcdefg"] for digit in range(8)]
        self.points = [UsdGeom.Imageable(stage.GetPrimAtPath(
            root+f"/calculator/base/display/digit_{digit}_point")) for digit in range(8)]
        self.displayed = None
        self.update_visuals()

    def update(self, seconds):
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

    def update_visuals(self):
        from pxr import UsdGeom
        from model import SEGMENTS
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
        from scene import CONFIG
        limit = CONFIG["episode"]["max_control_steps"]/CONFIG["simulation"]["control_hz"]
        return {"score": score(self.calculator.submitted, self.submission_time, limit),
                "area_cm2": AREA_CM2, "submitted": self.calculator.submitted,
                "submission_time_s": self.submission_time, "time_s": seconds, "events": self.events}

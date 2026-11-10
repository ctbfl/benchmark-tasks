"""Light schedule, physical press detection, score display and scoring."""

import numpy as np

PERIOD_S = 5.0          # one light every 5 s, first at t = 5 s
WINDOW_S = 1.0          # each light stays on for 1 s unless hit
REQUIRED_HITS = 5
PRESSED_M = -.004       # button travel that counts as a press
RELEASED_M = -.0015     # travel above which the button counts as released


def score(hits, limit_seconds):
    """1.0 if at least five hits occurred within the time limit, otherwise 0.0."""
    valid = [t for t in hits if 0 <= t <= limit_seconds]
    return float(len(valid) >= REQUIRED_HITS)


class Game:
    """Deterministic light sequence from the scene seed; time is counted in physics ticks."""

    def __init__(self, seed, physics_hz, limit_seconds, buttons):
        self.hz = physics_hz
        self.period = round(PERIOD_S*physics_hz)
        self.window = round(WINDOW_S*physics_hz)
        count = int(limit_seconds // PERIOD_S)
        if count*PERIOD_S >= limit_seconds:
            count -= 1
        self.buttons = list(buttons)
        self.sequence = [self.buttons[i] for i in np.random.default_rng(seed).integers(0, len(buttons), count)]
        self.hit_lights = set()
        self.hits = []          # hit times in seconds

    def light_index(self, tick):
        k, phase = divmod(tick, self.period)
        if 1 <= k <= len(self.sequence) and phase < self.window:
            return k
        return None

    def lit(self, tick):
        k = self.light_index(tick)
        if k is None or k in self.hit_lights:
            return None
        return self.sequence[k-1]

    def press(self, button, tick):
        if self.lit(tick) == button:
            self.hit_lights.add(self.light_index(tick))
            self.hits.append(tick/self.hz)
            return True
        return False


class Instruments:
    """Reads button joint travel each physics tick; drives the lamps and the display."""

    def __init__(self, base, stage, game):
        from pxr import UsdGeom
        from layout import BUTTONS
        self.base, self.stage, self.game = base, stage, game
        self.names = list(BUTTONS)
        device = base.scene["device"]
        self.ids = [device.joint_names.index("press_"+name) for name in self.names]
        self.down = np.zeros(len(self.names), dtype=bool)
        self.events = []
        root = "/World/envs/env_0/device"
        self.lamps = {name: [UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/button_{name}/cap_{state}"))
                             for state in ("on", "off")] for name in self.names}
        self.digits = [[UsdGeom.Imageable(stage.GetPrimAtPath(
            f"{root}/base/display/panel/digit_{digit}_{segment}")) for segment in "abcdefg"]
            for digit in range(2)]
        self.shown = None
        self.tick = 0

    def update(self, tick):
        self.tick = tick
        travel = self.base.scene["device"].data.joint_pos.torch[0, self.ids].cpu().numpy()
        pressed = travel < PRESSED_M
        for index in np.flatnonzero(pressed & ~self.down):
            name = self.names[index]
            lit = self.game.lit(tick)
            hit = self.game.press(name, tick)
            self.events.append({"time_s": tick/self.game.hz, "button": name, "lit": lit, "hit": hit,
                                "hits": len(self.game.hits)})
        self.down |= pressed
        self.down[travel > RELEASED_M] = False

    def visual_state(self, tick=None):
        tick = self.tick if tick is None else tick
        return self.game.lit(tick), len(self.game.hits)

    def show(self, lit, hits):
        from pxr import UsdGeom
        from layout import SEGMENTS
        if (lit, hits) == self.shown:
            return
        for name, (on, off) in self.lamps.items():
            on.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if name == lit else UsdGeom.Tokens.invisible)
            off.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible if name == lit else UsdGeom.Tokens.inherited)
        text = str(min(hits, 99)).rjust(2)
        for digit, char in enumerate(text):
            for letter, prim in zip("abcdefg", self.digits[digit]):
                prim.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if letter in SEGMENTS[char]
                                             else UsdGeom.Tokens.invisible)
        self.shown = (lit, hits)

    def update_visuals(self):
        self.show(*self.visual_state())

    def result(self, limit_seconds):
        return {"score": score(self.game.hits, limit_seconds), "hits": len(self.game.hits),
                "hit_times_s": list(self.game.hits), "required_hits": REQUIRED_HITS,
                "sequence": self.game.sequence, "time_s": self.tick/self.game.hz, "events": self.events}

"""Spin and stop credit; basket credit also needs timely hands-off response."""
import numpy as np
from scipy.spatial.transform import Rotation
from yam_arena.station import TABLE_TOP

from assets import BOARD_Z, LONG, SHORT, TRAYS, TRAY_FLOOR, TRAY_HALF
from scene import BOARD_XY, CONFIG

DT = 1 / CONFIG["simulation"]["physics_hz"]
BRAKE_TICKS = round(.30 / DT)
ARREST_TICKS = round(.0125 / DT)
RELEASE_TICKS = round(.50 / DT)
OBSERVATION_TICKS = round(.25 / DT)
VISIBLE_TURN = 2*np.pi
TURNING_SPEED = 4*np.pi
ARRESTED_SPEED = .5


class SpinTest:
    def __init__(self):
        self.previous = None
        self.angle = self.low = self.high = self.max_turn = 0.
        self.spin_tick = self.arrest_tick = None
        self.spin_speed = 0.
        self.brake_speed = 0.
        self.brake_tick = self.quiet_tick = None
        self.attempt_arrest_tick = None
        self.release_tick = None
        self.evidence = None

    def sample(self, tick, item, on_board):
        if not on_board:
            self.previous = None
            self.angle = self.low = self.high = 0.
            self.brake_tick = self.quiet_tick = self.release_tick = None
            self.attempt_arrest_tick = None
            return
        yaw, speed = item["yaw"], item["velocity"][5]
        held = (any(item["normal_impulse_ns"])
                or np.linalg.norm(item["contact_angular_impulse_nms"]) > 1e-10)
        previous = self.previous
        self.previous = (yaw, speed, held)
        if previous is None:
            return

        self.angle += (yaw-previous[0]+np.pi) % (2*np.pi)-np.pi
        self.low, self.high = min(self.low, self.angle), max(self.high, self.angle)
        self.max_turn = max(self.max_turn, self.high-self.low)
        if self.spin_tick is None and self.max_turn+1e-9 >= VISIBLE_TURN:
            self.spin_tick, self.spin_speed = tick, speed

        if self.spin_tick is not None and self.evidence is None:
            if self.brake_tick is not None and self.release_tick is None:
                deadline = (self.brake_tick+BRAKE_TICKS if self.attempt_arrest_tick is None
                            else self.attempt_arrest_tick+RELEASE_TICKS)
                if tick > deadline:
                    self.brake_tick = self.quiet_tick = self.attempt_arrest_tick = None
            if (self.brake_tick is None and held and not previous[2]
                    and abs(previous[1]) >= TURNING_SPEED):
                self.brake_tick = tick
                self.brake_speed = previous[1]
            if self.brake_tick is not None and self.attempt_arrest_tick is None:
                if held and abs(speed) <= ARRESTED_SPEED:
                    if self.quiet_tick is None:
                        self.quiet_tick = tick
                    if tick-self.quiet_tick+1 >= ARREST_TICKS:
                        self.attempt_arrest_tick = self.quiet_tick
                        if self.arrest_tick is None:
                            self.arrest_tick = self.quiet_tick
                else:
                    self.quiet_tick = None

        # Basket credit requires timely release and an uninterrupted board-level response.
        if self.attempt_arrest_tick is not None and self.evidence is None:
            if held:
                self.release_tick = None
            elif (self.release_tick is None
                  and tick-self.attempt_arrest_tick <= RELEASE_TICKS):
                self.release_tick = tick
            elif self.release_tick is not None and tick-self.release_tick >= OBSERVATION_TICKS:
                self.evidence = {"spin_speed_rad_s": self.spin_speed,
                                 "pre_brake_speed_rad_s": self.brake_speed,
                                 "brake_contact_tick": self.brake_tick,
                                 "arrest_tick": self.attempt_arrest_tick,
                                 "brake_release_tick": self.release_tick,
                                 "observation_end_tick": tick}


def check_spin(test, tick, item, pose, span):
    on_board = (np.all(np.abs(pose[:2]-BOARD_XY) < [.14, .155])
                and abs(pose[2]-span[2]-TABLE_TOP-BOARD_Z) < .003)
    test.sample(tick, item, bool(on_board))


def shell_span(pose):
    axis = Rotation.from_quat(pose[3:]).apply([1, 0, 0])
    return np.sqrt(SHORT**2+(LONG**2-SHORT**2)*axis**2)


class Score:
    def __init__(self, raw):
        self.trays = ["raw" if flag else "boiled" for flag in raw]
        self.tests = [SpinTest() for _ in raw]
        self.placed = [False for _ in raw]
        self.shell_broken = False
        self.ticks = 0

    def sample(self, row):
        for i, item in enumerate(row["eggs"]):
            self.shell_broken |= bool(item["shell_broken"])
            pose = np.asarray(item["pose"])
            span = shell_span(pose)
            test = self.tests[i]
            check_spin(test, row["tick"], item, pose, span)
            seated = (np.all(np.abs(pose[:2]-TRAYS[self.trays[i]]) + span[:2] < TRAY_HALF)
                      and abs(pose[2]-span[2]-TABLE_TOP-TRAY_FLOOR) < .003)
            self.placed[i] = bool(test.evidence is not None and seated
                                  and not any(item["normal_impulse_ns"]))
        self.ticks = row["tick"]

    def result(self):
        spun = [test.spin_tick is not None for test in self.tests]
        stopped = [test.arrest_tick is not None for test in self.tests]
        points = [(4*spin+3*stop+3*place)/10
                  for spin, stop, place in zip(spun, stopped, self.placed, strict=True)]
        if self.shell_broken:
            points = [0. for _ in points]
        return {"score": float(np.mean(points)), "per_egg_score": points,
                "shell_broken": self.shell_broken,
                "spun": spun, "stopped": stopped, "placed": self.placed.copy(),
                "response_window_complete": [test.evidence is not None for test in self.tests],
                "spin_turn_degrees": [float(np.rad2deg(test.max_turn)) for test in self.tests],
                "spin_ticks": [test.spin_tick for test in self.tests],
                "arrest_ticks": [test.arrest_tick for test in self.tests],
                "correct_trays": self.trays, "physics_ticks": self.ticks}


def evaluate(rows, raw):
    score = Score(raw)
    for row in rows:
        score.sample(row)
    return score.result()

"""Success evaluation from measured geometry and robot contact forces."""

import numpy as np

TARGET_GEAR = 1
START = (.3605, 0, .002)  # Folded bounding-box centre at the goal centre (.46, 0).
GOAL_LOW = (.29, -.12)
GOAL_HIGH = (.63, .12)
CONTROL_HZ = 20
MAX_STEPS = 2400
INSTRUCTION = (
    "Fully unfold the laptop stand and set it to its highest position. "
    "Spread the two sides to their full width, lift the upper supports, and seat both "
    "support rods securely in the base slots for the highest setting. "
    "Leave the entire stand inside the red tape rectangle, resting stably on the table. "
    "Release it and move both grippers away."
)


class Score:
    def __init__(self, gear=TARGET_GEAR, low=GOAL_LOW, high=GOAL_HIGH):
        assert 1 <= gear <= 7
        self.gear, self.low, self.high = gear, np.asarray(low), np.asarray(high)
        self.held_s = 0.0

    def update(self, m, robot_contact_N, dt):
        assert dt > 0 and np.isfinite(robot_contact_N) and robot_contact_N >= 0
        expanded = .180 <= m["width"] <= .190
        seated = all(s["gear"] == self.gear and abs(s["seat_error"][0]) < .0029
                     and abs(s["seat_error"][1]) < .0018 and abs(s["seat_error"][2]) < .0005
                     for s in m["sides"].values())
        contained = bool(np.all(np.asarray(m["low"])[:2] >= self.low)
                         and np.all(np.asarray(m["high"])[:2] <= self.high))
        grounded = -.0005 <= m["low"][2] <= .0005 and all(
            abs(s["base_z"]) < .0005 and s["base_tilt_deg"] < 2 for s in m["sides"].values())
        released = robot_contact_N < .01
        stable = m["max_speed"] < .01 and m["max_spin"] < .1 and m["joint_error"] < .0005
        ready = expanded and seated and contained and grounded and released and stable
        self.held_s = self.held_s + dt if ready else 0.0
        success = self.held_s >= 1.0 - 1e-8
        score = .25*expanded + .25*(expanded and seated) + .25*(expanded and seated and contained and grounded)
        return {"success":success, "score":1.0 if success else score, "held_s":self.held_s,
                "expanded":expanded,"seated":seated,"contained":contained,"grounded":grounded,
                "released":released,"stable":stable}

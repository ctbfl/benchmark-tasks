"""Partial credit for setup; full credit for a released, table-supported stand."""

import numpy as np

TARGET_GEAR = 1
START = (.3605, 0, .002)  # Folded bounding-box centre at the goal centre (.46, 0).
GOAL_LOW = (.29, -.12)
GOAL_HIGH = (.63, .12)
CONTROL_HZ = 20
MAX_STEPS = 2400
SLOT_HALF_LENGTH = .005  # Half of model.SLOT_LENGTH.
SLOT_HALF_WIDTH = .004  # Half of model.SLOT_WIDTH.
# Slot floor (4 mm) and rim (8 mm), relative to model.SEAT_Z (6.5 mm).
SLOT_HEIGHT_LOW = -.0025
SLOT_HEIGHT_HIGH = .0015
CONTACT_EPSILON_N = .01
INSTRUCTION = (
    "Fully spread the laptop stand, with both support rods seated in the slots "
    "for its highest setting. The whole stand's top-down outline must lie within "
    "the inner edges of the red tape rectangle. Leave both base rails resting on "
    "the tabletop, release the stand and move both grippers clear."
)


class Score:
    def __init__(self, gear=TARGET_GEAR, low=GOAL_LOW, high=GOAL_HIGH):
        assert 1 <= gear <= 7
        self.gear, self.low, self.high = gear, np.asarray(low), np.asarray(high)

    def update(self, m, robot_contact_N, table_contact_N):
        table_contact = np.asarray(table_contact_N)
        assert np.isfinite(robot_contact_N) and robot_contact_N >= 0
        assert table_contact.shape == (2,) and np.isfinite(table_contact).all()
        assert (table_contact >= 0).all()
        # Both sides spread open (full width is 185 mm).
        expanded = m["width"] >= .180
        # Pin centres lie inside the target openings, between each slot's floor and rim.
        slots = {}
        for side in ("L", "R"):
            s = m["sides"][side]
            x, y, z = s["seat_error"]
            slots[side] = bool(s["gear"] == self.gear
                               and abs(x) <= SLOT_HALF_LENGTH and abs(y) <= SLOT_HALF_WIDTH
                               and SLOT_HEIGHT_LOW <= z <= SLOT_HEIGHT_HIGH)
        seated = all(slots.values())
        # Top-down outline of the whole stand lies inside the tape rectangle.
        contained = bool(np.all(np.asarray(m["low"])[:2] >= self.low)
                         and np.all(np.asarray(m["high"])[:2] <= self.high))
        released = bool(robot_contact_N < CONTACT_EPSILON_N)
        supported = bool(np.all(table_contact > CONTACT_EPSILON_N))
        success = expanded and seated and contained and released and supported
        score = .25 * (expanded + sum(slots.values()) + success)
        return {"success":success, "score":score, "expanded":expanded,
                "left_seated":slots["L"], "right_seated":slots["R"],
                "seated":seated, "contained":contained,
                "released":released, "supported":supported}

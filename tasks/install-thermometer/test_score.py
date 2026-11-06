"""Regression cases for assembled, unsupported, tilted and obstructed outcomes."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from score import score


def installed():
    sample = {"stand": [.43, .19, .7475, 0, 0, 0, 1],
              "beaker": [.43, .16, .7605, 0, 0, 0, 1],
              "clamp": [.43, .28, 1.0925, 0, 0, 0, 1],
              "thermometer": [.43, .16, .7875, 0, 0, 0, 1],
              "screw_travel": 0., "jaw_angle": -.03,
              "clamp_robot_force_N": 0., "thermometer_robot_force_N": 0., "stand_robot_force_N": 0.}
    return [{**deepcopy(sample), "time_s": t} for t in np.linspace(0, 1, 21)]


class InstallationScoring(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(score(installed())["score"], 1.)

    def test_clamp_held_by_robot(self):
        rows = installed()
        for row in rows:
            row["clamp_robot_force_N"] = 1.
        self.assertEqual(score(rows)["score"], 0.)

    def test_thermometer_held_by_robot(self):
        rows = installed()
        for row in rows:
            row["thermometer_robot_force_N"] = .2
        self.assertEqual(score(rows)["score"], .25)

    def test_stand_supported_by_robot(self):
        rows = installed()
        for row in rows:
            row["stand_robot_force_N"] = 2.
        self.assertEqual(score(rows)["score"], 0.)

    def test_stand_tipped(self):
        rows = installed()
        for row in rows:
            row["stand"][3:] = Rotation.from_euler("x", 90, degrees=True).as_quat().tolist()
        self.assertEqual(score(rows)["score"], 0.)

    def test_transient_assembly(self):
        self.assertEqual(score(installed()[-1:])["score"], 0.)
        rows = installed()
        rows[0]["thermometer"][2] -= .01
        self.assertEqual(score(rows)["score"], .25)

    def test_time_limit(self):
        rows = installed()
        for row in rows:
            row["time_s"] += 120
        self.assertEqual(score(rows)["score"], 0.)

    def test_tilted(self):
        rows = installed()
        rotation = Rotation.from_euler("x", 25, degrees=True)
        point = np.array([.43, .16, 1.0925]) - rotation.apply([0, 0, .305])
        for row in rows:
            row["thermometer"] = [*point, *rotation.as_quat()]
        self.assertEqual(score(rows)["score"], .25)

    def test_dry_bulb(self):
        rows = installed()
        for row in rows:
            row["beaker"][0] += .1
        self.assertEqual(score(rows)["score"], .75)

    def test_scale_obstructed(self):
        rows = installed()
        for row in rows:
            row["clamp"][2] = .96
        result = score(rows)
        self.assertEqual(result["score"], .75)
        self.assertFalse(result["criteria"]["scale_clear"])

    def test_off_rod_and_loose_screw(self):
        rows = installed()
        for row in rows:
            row["clamp"][0] += .02
        self.assertEqual(score(rows)["score"], 0.)
        rows = installed()
        for row in rows:
            row["screw_travel"] = .002
        self.assertEqual(score(rows)["score"], 0.)


if __name__ == "__main__":
    unittest.main()

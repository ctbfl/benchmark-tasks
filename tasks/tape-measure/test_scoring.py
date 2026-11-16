"""Scoring boundaries and strategies that must not earn full credit."""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from task import measurement, score


class ScoringTests(unittest.TestCase):
    def test_measurement_conditions(self):
        wood = np.array([0., 0., 0., 0., 0., 0., 1.])
        zero = np.array([-.0965, 0., .035])
        mouth = np.array([.12, 0., .035])
        self.assertTrue(measurement(zero, mouth, wood, 193, 1.)["measured"])
        self.assertFalse(measurement(zero, mouth, wood, 193, 0.)["hooked"])
        self.assertFalse(measurement(zero+[.0011, 0, 0], mouth, wood, 193, 1.)["hooked"])
        self.assertFalse(measurement(zero, [.0964, 0, .035], wood, 193, 1.)["measured"])
        for angle, expected in ((9.99, True), (10.01, False), (90, False), (180, False)):
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.])
            self.assertEqual(measurement(zero, zero+.23*direction, wood, 193, 1.)["measured"], expected)

    def test_either_end_and_moved_bar(self):
        wood = np.array([0., 0., 0., 0., 0., 0., 1.])
        zero, mouth = np.array([-.0965, 0., .035]), np.array([.12, 0., .035])
        self.assertTrue(measurement(-zero, -mouth, wood, 193, 1.)["measured"])
        rotation = Rotation.from_euler("xyz", [22, 33, 44], degrees=True)
        position = np.array([.3, -.4, .8])
        self.assertTrue(measurement(position+rotation.apply(zero), position+rotation.apply(mouth),
            np.r_[position, rotation.as_quat()], 193, 1.)["measured"])

    def test_score_and_submission_order(self):
        for submitted, seconds, hooked, measured, expected in (
            ("193", 100, None, None, 0.),
            ("193", 100, 50, None, .25),
            (None, 120, 50, 60, .5),
            ("173", 100, 50, 60, .5),
            ("Error", 100, 50, 60, .5),
            ("193", 100, 50, 60, 1.),
            ("193", 120, 50, 60, 1.),
            ("193", 120.001, 50, 60, .5),
            ("193", 40, 50, 60, 0.),
            ("193", 55, 50, 60, .25),
            ("193", 125, 121, 122, 0.),
        ):
            self.assertEqual(score(submitted, seconds, 193, 120, hooked, measured), expected)


if __name__ == "__main__":
    unittest.main()

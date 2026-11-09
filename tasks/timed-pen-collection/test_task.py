"""Regression checks for switch debouncing, clock units, and hollow containment."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from task_runtime import Timer, in_holder


class TaskTests(unittest.TestCase):
    def test_button_hold_release_and_stop(self):
        timer = Timer()
        self.assertIsNone(timer.update(0., -.0005))
        self.assertEqual(timer.update(1., -.004), "start")
        self.assertIsNone(timer.update(2., -.004))
        self.assertIsNone(timer.update(3., -.002))
        self.assertIsNone(timer.update(4., -.004))
        timer.update(5., 0.)
        self.assertEqual(timer.update(6., -.004), "stop")
        timer.update(10., -.004)
        self.assertEqual(timer.display, "00:05:00")
        self.assertEqual(len(timer.events), 2)

    def test_centiseconds_and_minute_rollover(self):
        timer = Timer()
        timer.update(0., -.004)
        timer.update(59.99, 0.)
        self.assertEqual(timer.display, "00:59:99")
        timer.update(60., 0.)
        self.assertEqual(timer.display, "01:00:00")
        timer.update(60.125, 0.)
        self.assertEqual(timer.display, "01:00:12")

    def test_cavity_geometry(self):
        cup = [0, 0, 0, 0, 0, 0, 1]
        self.assertTrue(in_holder([0, 0, .081, 0, 0, 0, 1], cup))
        self.assertFalse(in_holder([.1, 0, .081, 0, 0, 0, 1], cup))
        self.assertFalse(in_holder([0, 0, .3, 0, 0, 0, 1], cup))
        self.assertFalse(in_holder([0, 0, .081, *Rotation.from_euler("y", 90, degrees=True).as_quat()], cup))
        rotation = Rotation.from_euler("z", 40, degrees=True)
        moved_cup = [1, 2, 3, *rotation.as_quat()]
        moved_pen = [1, 2, 3.081, *rotation.as_quat()]
        self.assertTrue(in_holder(moved_pen, moved_cup))

    def test_seeded_initial_layout(self):
        from scene import pen_placements
        first, repeated, other = pen_placements(0), pen_placements(0), pen_placements(1)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 5)

    def test_score_requires_sequence_and_final_retention(self):
        from copy import deepcopy
        from score import score
        initial = {"time_s": 0., "button_m": 0., "cup": [0, 0, 0, 0, 0, 0, 1],
            "pens": [[.2+i*.02, 0, .005, *Rotation.from_euler("y", 90, degrees=True).as_quat()] for i in range(5)],
            "robot_forces_N": [0.]*5, "supported": []}
        start = {**initial, "time_s": 1., "button_m": -.004}
        release = {**initial, "time_s": 1.1}
        collected = {**initial, "time_s": 3., "pens": [[(i-2)*.009, 0, .081, 0, 0, 0, 1] for i in range(5)],
                     "supported": list(range(5))}
        stop = {**collected, "time_s": 4., "button_m": -.004}
        samples = [initial, start, release, collected, stop]
        self.assertEqual(score(samples), 1.)
        self.assertEqual(score(samples[:-1]), 0.)
        held = deepcopy(samples)
        held[-1]["robot_forces_N"][0] = .1
        self.assertEqual(score(held), 0.)
        self.assertEqual(score(samples+[{**initial, "time_s": 5.}]), 0.)
        in_flight = deepcopy(samples)
        in_flight[-1]["supported"] = []
        self.assertEqual(score(in_flight), 0.)


if __name__ == "__main__":
    unittest.main()

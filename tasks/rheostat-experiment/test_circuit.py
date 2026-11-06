"""Electrical boundaries and history-dependent scoring regressions."""
import unittest

from circuit import MAX_STEPS, HZ, PHYSICS_HZ, Score, brightness, voltage, intact

TARGET = -.26 * 3/7


class CircuitTests(unittest.TestCase):
    def test_divider_and_wired_direction(self):
        self.assertAlmostEqual(voltage(0), 1.)
        self.assertAlmostEqual(voltage(TARGET), 5.)
        self.assertAlmostEqual(voltage(-.13), 15.)
        self.assertAlmostEqual(voltage(.13), 15/29)
        self.assertGreater(voltage(-.01), voltage(0))
        self.assertLess(voltage(.01), voltage(0))

    def test_brightness_thresholds(self):
        self.assertEqual([brightness(v) for v in (0,1,3,5,7,15)], [0,0,0,.5,1,1])

    def test_dwell_and_leaving_target(self):
        score = Score()
        for _ in range(PHYSICS_HZ-1):
            self.assertFalse(score.update(TARGET)["success"])
        self.assertTrue(score.update(TARGET)["success"])
        self.assertFalse(score.update(0)["success"])
        self.assertEqual(score.held, 0)

    def test_overshoot_is_not_forgotten(self):
        score = Score()
        score.update(-.13)
        for _ in range(PHYSICS_HZ*2):
            self.assertFalse(score.update(TARGET)["success"])
        self.assertEqual(score.peak, 15.)

    def test_timeout(self):
        score = Score()
        score.ticks = MAX_STEPS * PHYSICS_HZ // HZ
        for _ in range(PHYSICS_HZ):
            self.assertFalse(score.update(TARGET)["success"])

    def test_invalid_state_fails(self):
        for value in (float("nan"),float("inf"),float("-inf")):
            with self.assertRaises(ValueError):
                voltage(value)
            with self.assertRaises(ValueError):
                brightness(value)

    def test_displaced_or_toppled_base_cannot_score(self):
        initial = [.52,0,.7475,0,0,0,1]
        self.assertTrue(intact(initial,initial))
        for root in ([.52,0,.80,0,0,0,1],[.52,.10,.7475,0,0,0,1],
                     [.52,0,.7475,.707,0,0,.707]):
            self.assertFalse(intact(root,initial))
        score = Score()
        for _ in range(2*PHYSICS_HZ):
            self.assertFalse(score.update(TARGET,False)["success"])


if __name__ == "__main__":
    unittest.main()

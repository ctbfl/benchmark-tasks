"""Check that a broken ink line cannot accumulate qualified orbit."""

import unittest

import numpy as np

from geometry import CENTER, HOLE, ORBIT, RATIO, polar
from scene import DT
from scoring import BOARD_TOP, RosetteScore


def sample(scorer, angle, lift=0., load=.2):
    yaw = -RATIO*angle
    center = CENTER + ORBIT*polar(angle)
    bore = center + HOLE*polar(yaw)
    wheel = np.r_[center, BOARD_TOP, 0., 0., np.sin(yaw/2), np.cos(yaw/2)]
    nib = np.r_[bore, BOARD_TOP+lift, 0., 0., 0., 1.]
    scorer.update(wheel, nib, [load, .1, .1])
    return bore


class InkContinuity(unittest.TestCase):
    def check_dropout(self, advance, lift, segments, *, start=2.):
        scorer = RosetteScore()
        sample(scorer, start-.02)
        first = sample(scorer, start)
        for angle in np.linspace(start, start+advance, round(.10/DT)+1)[1:]:
            sample(scorer, angle, lift=lift, load=0.)
        last = sample(scorer, start+advance)
        sample(scorer, start+advance+.02)
        self.assertEqual(scorer.generation, segments)
        self.assertEqual(scorer.ink_segments, segments)
        self.assertEqual([mark[-1] for mark in scorer.stroke], [0, 1, int(segments == 1), 1])
        expected_arc = .04+advance if segments == 1 else .02
        self.assertAlmostEqual(scorer.best, expected_arc)
        return float(np.linalg.norm(last-first))

    def test_lift_over_nearly_two_millimetres_of_ink(self):
        travel = self.check_dropout(.0018/(ORBIT+RATIO*HOLE), .0004, segments=2)
        self.assertAlmostEqual(travel, .0018, delta=.00002)

    def test_lift_over_1_8_mm_of_wheel_travel(self):
        self.check_dropout(.0018/ORBIT, .0004, segments=2, start=.90)

    def test_surface_excursion_with_no_ink(self):
        self.check_dropout(.0018/(ORBIT+RATIO*HOLE), 0., segments=2)

    def test_stationary_force_chatter(self):
        self.check_dropout(0., 0., segments=1)

    def test_stationary_lift(self):
        self.check_dropout(0., .0004, segments=2)

    def test_surface_overlap_during_chatter(self):
        self.check_dropout(.0004/(ORBIT+RATIO*HOLE), 0., segments=1)

    def test_return_to_last_mark_cannot_hide_an_excursion(self):
        scorer = RosetteScore()
        sample(scorer, 1.98)
        sample(scorer, 2.)
        advance = .0018/(ORBIT+RATIO*HOLE)
        for fraction in np.r_[np.linspace(0, 1, 13)[1:], np.linspace(1, 0, 13)[1:]]:
            sample(scorer, 2.+advance*fraction, load=0.)
        sample(scorer, 2.)
        sample(scorer, 2.02)
        self.assertEqual(scorer.generation, 2)
        self.assertEqual(scorer.stroke[-2][-1], 0)
        self.assertAlmostEqual(scorer.best, .02)


if __name__ == "__main__":
    unittest.main()

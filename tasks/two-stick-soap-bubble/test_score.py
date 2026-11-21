"""Check observable film and floating-bubble credit without size or hold gates."""
import unittest

import numpy as np

from film import Bubble, Film
from geometry import STRANDS, TABLE_TOP


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.film = Film(1 / 240)
        self.center = np.array([.35, 0., TABLE_TOP + .65])
        self.upper = self.center + [[0., -.002, 0.], [0., 0., .002], [0., .002, 0.]]
        self.lower = self.center + [[0., -.002, 0.], [0., 0., -.002], [0., .002, 0.]]
        self.cord = np.tile(self.center, (sum(n for _, n in STRANDS), 1))
        self.wands = np.array([self.center + [.15, 0., 0.]])

    def update(self):
        self.film.update(self.upper, self.lower, self.cord, self.wands)

    def form_film(self):
        self.film.wet_until = 1.
        self.update()
        self.assertEqual(self.film.score, .5)

    def test_dry_loop_has_no_credit(self):
        self.update()
        self.assertEqual(self.film.score, 0.)

    def test_small_flat_film_scores_without_inflation(self):
        self.form_film()
        self.assertEqual(self.film.volume, 0.)
        self.assertTrue(self.film.result()["film_formed"])
        self.assertFalse(self.film.result()["bubble_floating"])

    def test_wet_loop_still_in_bath_has_no_credit(self):
        from geometry import SURFACE

        offset = np.array([.2, 0., SURFACE - .01 - self.center[2]])
        self.upper += offset
        self.lower += offset
        self.cord += offset
        self.film.wet_until = 1.
        self.update()
        self.assertEqual(self.film.score, 0.)

    def test_small_bubble_near_wand_scores_without_a_hold(self):
        self.form_film()
        center = self.center + [.05, 0., 0.]
        self.film.bubble = Bubble(center.copy(), np.array([.1, 0., 0.]), .005)
        self.wands = np.array([center - [.018, 0., 0.]])
        self.update()
        self.assertEqual(self.film.score, 1.)
        self.assertFalse(np.array_equal(self.film.bubble.center, center))

    def test_burst_before_floating_keeps_only_film_credit(self):
        self.form_film()
        self.film.bubble = Bubble(np.array([.35, 0., TABLE_TOP - .02]), np.zeros(3), .005)
        self.update()
        self.assertIsNone(self.film.bubble)
        self.assertEqual(self.film.score, .5)

    def test_later_burst_does_not_revoke_completed_events(self):
        self.test_small_bubble_near_wand_scores_without_a_hold()
        self.film.bubble.age = 5.
        self.update()
        self.assertIsNone(self.film.bubble)
        self.assertEqual(self.film.score, 1.)


if __name__ == "__main__":
    unittest.main()

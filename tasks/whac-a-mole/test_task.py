"""Light schedule and scoring checks; no simulator or model calls."""

import unittest

from task import Game, score

BUTTONS = ("near_left", "near_right", "far_left", "far_right")
HZ = 240


def game(seed=0):
    return Game(seed, HZ, 120., BUTTONS)


class TaskTests(unittest.TestCase):
    def test_schedule(self):
        g = game()
        self.assertEqual(len(g.sequence), 23)
        self.assertIsNone(g.lit(0))
        self.assertIsNone(g.lit(5*HZ-1))
        self.assertEqual(g.lit(5*HZ), g.sequence[0])
        self.assertEqual(g.lit(6*HZ-1), g.sequence[0])
        self.assertIsNone(g.lit(6*HZ))
        self.assertEqual(g.lit(115*HZ), g.sequence[22])
        self.assertIsNone(g.lit(120*HZ))

    def test_seed_is_deterministic(self):
        self.assertEqual(game(0).sequence, game(0).sequence)
        self.assertNotEqual(game(0).sequence, game(1).sequence)

    def test_only_lit_button_in_window_counts_once(self):
        g = game()
        lit = g.sequence[0]
        other = next(name for name in BUTTONS if name != lit)
        self.assertFalse(g.press(lit, 5*HZ-1))        # before the light
        self.assertFalse(g.press(other, 5*HZ+10))     # wrong button
        self.assertTrue(g.press(lit, 5*HZ+20))
        self.assertIsNone(g.lit(5*HZ+21))             # lamp turns off after a hit
        self.assertFalse(g.press(lit, 5*HZ+30))       # no second hit in one window
        self.assertFalse(g.press(lit, 6*HZ))          # after the window
        self.assertEqual(len(g.hits), 1)

    def test_five_hits_required_within_limit(self):
        g = game()
        for k in range(1, 5):
            g.press(g.sequence[k-1], k*5*HZ+12)
        self.assertEqual(score(g.hits, 120.), 0.)
        g.press(g.sequence[4], 25*HZ+12)
        self.assertEqual(score(g.hits, 120.), 1.)
        self.assertEqual(score([1., 2., 3., 4., 121.], 120.), 0.)


if __name__ == "__main__":
    unittest.main()

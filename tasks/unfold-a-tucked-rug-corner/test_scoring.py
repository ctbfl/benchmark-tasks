"""Check conditional credit, whole-footprint containment and false flatness."""
import unittest
import numpy as np

from scoring import grade, THICKNESS


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.backing = np.array([[.15, -.2, .0003], [.75, -.2, .0003],
                                 [.75, .2, .0003], [.15, .2, .0003]])
        self.pile = self.backing + [0., 0., THICKNESS]
        self.quads = [[0, 1, 2, 3]]

    def score(self, shift=(0., 0., 0.)):
        return grade(self.pile + shift, self.backing + shift,
                     self.quads, 0.)

    def tuck_corner(self):
        self.pile[1] = [.65, -.1, .0003]
        self.backing[1] = [.65, -.1, THICKNESS + .0003]

    def test_unfolded_inside_full_credit(self):
        self.assertEqual(self.score()["score"], 1.)

    def test_unfolded_outside_half_credit_even_with_center_inside(self):
        self.assertEqual(self.score((.021, 0., 0.))["score"], .5)

    def test_inside_without_unfolding_has_no_credit(self):
        self.tuck_corner()
        self.assertEqual(self.score()["score"], 0.)

    def test_outside_without_unfolding_has_no_credit(self):
        self.tuck_corner()
        self.assertEqual(self.score((.021, 0., 0.))["score"], 0.)

    def test_flat_held_above_support_has_no_credit(self):
        self.assertEqual(self.score((0., 0., .02))["score"], 0.)

    def test_touching_inner_edge_is_inside(self):
        self.assertEqual(self.score((.02, 0., 0.))["score"], 1.)

    def test_crossing_one_edge_vertex_is_outside(self):
        self.pile[2, 1] = .221
        self.backing[2, 1] = .221
        self.assertEqual(self.score()["score"], .5)

    def test_native_triangles_match_quad_area_and_score(self):
        result = grade(self.pile, self.backing, [[0, 1, 2], [0, 2, 3]], 0.)
        self.assertEqual(result, self.score())


if __name__ == "__main__":
    unittest.main()

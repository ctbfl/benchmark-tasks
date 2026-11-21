"""CPU checks for contact location and historical clamp credit."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import BASE_ROOT_X, CLAMP_Y, EDGE_X, PAD_THICKNESS, PAD_X, PAD_Z, PARK_ANGLE, PLATE_THICKNESS, TABLE_Z
from score import Score, TABLE_BOTTOM, jaw_seating
from scene import HZ

BOARD = np.array([BASE_ROOT_X, 0., TABLE_Z, 0., 0., 0., 1.])
EMPTY_PATCH = np.empty((0, 7))
SUPPORT = np.array([[0., 0., -18.], [0., 0., 12.]])
CLEAR = np.zeros((2, 3))


def clamp_at(site, *, opened=False, off_square=False, working=False, shift=0.):
    y = (.025 if off_square else CLAMP_Y[site])
    x = BASE_ROOT_X+.005 + shift
    force = np.zeros((2, 2, 3))
    hand = np.zeros((2, 2, 3))
    if working:
        hand[0, 0, 2] = 6.
    links = []
    for sign, z in ((1, TABLE_Z + PLATE_THICKNESS), (-1, TABLE_BOTTOM)):
        orientation = Rotation.from_euler("y", -sign * PARK_ANGLE / 2)
        face = np.array([x, y, z + sign * (.011 if opened else 0.)])
        center = face + [0., 0., sign * PAD_THICKNESS / 2]
        links.append(np.r_[center - orientation.apply([PAD_X, 0., sign * PAD_Z]), orientation.as_quat()])
    if opened:
        return np.array(links), .92, force, [EMPTY_PATCH, EMPTY_PATCH], hand
    force[0, 0, 2], force[1, 1, 2] = 12., -12.
    upper = np.array([[x, y - (.008 if off_square else 0.), TABLE_Z + PLATE_THICKNESS, 0., 0., 1., 12.]])
    lower = np.array([[x, y, TABLE_BOTTOM, 0., 0., -1., 12.]])
    return np.array(links), .49, force, [upper, lower], hand


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.scorer = Score()
        self.step = 0

    def sample(self, samples, support=False, steps=1):
        left = SUPPORT if support else CLEAR
        for _ in range(steps):
            self.step += 1
            metric = self.scorer.update(self.step, BOARD, np.zeros(6), samples, left, left, [.05, .05], np.array([support, support]))
        return metric

    def attach_first(self):
        self.sample({"clamp_a": clamp_at(0, opened=True, working=True)}, support=True)
        first = {"clamp_a": clamp_at(0)}
        result = self.sample(first, support=True)
        self.assertEqual(result["credited_pairs"], {"clamp_a": 0})
        return first

    def test_supported_installation_and_untouched_retention(self):
        first = self.attach_first()
        self.sample(first | {"clamp_b": clamp_at(1, opened=True, working=True)}, support=True)
        both = first | {"clamp_b": clamp_at(1)}
        self.assertEqual(self.sample(both, support=True)["credited_pairs"], {"clamp_a": 0, "clamp_b": 1})
        self.assertEqual(self.sample(both, steps=2 * HZ - 1)["score"], 0.)
        self.assertEqual(self.sample(both)["score"], 1.)

    def test_unsupported_clamping_then_regrasping_and_touching(self):
        first = self.attach_first()
        self.sample(first | {"clamp_b": clamp_at(1, opened=True, working=True)})
        self.sample(first | {"clamp_b": clamp_at(1, working=True)})
        result = self.sample(first | {"clamp_b": clamp_at(1, working=True)}, support=True)
        self.assertEqual(result["seated_sites"], [0, 1])
        self.assertEqual(result["credited_pairs"], {"clamp_a": 0})
        self.assertEqual(self.sample(first | {"clamp_b": clamp_at(1)}, steps=2 * HZ)["score"], .5)

    def test_right_hand_contact_with_other_clamp_does_not_arm_the_event(self):
        self.attach_first()
        self.sample({"clamp_a": clamp_at(0, working=True), "clamp_b": clamp_at(1, opened=True)},
                    support=True)
        result = self.sample({"clamp_a": clamp_at(0), "clamp_b": clamp_at(1, working=True)},
                             support=True)
        self.assertEqual(result["credited_pairs"], {"clamp_a": 0})

    def test_world_envelope_passes_but_upper_contact_misses_square(self):
        self.sample({"clamp_a": clamp_at(0, opened=True, working=True)}, support=True)
        outside = clamp_at(0, off_square=True)
        links, hinge, forces, patches, _ = outside
        x, y, z = patches[0][0, :3]
        self.assertTrue(EDGE_X <= x <= EDGE_X + .065 and 0. < y < .02)
        self.assertTrue(abs(z - TABLE_Z - PLATE_THICKNESS) < .006)
        opened, site, loaded, *_ = jaw_seating(BOARD, links, hinge, forces, patches)
        self.assertEqual((opened, site, loaded), (None, None, True))
        self.sample({"clamp_a": outside}, support=True)
        self.assertEqual(self.sample({"clamp_a": outside}, steps=2 * HZ)["score"], 0.)

    def test_support_break_during_open_straddle(self):
        opened = {"clamp_a": clamp_at(0, opened=True, working=True)}
        self.sample(opened, support=True)
        self.sample(opened)
        self.sample(opened, support=True)
        closed = {"clamp_a": clamp_at(0, working=True)}
        self.assertFalse(self.sample(closed, support=True)["credited_pairs"])

    def test_load_redistribution_does_not_recredit_or_move_the_clamp(self):
        first = self.attach_first()
        links, hinge, forces, patches, hand = first["clamp_a"]
        unloaded_patch = patches[0].copy()
        unloaded_patch[:, 6] = 0.
        paused = {"clamp_a": (links, hinge, forces, [unloaded_patch, patches[1]], hand)}
        result = self.sample(paused, steps=2 * HZ)
        self.assertEqual(result["credited_pairs"], {"clamp_a": 0})
        self.assertEqual(result["score"], 0.)
        self.assertEqual(self.sample(first, steps=2 * HZ)["score"], .5)

    def test_spring_closure_can_finish_after_handle_release(self):
        opened = {"clamp_a": clamp_at(0, opened=True, working=True)}
        self.sample(opened, support=True)
        links, hinge, forces, patches, hand = clamp_at(0)
        empty = [p.copy() for p in patches]
        for p in empty:
            p[:, 6] = 0.
        self.sample({"clamp_a": (links, hinge, forces, empty, hand)}, support=True, steps=HZ)
        seated = self.sample({"clamp_a": clamp_at(0)}, support=True)
        self.assertEqual(seated["credited_pairs"], {"clamp_a": 0})

    def test_support_contact_has_no_calibrated_force_threshold(self):
        global SUPPORT
        original = SUPPORT.copy()
        SUPPORT[:] = 0.
        try:
            self.attach_first()
        finally:
            SUPPORT[:] = original

    def test_relocation_or_substitution_requires_a_new_installation(self):
        self.attach_first()
        moved = {"clamp_a": clamp_at(0, shift=.005)}
        self.assertFalse(self.sample(moved, steps=2 * HZ)["credited_pairs"])
        self.setUp()
        self.attach_first()
        replacement = {"clamp_a": clamp_at(1, opened=True), "clamp_b": clamp_at(0)}
        self.assertEqual(self.sample(replacement, steps=2 * HZ)["score"], 0.)


if __name__ == "__main__":
    unittest.main()

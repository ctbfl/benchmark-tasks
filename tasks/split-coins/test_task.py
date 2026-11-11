"""Scoring and layout checks that run without Isaac Sim: python -m unittest -v test_task"""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from build_assets import COINS, FRAMES
from scene import COIN_NAMES, COINS_PER_KIND, CONFIG
from task_runtime import evaluate, inside_frame

FLAT = [0., 0., 0., 1.]


def placed():
    """Every coin resting flat near the centre of its own frame."""
    poses = {}
    for i, name in enumerate(COIN_NAMES):
        (cx, cy), _ = FRAMES[name.split("_")[0]]
        poses[name] = [cx + .001*i, cy, .748, *FLAT]
    return poses


class ScoringTest(unittest.TestCase):
    def test_all_sorted_scores_one(self):
        result = evaluate(placed())
        self.assertEqual(result["score"], 1.)
        self.assertEqual(result["pennies_in_left_frame"], COINS_PER_KIND)
        self.assertEqual(result["quarters_in_right_frame"], COINS_PER_KIND)

    def test_one_coin_in_the_wrong_frame_scores_zero(self):
        poses = placed()
        poses["penny_0"][:2] = FRAMES["quarter"][0]
        self.assertEqual(evaluate(poses)["score"], 0.)

    def test_whole_coin_must_be_inside_the_tape_inner_edge(self):
        (cx, cy), (sx, _) = FRAMES["quarter"]
        r = COINS["quarter"]["radius"]
        self.assertTrue(inside_frame(np.array([cx + sx/2 - r - .0005, cy, .748, *FLAT]), "quarter"))
        self.assertFalse(inside_frame(np.array([cx + sx/2 - r + .0005, cy, .748, *FLAT]), "quarter"))

    def test_tilted_coin_uses_its_projected_footprint(self):
        (cx, cy), (_, sy) = FRAMES["penny"]
        standing = Rotation.from_euler("x", 90, degrees=True).as_quat().tolist()
        # On its edge with the face normal along Y, the coin spans only its thickness in Y.
        self.assertTrue(inside_frame(np.array([cx, cy + sy/2 - .002, .758, *standing]), "penny"))
        self.assertFalse(inside_frame(np.array([cx, cy + sy/2 - .002, .748, *FLAT]), "penny"))


class DemoFilesTest(unittest.TestCase):
    def test_actions_fit_the_interface_and_time_limit(self):
        actions = np.load(Path(__file__).parent / "demo_actions.npy")
        self.assertEqual(actions.shape[1], 14)
        self.assertTrue(np.isfinite(actions).all())
        self.assertLessEqual(len(actions), CONFIG["episode"]["max_control_steps"])
        self.assertEqual(CONFIG["episode"]["max_control_steps"] / CONFIG["simulation"]["control_hz"], 120)


if __name__ == "__main__":
    unittest.main()

"""Scoring and layout checks that run without Isaac Sim: python -m unittest -v test_task"""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from build_assets import COINS, FRAMES, JAR_FLOOR, JAR_HEIGHT, JAR_INNER
from scene import COIN_NAMES, COINS_PER_KIND, CONFIG, JAR_XY
from score import score
from task_runtime import evaluate, inside_frame

FLAT = [0., 0., 0., 1.]


def placed():
    """Every coin resting flat near the centre of its own frame."""
    poses = {"jar": [*JAR_XY, .7475, *FLAT]}
    for i, name in enumerate(COIN_NAMES):
        (cx, cy), _ = FRAMES[name.split("_")[0]]
        poses[name] = [cx + .001*i, cy, .748, *FLAT]
    return poses


def jarred(center, rotation):
    """Ten coins in two columns inside a jar, transformed together into world space."""
    poses = {"jar": [*center, *rotation.as_quat()]}
    for i, name in enumerate(COIN_NAMES):
        local = [.014 if i % 2 else -.014, 0., .01 + .005*(i//2)]
        poses[name] = [*(np.asarray(center) + rotation.apply(local)), *rotation.as_quat()]
    return poses


class ScoringTest(unittest.TestCase):
    def test_coins_in_jar_in_either_frame_earn_no_credit(self):
        for kind in ("penny", "quarter"):
            for angle in (0, 45, 90, 180):
                with self.subTest(frame=kind, jar_tilt_degrees=angle):
                    rotation = Rotation.from_euler("y", angle, degrees=True)
                    poses = jarred([*FRAMES[kind][0], .85], rotation)
                    result = evaluate(poses)
                    self.assertEqual(result["score"], 0.)
                    self.assertEqual(result["pennies_in_left_frame"], 0)
                    self.assertEqual(result["quarters_in_right_frame"], 0)
                    self.assertEqual(score([{"time_s": 120., "poses": poses}]), 0.)

    def test_poured_coin_earns_credit_while_jarred_coins_do_not(self):
        for jar_kind, poured_kind in (("penny", "quarter"), ("quarter", "penny")):
            with self.subTest(jar_frame=jar_kind):
                poses = jarred([*FRAMES[jar_kind][0], .7475], Rotation.identity())
                poses[poured_kind + "_0"] = [*FRAMES[poured_kind][0], .748, *FLAT]
                result = evaluate(poses)
                self.assertEqual(result["score"], .1)
                self.assertEqual(sum(result["inside"].values()), 1)

    def test_jar_checks_local_height_and_radius(self):
        poses = placed()
        poses["jar"] = [*FRAMES["penny"][0], .7475, *FLAT]
        for local, expected in (([0., 0., .02], .9),
                                ([0., 0., JAR_FLOOR / 2], 1.),
                                ([0., 0., JAR_HEIGHT + .02], 1.),
                                ([JAR_INNER + COINS["penny"]["radius"] + .005, 0., .02], 1.)):
            with self.subTest(local_position=local):
                poses["penny_0"][:3] = (np.asarray(poses["jar"][:3]) + local).tolist()
                self.assertEqual(evaluate(poses)["score"], expected)

    def test_all_sorted_scores_one(self):
        result = evaluate(placed())
        self.assertEqual(result["score"], 1.)
        self.assertEqual(result["pennies_in_left_frame"], COINS_PER_KIND)
        self.assertEqual(result["quarters_in_right_frame"], COINS_PER_KIND)

    def test_one_coin_in_the_wrong_frame_retains_other_coins_credit(self):
        for name, wrong_kind in (("penny_0", "quarter"), ("quarter_0", "penny")):
            with self.subTest(coin=name):
                poses = placed()
                poses[name][:2] = FRAMES[wrong_kind][0]
                self.assertEqual(evaluate(poses)["score"], .9)

    def test_each_correct_coin_earns_one_tenth(self):
        for count in range(len(COIN_NAMES) + 1):
            with self.subTest(correct_count=count):
                poses = placed()
                for name in COIN_NAMES[count:]:
                    poses[name][:2] = [.54, 0.]
                result = evaluate(poses)
                self.assertEqual(result["score"], count / 10)
                self.assertEqual(result["pennies_in_left_frame"] + result["quarters_in_right_frame"], count)

    def test_partly_outside_coin_does_not_earn_credit(self):
        poses = placed()
        (cx, cy), (sx, _) = FRAMES["quarter"]
        poses["quarter_0"][:2] = [cx + sx/2 - COINS["quarter"]["radius"] + .0005, cy]
        self.assertEqual(evaluate(poses)["score"], .9)

    def test_offline_score_preserves_partial_credit_at_the_time_limit(self):
        poses = placed()
        poses["penny_0"][:2] = FRAMES["quarter"][0]
        self.assertEqual(score([{"time_s": 120., "poses": poses}]), .9)
        self.assertEqual(score([{"time_s": 120.05, "poses": poses}]), 0.)

    def test_offline_score_uses_final_state_not_peak_progress(self):
        final = placed()
        final["penny_0"][:2] = [.54, 0.]
        samples = [{"time_s": 100., "poses": placed()}, {"time_s": 120., "poses": final}]
        self.assertEqual(score(samples), .9)

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

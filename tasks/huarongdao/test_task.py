"""Check legal moves and scoring boundaries independently of the simulator."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from task import INITIAL, GOAL, ROTATION, cell, solve, score


class PuzzleTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.array([np.r_[cell(i), ROTATION] for i in range(8)])

    def result(self, time=120, contacts=None):
        return score(self.poses, np.zeros(8) if contacts is None else contacts, time)["score"]

    def test_shortest_solution_is_legal(self):
        board = list(INITIAL)
        moves = solve()
        self.assertEqual(moves, [5, 4, 7, 8])
        for tile in moves:
            a, b = board.index(tile), board.index(0)
            self.assertEqual(abs(a//3-b//3)+abs(a%3-b%3), 1)
            board[a], board[b] = 0, tile
        self.assertEqual(tuple(board), GOAL)

    def test_initial_and_swapped_fail(self):
        self.poses[:, :3] = [cell(INITIAL.index(i)) for i in range(1, 9)]
        self.assertEqual(self.result(), 0)
        self.poses[:, :3] = [cell(i) for i in range(8)]
        self.poses[[0, 1]] = self.poses[[1, 0]]
        self.assertEqual(self.result(), 0)

    def test_tolerance_for_only_one_tile(self):
        self.assertEqual(self.result(), 1)
        self.poses[7, 0] += .010
        self.assertEqual(self.result(), 1)
        self.poses[6, 0] += .0031
        self.assertEqual(self.result(), 0)
        self.poses[6, 0] -= .0031
        self.poses[7, 0] += .0001
        self.assertEqual(self.result(), 0)

    def test_lifted_rotated_held_and_late_fail(self):
        self.assertEqual(self.result(time=120.001), 0)
        self.assertEqual(self.result(contacts=np.ones(8)), 0)
        self.poses[0, 2] += .002
        self.assertEqual(self.result(), 0)
        self.poses[0, 2] -= .002
        self.poses[0, 3:] = Rotation.from_euler("x", 180, degrees=True).as_quat()
        self.assertEqual(self.result(), 0)

    def test_bad_inputs_fail_fast(self):
        self.poses[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.result()
        with self.assertRaises(ValueError):
            solve((0,)*9)


if __name__ == "__main__":
    unittest.main()

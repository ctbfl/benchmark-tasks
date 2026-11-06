"""Submission semantics and arithmetic checks; no simulator or model calls."""

import unittest

from task import Calculator, score


class TaskTests(unittest.TestCase):
    def test_equals_is_required(self):
        calculator = Calculator()
        for key in ("5", "0"):
            calculator.press(key)
        self.assertEqual(score(calculator.submitted, None, 120), 0)
        calculator.press("equals")
        self.assertEqual(score(calculator.submitted, 90, 120), 1)

    def test_triangle_arithmetic(self):
        calculator = Calculator()
        for key in ("1", "2", "dot", "5", "multiply", "8", "divide", "2", "equals"):
            calculator.press(key)
        self.assertEqual(calculator.submitted, "50")

    def test_wrong_answer_is_final(self):
        calculator = Calculator()
        for key in ("1", "0", "0", "equals", "C", "5", "0", "equals"):
            calculator.press(key)
        self.assertEqual(calculator.submitted, "100")
        self.assertEqual(score(calculator.submitted, 80, 120), 0)

    def test_numeric_score_and_deadline(self):
        for text in ("50", "50.0", "050"):
            self.assertEqual(score(text, 120, 120), 1)
        for text in (None, "Error", "nan", "inf", "100", "49.9999", "50.0001"):
            self.assertEqual(score(text, 10, 120), 0)
        for time in (None, -1, 120.0001, float("nan")):
            self.assertEqual(score("50", time, 120), 0)

    def test_division_by_zero(self):
        calculator = Calculator()
        for key in ("5", "divide", "0", "equals"):
            calculator.press(key)
        self.assertEqual(calculator.submitted, "Error")
        self.assertEqual(score(calculator.submitted, 10, 120), 0)


if __name__ == "__main__":
    unittest.main()

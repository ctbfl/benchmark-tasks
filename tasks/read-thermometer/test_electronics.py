"""Logic and authored asset checks; physical contact checks run in the task scene."""

import json
from decimal import Decimal
import unittest

from pxr import Usd, UsdGeom, UsdPhysics

from electronics import (
    ASSETS, CALCULATOR_LAYOUT, CalculatorState, TimerState, pressed_buttons,
    update_calculator_display, update_timer_display,
)


def click(state, *keys):
    state.update_buttons(set(keys))
    state.update_buttons(set())


class ElectronicsTests(unittest.TestCase):
    def test_countdown_pause_resume_and_zero(self):
        timer = TimerState()
        self.assertEqual(timer.display, "00:15")
        timer.advance(2.5)
        self.assertEqual(timer.display, "00:13")
        click(timer, "START")
        timer.advance(50)
        self.assertEqual(timer.remaining_seconds, 12.5)
        click(timer, "START")
        timer.advance(12.5)
        self.assertEqual(timer.display, "00:00")
        self.assertFalse(timer.running)
        click(timer, "START")
        timer.advance(30)
        self.assertEqual(timer.remaining_seconds, 0)

    def test_independent_wrap_and_reset_chord(self):
        timer = TimerState(3599, False)
        click(timer, "M")
        self.assertEqual(timer.display, "00:59")
        click(timer, "S")
        self.assertEqual(timer.display, "00:00")
        click(timer, "M")
        self.assertEqual(timer.display, "01:00")
        click(timer, "START")
        click(timer, "M", "S")
        self.assertEqual(timer.display, "00:00")
        self.assertFalse(timer.running)

    def test_held_key_is_single_press_and_chord_overlap(self):
        timer = TimerState(0, False)
        timer.update_buttons({"M"})
        timer.update_buttons({"M"})
        self.assertEqual(timer.display, "01:00")
        timer.update_buttons({"M", "S"})
        self.assertEqual(timer.display, "00:00")
        timer.update_buttons({"S"})
        self.assertEqual(timer.display, "00:00")
        timer.update_buttons(set())
        click(timer, "S")
        self.assertEqual(timer.display, "00:01")

    def test_physical_press_hysteresis(self):
        self.assertEqual(pressed_buttons({"4": -.001}), set())
        self.assertEqual(pressed_buttons({"4": -.0017}), {"4"})
        self.assertEqual(pressed_buttons({"4": -.001}, {"4"}), {"4"})
        self.assertEqual(pressed_buttons({"4": -.0006}, {"4"}), set())
        self.assertEqual(pressed_buttons({"4": -.0009}, press_threshold=-.0008,
                                         release_threshold=-.00025), {"4"})

    def test_temperature_entry_and_repeated_equals(self):
        calculator = CalculatorState()
        for key in ("4", "5", "="):
            click(calculator, key)
        self.assertEqual(calculator.display, "45")
        self.assertEqual(calculator.last_submission, Decimal(45))
        self.assertEqual(calculator.submission_count, 1)
        click(calculator, "=")
        self.assertEqual(calculator.last_submission, Decimal(45))
        self.assertEqual(calculator.submission_count, 2)
        click(calculator, "1")
        self.assertEqual(calculator.display, "1")
        self.assertIsNone(calculator.last_submission)

    def test_arithmetic_decimal_sign_clear_entry_percent_root_and_error(self):
        cases = [(("1", "2", "+", "3", "="), "15"),
                 (("9", "-", "1", "2", "="), "-3"),
                 (("6", "*", "7", "="), "42"),
                 (("1", "/", "4", "="), "0.25"),
                 (("1", ".", "2", "+", "3", ".", "4", "="), "4.6"),
                 (("2", "+", "3", "=", "="), "8"),
                 (("2", "+/-", "*", "3", "="), "-6"),
                 (("4", "+", "9", "CE", "5", "="), "9"),
                 (("4", "C", "5", "="), "5"),
                 (("5", "0", "%"), "0.5"),
                 (("8", "1", "SQRT"), "9"),
                 (("1", "+/-", "SQRT"), "Error"),
                 (("1", "/", "0", "="), "Error"),
                 (("9", "0", "0", "0", "0", "0", "0", "0", "*", "9", "="), "Error")]
        for sequence, expected in cases:
            with self.subTest(sequence=sequence):
                calculator = CalculatorState()
                for key in sequence:
                    click(calculator, key)
                self.assertEqual(calculator.display, expected)

    def test_asset_joints_and_display_updates(self):
        metadata = json.loads((ASSETS / "metadata.json").read_text())
        for name, state, updater, keys in (
            ("timer", TimerState(), update_timer_display, {"M", "S", "START"}),
            ("calculator", CalculatorState(), update_calculator_display,
             {key for row in CALCULATOR_LAYOUT for key in row}),
        ):
            stage = Usd.Stage.Open(str(ASSETS / f"{name}.usda"))
            root = stage.GetDefaultPrim()
            self.assertTrue(root.HasAPI(UsdPhysics.ArticulationRootAPI))
            self.assertEqual(set(metadata[name]["buttons"]), keys)
            for key, value in metadata[name]["buttons"].items():
                joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(
                    f"{root.GetPath()}/{'Joints' if name == 'calculator' else 'joints'}/{value['joint']}"))
                self.assertAlmostEqual(joint.GetLowerLimitAttr().Get(), -value['travel_m'])
                self.assertEqual(joint.GetUpperLimitAttr().Get(), 0)
                self.assertEqual(joint.GetAxisAttr().Get(), "Z")
                self.assertTrue(stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0]).HasAPI(UsdPhysics.RigidBodyAPI))
            if name == "calculator":
                for key in ("4", "5", "="):
                    click(state, key)
            else:
                state.advance(15)
            updater(stage, str(root.GetPath()), state)
            character = "5" if name == "calculator" else "0"
            index = 7 if name == "calculator" else 3
            display = (f"{root.GetPath()}/base/display/digit_{index}_" if name == "calculator"
                       else f"{root.GetPath()}/body/display/digit_{index}/")
            self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath(display + "a")).GetVisibilityAttr().Get(), "inherited")
            self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath(display + "g")).GetVisibilityAttr().Get(),
                             "inherited" if character == "5" else "invisible")


if __name__ == "__main__":
    unittest.main()

"""Fast checks of transport timing, conservation and task completion."""

import math
import unittest

from config import (
    CLOSED_ANGLE, CUP_CAPACITY_G, DENSITY_G_M3, DT, MAX_ANGLE, MAX_FLOW_G_S, SETTLE_S,
    RESERVOIR_FILL_HEIGHT, RESERVOIR_FLOOR_Z, RESERVOIR_INITIAL_G, RESERVOIR_RADIUS,
)
from water import WaterTask, flight_time, flow_g_s


class WaterTests(unittest.TestCase):
    def test_symmetric_monotonic_valve(self):
        for i in range(11):
            angle = MAX_ANGLE * i / 10
            self.assertEqual(flow_g_s(angle), flow_g_s(-angle))
            self.assertLessEqual(flow_g_s(angle), flow_g_s(angle + 0.01))
        self.assertEqual(flow_g_s(0), 0)
        self.assertEqual(flow_g_s(MAX_ANGLE), MAX_FLOW_G_S)

    def test_central_dead_zone(self):
        for angle in (-CLOSED_ANGLE,-CLOSED_ANGLE/2,0,CLOSED_ANGLE/2,CLOSED_ANGLE):
            self.assertEqual(flow_g_s(angle),0)
        self.assertGreater(flow_g_s(CLOSED_ANGLE+0.001),0)

    def test_flight_delay_and_tail(self):
        water = WaterTask(3)
        first_hit = flight_time(water.surface_z)
        for _ in range(round(0.1 / DT)):
            water.step(MAX_ANGLE, DT)
        self.assertEqual(water.cup_g, 0)
        self.assertAlmostEqual(water.airborne_g, 3)
        self.assertAlmostEqual(water.reservoir_g, RESERVOIR_INITIAL_G - 3)
        while water.time + DT < first_hit:
            water.step(0, DT)
            self.assertEqual(water.cup_g, 0)
        water.step(0, DT)
        self.assertGreater(water.cup_g, 0)
        self.assertFalse(water.success)
        for _ in range(round((0.3 + SETTLE_S) / DT)):
            water.step(0, DT)
        self.assertAlmostEqual(water.cup_g, 3)
        self.assertEqual(water.airborne_g, 0)
        self.assertTrue(water.success)

    def test_conservation_and_cylinder_volume(self):
        water = WaterTask()
        for i in range(1200):
            water.step(MAX_ANGLE * math.sin(i * DT * 5), DT)
            self.assertAlmostEqual(water.emitted_g, water.cup_g + water.airborne_g + water.spilled_g, places=9)
            volume = sum(math.pi * r*r * length for _, r, length in water.cylinders())
            self.assertAlmostEqual(volume * DENSITY_G_M3, water.airborne_g, places=9)
            stored_volume = (water.reservoir_surface_z - RESERVOIR_FLOOR_Z) * math.pi * RESERVOIR_RADIUS**2
            self.assertAlmostEqual(stored_volume * DENSITY_G_M3 + water.cup_g + water.airborne_g + water.spilled_g,
                                   RESERVOIR_INITIAL_G, places=8)

    def test_reservoir_empty_and_refill(self):
        water = WaterTask()
        for _ in range(math.ceil(RESERVOIR_INITIAL_G / MAX_FLOW_G_S) + 2):
            water.step(MAX_ANGLE, 1.0)
        self.assertEqual(water.flow, 0)
        self.assertEqual(water.reservoir_g, 0)
        self.assertAlmostEqual(water.cup_g + water.spilled_g, RESERVOIR_INITIAL_G)
        water.reset(75)
        self.assertEqual(water.reservoir_g, RESERVOIR_INITIAL_G)
        self.assertAlmostEqual(water.reservoir_surface_z, RESERVOIR_FLOOR_Z + RESERVOIR_FILL_HEIGHT)

    def test_overflow_and_reset(self):
        water = WaterTask(CUP_CAPACITY_G)
        for _ in range(round(22 / DT)):
            water.step(MAX_ANGLE, DT)
        for _ in range(round(1 / DT)):
            water.step(0, DT)
        self.assertAlmostEqual(water.cup_g, CUP_CAPACITY_G)
        self.assertGreater(water.spilled_g, 0)
        self.assertFalse(water.success)
        self.assertAlmostEqual(water.emitted_g, water.cup_g + water.spilled_g)
        water.reset(50)
        self.assertEqual(water.emitted_g + water.airborne_g + water.cup_g + water.spilled_g, 0)
        self.assertEqual(water.reading_g, 0)

    def test_step_convergence(self):
        results = []
        for dt in (1/120, 1/240, 1/480):
            water = WaterTask()
            for i in range(round(0.6 / dt)):
                water.step(MAX_ANGLE if i*dt < 0.4 else 0, dt)
            results.append(water.cup_g)
        self.assertLess(max(results) - min(results), 0.1)

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            WaterTask(CUP_CAPACITY_G + 1)
        with self.assertRaises(ValueError):
            WaterTask().step(float("nan"), DT)
        with self.assertRaises(ValueError):
            WaterTask().step(0, 0)


if __name__ == "__main__":
    unittest.main()

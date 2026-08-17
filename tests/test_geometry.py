import unittest

import numpy as np

from coat_analyzer.geometry import solve_harmonic_coordinate


class HarmonicCoordinateTests(unittest.TestCase):
    def test_rectangle_maps_left_to_right(self):
        polygon = [[10, 10], [90, 10], [90, 50], [10, 50]]
        result = solve_harmonic_coordinate(
            polygon, left_edges=[3], right_edges=[1], image_size=(100, 60), max_side=120
        )
        self.assertLess(float(result.coordinate[30, 15]), 0.2)
        self.assertAlmostEqual(float(result.coordinate[30, 50]), 0.5, delta=0.08)
        self.assertGreater(float(result.coordinate[30, 85]), 0.8)

    def test_curved_band_stays_defined_inside_roi(self):
        polygon = [[10, 50], [20, 20], [50, 8], [80, 20], [90, 50], [75, 45], [50, 35], [25, 45]]
        result = solve_harmonic_coordinate(
            polygon, left_edges=[0], right_edges=[3], image_size=(100, 70), max_side=120
        )
        values = result.coordinate[result.mask]
        self.assertTrue(np.isfinite(values).all())
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)


if __name__ == "__main__":
    unittest.main()

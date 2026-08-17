import unittest

import numpy as np
from PIL import Image

from coat_analyzer.analysis import analyze_image, rgb_to_lstar


class AnalysisTests(unittest.TestCase):
    def test_lstar_orders_black_gray_white(self):
        rgb = np.array([[[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]], dtype=np.float32)
        values = rgb_to_lstar(rgb)[0]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertAlmostEqual(float(values[2]), 100.0, places=3)

    def test_three_regions_report_expected_darkness_order(self):
        pixels = np.zeros((60, 120, 3), dtype=np.uint8)
        pixels[:, :40] = 220
        pixels[:, 40:80] = 40
        pixels[:, 80:] = 150
        image = Image.fromarray(pixels)
        output = analyze_image(
            image=image,
            polygon=[[5, 5], [115, 5], [115, 55], [5, 55]],
            left_edges=[3], right_edges=[1], exclusions=[], gray_card=[],
            split_left=1 / 3, split_right=2 / 3, max_side=160,
        )
        regions = output.metrics["regions"]
        self.assertGreater(regions["middle"]["mean_darkness"], regions["right"]["mean_darkness"])
        self.assertGreater(regions["right"]["mean_darkness"], regions["left"]["mean_darkness"])

    def test_gray_card_and_exclusion_are_recorded(self):
        pixels = np.full((70, 130, 3), 140, dtype=np.uint8)
        pixels[10:60, 105:125] = 200
        image = Image.fromarray(pixels)
        output = analyze_image(
            image=image,
            polygon=[[5, 5], [95, 5], [95, 65], [5, 65]],
            left_edges=[3], right_edges=[1],
            exclusions=[[[40, 20], [55, 20], [55, 40], [40, 40]]],
            gray_card=[[105, 10], [125, 10], [125, 60], [105, 60]],
            split_left=0.4, split_right=0.6, gray_target_l=50.0, max_side=160,
        )
        self.assertTrue(output.metrics["gray_card"]["available"])
        self.assertGreater(output.metrics["excluded_pixel_count"], 0)
        middle = output.metrics["regions"]["middle"]
        self.assertNotAlmostEqual(middle["mean_darkness"], middle["mean_corrected_darkness"])
        self.assertNotEqual(
            middle["histogram_l_raw"]["counts"],
            middle["histogram_l_corrected"]["counts"],
        )

    def test_white_black_ranges_leave_gray_pixels_unclassified(self):
        pixels = np.zeros((50, 90, 3), dtype=np.uint8)
        pixels[:, :30] = 240
        pixels[:, 30:60] = 130
        pixels[:, 60:] = 20
        output = analyze_image(
            image=Image.fromarray(pixels),
            polygon=[[2, 2], [87, 2], [87, 47], [2, 47]],
            left_edges=[3], right_edges=[1], exclusions=[], gray_card=[],
            split_left=1 / 3, split_right=2 / 3,
            white_depth_min=0, white_depth_max=20,
            black_depth_min=80, black_depth_max=100, max_side=120,
        )
        areas = output.metrics["hair_classification"]["modes"]["raw"]
        self.assertGreater(areas["regions"]["left"]["white_area_fraction"], 0.9)
        self.assertGreater(areas["regions"]["middle"]["unclassified_area_fraction"], 0.9)
        self.assertGreater(areas["regions"]["right"]["black_area_fraction"], 0.9)
        self.assertEqual(set(np.unique(output.classification_maps["raw"])), {0, 1, 2})

    def test_overlapping_hair_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能重叠"):
            analyze_image(
                image=Image.new("RGB", (40, 40), "gray"),
                polygon=[[2, 2], [37, 2], [37, 37], [2, 37]],
                left_edges=[3], right_edges=[1], exclusions=[], gray_card=[],
                split_left=0.4, split_right=0.6,
                white_depth_min=0, white_depth_max=70,
                black_depth_min=60, black_depth_max=100,
            )


if __name__ == "__main__":
    unittest.main()

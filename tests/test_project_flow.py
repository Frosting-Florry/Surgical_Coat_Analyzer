import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coat_analyzer.exporter import export_project
from coat_analyzer.project_store import (
    assign_group,
    create_group,
    create_project,
    get_image,
    list_images,
    open_project,
    save_annotation,
)


class ProjectFlowTests(unittest.TestCase):
    def test_create_group_annotate_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "images"
            source.mkdir()
            pixels = np.full((60, 100, 3), 210, dtype=np.uint8)
            pixels[:, 35:65] = 45
            Image.fromarray(pixels).save(source / "mouse-01.jpg")
            database = create_project(root / "experiment.sqlite", "Test", source)
            self.assertEqual(open_project(database)["name"], "Test")
            image = list_images(database)[0]
            group = create_group(database, "Treatment")
            assign_group(database, group["id"], [image["id"]])
            annotation = {
                "roi": [[5, 5], [95, 5], [95, 55], [5, 55]],
                "left_edges": [3], "right_edges": [1], "exclusions": [], "gray_card": [],
                "split_left": 0.4, "split_right": 0.6,
                "white_depth_min": 2.0, "white_depth_max": 22.0,
                "black_depth_min": 72.0, "black_depth_max": 98.0,
            }
            save_annotation(database, image["id"], annotation, "reviewed")
            saved = get_image(database, image["id"])
            self.assertEqual(saved["group_name"], "Treatment")
            self.assertEqual(saved["left_edges"], [3])
            self.assertEqual(saved["white_depth_max"], 22.0)
            self.assertEqual(saved["black_depth_min"], 72.0)
            output = export_project(database, root)
            self.assertTrue((output / "region_summary.csv").is_file())
            self.assertTrue((output / "lstar_histograms.csv").is_file())
            self.assertTrue((output / "color_depth_histograms.csv").is_file())
            self.assertTrue((output / "center_darkening_index.csv").is_file())
            self.assertTrue((output / "hair_area_summary.csv").is_file())
            self.assertEqual(len(list((output / "annotated_images").glob("*.png"))), 1)
            self.assertEqual(len(list((output / "histogram_plots" / "raw" / "individual").glob("*.png"))), 1)
            self.assertEqual(len(list((output / "histogram_plots" / "raw" / "groups").glob("*.png"))), 1)
            with (output / "region_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["region"] for row in rows}, {"left", "middle", "right"})
            self.assertTrue(all(row["group"] == "Treatment" for row in rows))

    def test_default_database_is_created_in_image_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "images"
            source.mkdir()
            Image.new("RGB", (20, 20), "gray").save(source / "mouse.png")
            database = create_project(None, "Default Project", source)
            self.assertEqual(database.parent, source.resolve())
            self.assertEqual(database.name, "Default Project.sqlite")

    def test_version_one_project_is_migrated_for_hair_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "legacy.sqlite"
            conn = sqlite3.connect(database)
            conn.executescript("""
                CREATE TABLE project (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                    app_version TEXT NOT NULL, schema_version INTEGER NOT NULL,
                    source_root TEXT NOT NULL, standard_left REAL NOT NULL,
                    standard_right REAL NOT NULL, use_standard_export INTEGER NOT NULL,
                    gray_target_l REAL NOT NULL
                );
                CREATE TABLE annotation (
                    image_id INTEGER PRIMARY KEY, roi_json TEXT, left_edges_json TEXT,
                    right_edges_json TEXT, exclusions_json TEXT, gray_card_json TEXT,
                    split_left REAL NOT NULL, split_right REAL NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO project VALUES
                    ('p','Legacy','2026-01-01','0.1.0',1,'.',0.4,0.6,1,50.0);
            """)
            conn.commit()
            conn.close()
            self.assertEqual(open_project(database)["schema_version"], 2)
            conn = sqlite3.connect(database)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(annotation)")}
            conn.close()
            self.assertTrue({"white_depth_min", "white_depth_max", "black_depth_min", "black_depth_max"} <= columns)


if __name__ == "__main__":
    unittest.main()

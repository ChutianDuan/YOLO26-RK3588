"""Small regression checks for frame alignment, metrics and ignored regions."""
import json
from pathlib import Path
import tempfile
import unittest

import compare
from evaluate_bdd100k_tracking import ObjectRecord


class ComparisonTests(unittest.TestCase):
    def test_ap_and_operating_point(self):
        labels = {0: [ObjectRecord(0, "a", "person", (0, 0, 10, 10)),
                      ObjectRecord(0, "b", "person", (30, 30, 40, 40))]}
        predictions = {0: [ObjectRecord(0, "1", "person", (0, 0, 10, 10), .9),
                           ObjectRecord(0, "2", "person", (0, 0, 10, 10), .8),
                           ObjectRecord(0, "3", "person", (30, 30, 40, 40), .7)]}
        score = compare.score_predictions(predictions, labels, {0: []})
        person = score["per_class"]["person"]
        self.assertAlmostEqual(person["ap50"], (51 + 50 * 2 / 3) / 101, places=6)
        self.assertEqual(person["operating_point"]["false_positives"], 1)
        self.assertEqual(person["operating_point"]["true_positives"], 2)
        self.assertEqual(person["operating_point"]["f1"], .8)

    def test_normal_match_precedes_ignore_and_floor_only_class(self):
        labels = {0: [ObjectRecord(0, "a", "person", (0, 0, 10, 10))]}
        crowds = {0: [ObjectRecord(0, "c", "person", (0, 0, 10, 10)),
                      ObjectRecord(0, "d", compare.IGNORE, (30, 30, 40, 40))]}
        predictions = {0: [ObjectRecord(0, "1", "person", (0, 0, 10, 10), .9),
                           ObjectRecord(0, "2", "car", (30, 30, 40, 40), .8),
                           ObjectRecord(0, "3", "bike", (50, 50, 60, 60), .002)]}
        score = compare.score_predictions(predictions, labels, crowds)
        self.assertEqual(score["operating_point"]["true_positives"], 1)
        self.assertEqual(score["operating_point"]["false_positives"], 0)
        self.assertEqual(score["ignored_predictions_at_025"], 1)
        self.assertIsNone(score["per_class"]["bike"]["ap50"])
        self.assertEqual(score["per_class"]["bike"]["operating_point"]["predictions"], 0)

    def test_missing_labeled_frame_and_duplicate_frame_fail(self):
        manifest = {"video": {"width": 100, "height": 100, "decoded_frame_count": 3},
                    "classes": ["person"], "annotated_class_ids": [0]}
        row = {"type": "frame", "frame_index": 0, "width": 100, "height": 100,
               "is_detection_frame": True, "detections": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Missing labeled"):
                compare.load_predictions(path, manifest, {0, 2})
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                compare.load_predictions(path, manifest, {0})

    def test_tracking_and_nonfinite_results_fail(self):
        manifest = {"video": {"width": 100, "height": 100, "decoded_frame_count": 3},
                    "classes": ["person"], "annotated_class_ids": [0]}
        row = {"type": "frame", "frame_index": 0, "width": 100, "height": 100,
               "is_detection_frame": False, "detections": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "disable LK"):
                compare.load_predictions(path, manifest, {0})
            row["is_detection_frame"] = True
            row["detections"] = [{"class_id": 0, "score": float("nan"), "bbox": [0, 0, 10, 10]}]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                compare.load_predictions(path, manifest, {0})


if __name__ == "__main__":
    unittest.main()

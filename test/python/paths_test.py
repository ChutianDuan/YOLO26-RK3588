import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from paths import WORKSPACE, TEST_ROOT, resolve_artifact, relocate_records, load_history


class HistoricalPaths(unittest.TestCase):
    def test_validation_prefix_and_boundary(self):
        self.assertEqual(resolve_artifact(WORKSPACE / 'yolo_run/validation/a/b.jsonl'), TEST_ROOT / 'validation/a/b.jsonl')
        sibling = WORKSPACE / 'yolo_run/validation-other/a'
        self.assertEqual(resolve_artifact(sibling), sibling)

    def test_camera_and_current_paths(self):
        self.assertEqual(resolve_artifact(WORKSPACE / 'Camera/x.jpg'), TEST_ROOT / 'camera/x.jpg')
        self.assertEqual(resolve_artifact(TEST_ROOT / 'validation/x'), TEST_ROOT / 'validation/x')

    def test_provenance_command_is_preserved(self):
        old = str(WORKSPACE / 'yolo_run/validation/x.jsonl')
        record = {'runs': [{'output': old, 'command': ['binary', old]}]}
        result = relocate_records(record)
        self.assertEqual(record['runs'][0]['output'], old)
        self.assertEqual(result['runs'][0]['command'], ['binary', old])
        self.assertEqual(result['runs'][0]['output'], str(TEST_ROOT / 'validation/x.jsonl'))

    def test_loading_report_preserves_original_file(self):
        # The path adapter must remain testable without old experiment outputs.
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'report.json'
            old = str(WORKSPACE / 'yolo_run/validation/example.jsonl')
            raw = json.dumps({'runs': [{'output': old, 'command': ['binary', old]}]}).encode()
            file.write_bytes(raw)
            relocated = load_history(file)
            self.assertEqual(relocated['runs'][0]['output'], str(TEST_ROOT / 'validation/example.jsonl'))
            self.assertEqual(relocated['runs'][0]['command'], ['binary', old])
            self.assertEqual(file.read_bytes(), raw)


if __name__ == '__main__':
    unittest.main()

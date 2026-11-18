"""Accounting and timeline tests without opening any hardware device."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from benchmark_split_models import analyze, make_long_fixture, Monitor, native_build_commands, stats


class SplitBenchmarkTest(unittest.TestCase):
    def test_build_verification_supports_separate_production_and_test_trees(self):
        commands = native_build_commands(Path('/project/app/build/camera_live'),
                                         Path('/project/test/build/bin/live_npu_test'))
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][2], '/project/app/build')
        self.assertNotIn('live_npu_test', commands[0])
        self.assertEqual(commands[1][2], '/project/test/build')
        self.assertNotIn('camera_live', commands[1])
        self.assertEqual(len(native_build_commands(Path('/test/build/bin/camera_live'),
                                                   Path('/test/build/bin/live_npu_test'))), 1)

    def test_missing_samples_do_not_become_zero(self):
        self.assertIsNone(stats([None])['mean'])
        self.assertEqual(stats([None, 20, 40])['mean'], 30)

    def test_count_completed_results_in_fixed_wall_window(self):
        events = [{'type': 'camera_live', 'workers': 3, 'npu_mode': 'split'}]
        for seq in range(240):
            stamp = 10000 + seq * 1000 / 30
            events.append({'type': 'frame', 'sequence': seq, 'timestamp_ms': stamp,
                           'raw_timestamp_ms': seq * 1000 / 30, 'output_monotonic_ms': stamp + 5,
                           'output_latency_ms': 5, 'correction_timestamp_ms': None, 'correction_age_ms': None})
        for seq, worker in [(89, 0), (90, 1), (209, 2)]:
            events.append({'type': 'detection', 'sequence': seq, 'worker': worker,
                           'timestamp_ms': 10000 + seq * 1000 / 30,
                           'completed_ms': 10000 + seq * 1000 / 30 + 100,
                           'infer_ms': 90, 'preprocess_ms': 1, 'postprocess_ms': 1})
        events.append({'type': 'summary', 'status': 'completed', 'frame_count': 240,
                       'accepted_results': 3, 'inference_count': 4, 'late_results_dropped': 1,
                       'source': {'dequeued': 240}, 'result_queue_peak': 3, 'json_queue_peak': 2,
                       'decoded_pool_dropped': 0})
        result = analyze(events, [], 240, 30, 'short')
        self.assertTrue(all(result['checks'].values()))
        # seq89 completes inside the window; seq209 completes outside it.
        self.assertEqual(result['steady']['effective_detection_fps'], .5)
        self.assertEqual(result['steady']['source_detection_coverage'], 2 / 120)
        events[-1]['inference_count'] = 3
        self.assertFalse(analyze(events, [], 240, 30, 'short')['checks']['inference_accounting'])

    def test_long_fixture_has_no_time_or_sequence_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '0.jpg').write_bytes(b'fixture')
            original = root / 'manifest.jsonl'
            original.write_text(json.dumps({'type': 'mjpeg_fixture', 'width': 800, 'height': 600, 'negotiated_fps': 30}) + '\n' +
                                json.dumps({'file': '0.jpg', 'sequence': 0, 'timestamp_ms': 0}) + '\n')
            manifest, count = make_long_fixture(original, root / 'long', 600)
            data = [json.loads(line) for line in manifest.read_text().splitlines()][1:]
            self.assertEqual(count, 18001)
            self.assertGreaterEqual(data[-1]['timestamp_ms'], 600000)
            self.assertTrue(all(a['timestamp_ms'] < b['timestamp_ms'] and a['sequence'] < b['sequence'] for a, b in zip(data, data[1:])))
            self.assertEqual((manifest.parent / '0.jpg').resolve(), (root / '0.jpg').resolve())

    def test_dma_aliases_are_counted_once(self):
        monitor = Monitor.__new__(Monitor)
        monitor.proc = Path('/fakeproc')
        monitor.read = lambda path: ('pos:\t0\nino:\t42\nsize:\t1048576\nexp_name:\tsystem\n'
                                     if 'fdinfo' in str(path) else None)
        result = monitor.dma_buffers([Path('/fakeproc/fd/3'), Path('/fakeproc/fd/4')])
        self.assertEqual(result, {'dmabuf_mib': 1, 'dmabuf_objects': 1})


if __name__ == '__main__':
    unittest.main()

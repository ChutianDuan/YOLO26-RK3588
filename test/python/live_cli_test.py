#!/usr/bin/python3
"""Hardware pipeline tests; all delayed/error controls exist only in fault CLI."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

binary, fault, fixture_tool, source_dir = map(Path, sys.argv[1:])
env = dict(os.environ)
for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
    env.pop(key, None)

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

with tempfile.TemporaryDirectory(prefix='camera_live_cli_') as tmp:
    work = Path(tmp)
    fixture = work / 'fixture'
    video = '/home/yolo/test/rknn_video_compare_20261018_002d290d/video/002d290d-90f2bab2.mov'
    subprocess.run([str(fixture_tool), 'video', video, str(fixture), '90'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30, env=env)
    common = ['--preview', 'off', '--replay', str(fixture / 'manifest.jsonl')]

    def run(name, extra=(), exe=binary, ok=True):
        output = work / (name + '.jsonl')
        p = subprocess.run([str(exe), *common, '--output', str(output), *extra], env=env,
                           capture_output=True, text=True, timeout=20)
        assert (p.returncode == 0) == ok, (name, p.stderr)
        assert not p.stdout, p.stdout
        return output, p

    for name, extra, exe in [
        ('mpp', [], binary),
        ('unstabilized', ['--stabilize', 'off'], binary),
        ('fixed-crop', ['--stabilize-alpha', '1'], binary),
        ('explicit-mpp', ['--decoder', 'mpp', '--max-frames', '15'], binary),
        ('slow-infer', ['--test-infer-delay-ms', '150'], fault),
        ('slow-output', ['--test-output-delay-ms', '75'], fault),
        ('reorder', ['--npu-mode', 'split', '--test-infer-delay-ms', '150', '--test-reorder', 'on'], fault),
        ('rate', ['--detect-fps', '4'], binary),
    ]:
        output, _ = run(name, extra, exe)
        data = rows(output)
        assert data[0]['type'] == 'camera_live' and data[-1]['status'] == 'completed'
        summary = data[-1]
        enabled = name != 'unstabilized'
        assert data[0]['schema_version'] == 2
        assert data[0]['stabilization']['enabled'] == enabled
        assert data[0]['bbox_coordinates'] == ('stabilized_800x600' if enabled else 'source_800x600')
        assert summary['stabilization']['enabled'] == enabled
        if enabled:
            assert summary['stabilization']['frames'] == summary['decoded_count']
        else:
            assert summary['stabilization']['frames'] == 0

        frames = [r for r in data if r['type'] == 'frame']
        detections = [r for r in data if r['type'] == 'detection']
        assert frames and len(frames) == summary['frame_count']
        for row in frames + detections:
            st = row['stabilization']; x,y,w,h = st['roi']
            assert st['enabled'] == enabled
            if enabled:
                assert (w,h) == (720,540) and 0 <= x <= 80 and 0 <= y <= 60
                assert (x|y|w|h) & 1 == 0
                if st['mask_source_timestamp_ms'] is not None:
                    assert st['mask_source_timestamp_ms'] <= st['mask_target_timestamp_ms'] < row['timestamp_ms']
            else:
                assert (x,y,w,h) == (0,0,800,600)
        by_sequence = {row['sequence']:row['stabilization']['roi'] for row in frames}
        assert all(row['sequence'] not in by_sequence or row['stabilization']['roi'] == by_sequence[row['sequence']] for row in detections)

        assert summary['json_queue_peak'] <= 64 and summary['result_queue_peak'] <= 3
        assert all(a['sequence'] < b['sequence'] and a['timestamp_ms'] < b['timestamp_ms'] for a,b in zip(frames,frames[1:]))
        assert all(a['sequence'] < b['sequence'] for a,b in zip(detections,detections[1:]))
        assert all(r['correction_timestamp_ms'] is None or r['correction_timestamp_ms'] <= r['timestamp_ms'] for r in frames)
        assert all(t['last_match_age_ms'] <= 500 for r in frames for t in r['tracks'])
        if name == 'slow-infer':
            assert summary['inference_pending_dropped'] > 0 and summary['frame_count'] >= 85
        if name == 'slow-output':
            assert summary['output_pending_dropped'] > 0 and summary['decoded_count'] >= 85
        if name == 'reorder':
            assert summary['late_results_dropped'] > 0
        if name == 'rate':
            assert 7 <= summary['inference_count'] <= 14
    for name, extra in [('write-fail', ['--test-fail-write-after','3']),
                        ('write-full', ['--test-write-delay-ms','1000'])]:
        _, p = run(name, extra, fault, ok=False)
        assert 'write failure' in p.stderr or 'queue full' in p.stderr
    for name, extra in [('bad-stabilize',['--stabilize','bad']), ('bad-alpha',['--stabilize-alpha','0']),
                        ('nan-alpha',['--stabilize-alpha','nan']), ('large-alpha',['--stabilize-alpha','1.01']), ('removed-decoder',['--decoder','opencv']), ('bad-mode',['--npu-mode','invalid']), ('bad-time',['--max-predict-ms','nan']),
                        ('negative',['--detect-fps','-1']), ('repeat',['--preview','on']),
                        ('bad-frames',['--max-frames','1junk']), ('production-hook',['--test-infer-delay-ms','1'])]:
        output, _ = run(name, extra, ok=False)
        assert not output.exists()
    output, _ = run('overwrite', ['--max-frames','2'])
    original = output.read_bytes()
    p = subprocess.run([str(binary), *common, '--output', str(output)], capture_output=True, env=env, timeout=10)
    assert p.returncode and output.read_bytes() == original
    interrupted = work / 'interrupted.jsonl'
    with (work / 'signal.log').open('w') as log:
        p = subprocess.Popen([str(binary), *common, '--output', str(interrupted)], env=env,
                             stdout=subprocess.PIPE, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if interrupted.exists() and any(r['type'] == 'frame' for r in rows(interrupted)):
                    break
                assert p.poll() is None
                time.sleep(.02)
            else:
                raise AssertionError('no output before signal deadline')
            p.send_signal(signal.SIGINT)
            out, _ = p.communicate(timeout=5)
            assert p.returncode == 0 and out == b'' and rows(interrupted)[-1]['status'] == 'interrupted'
        finally:
            if p.poll() is None:
                p.kill(); p.wait()
print('PASS MPP-only, timestamp association, slow inference/output, out-of-order results, JSON failure/backpressure, CLI protection and SIGINT')

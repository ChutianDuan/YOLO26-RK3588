#!/usr/bin/python3
"""Real USB camera integration; fault injection affects only the test executable."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

binary, fault_binary, capture_test = map(Path, sys.argv[1:])
env = dict(os.environ)
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def run(exe, *args, ok=True):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True, env=env, timeout=25)
    assert (result.returncode == 0) == ok, (args, result.stderr)
    assert result.stdout == '', result.stdout
    return result

with tempfile.TemporaryDirectory(prefix='camera_cli_') as directory:
    work = Path(directory)
    subprocess.run([str(capture_test), '--capture'], check=True, env=env, capture_output=True, timeout=15)
    for mode, options in [('default', []), ('dmabuf', ['--io', 'dmabuf'])]:
        count = 10
        output = work / f'{mode}.jsonl'
        picture = work / f'{mode}.png'
        run(binary, *options, '--max-frames', count, '--output', output, '--snapshot', picture)
        rows = read(output)
        frames = [r for r in rows if r['type'] == 'frame']
        meta, summary = rows[0], rows[-1]
        assert len(frames) == count and summary['status'] == 'completed'
        assert meta['io_mode'] == 'dmabuf' and meta['buffers'] == 4 and meta['bytes_per_line'] == 1600
        assert summary['application_upload_bytes'] == 0
        assert 'validation_frames' not in summary and 'checksum_samples' not in summary
        assert all('rgb_checksum' not in r for r in rows)
        assert summary['snapshot_conversion_count'] == 1
        assert summary['resources']['first_fd'] == summary['resources']['last_fd']
        assert all(r['width'] == 800 and r['height'] == 600 for r in frames)
        assert all(a['sequence'] < b['sequence'] and a['timestamp_ms'] < b['timestamp_ms'] for a, b in zip(frames, frames[1:]))
        assert picture.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    for options in ([], ['--io', 'dmabuf']):
        run(fault_binary, *options, '--max-frames', 1, '--output', work / 'forced.jsonl', ok=False)
        assert not (work / 'forced.jsonl').exists()
    for args in [('--io', 'copy'), ('--io', 'auto'), ('--io', 'bad'), ('--max-frames', '-1'), ('--max-frames', '1junk'), ('--max-frames', '1', '--max-frames', '2'),
                 ('--snapshot', 'bad.avi'), ('--output', str(work / 'dmabuf.jsonl'))]:
        run(binary, *args, ok=False)
    interrupted = work / 'interrupted.jsonl'
    with (work / 'interrupt.stderr').open('w') as log:
        process = subprocess.Popen([str(binary), '--io', 'dmabuf', '--output', str(interrupted)], env=env,
                                   stdout=subprocess.PIPE, stderr=log, text=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if interrupted.exists() and any(r['type'] == 'frame' for r in read(interrupted)):
                    break
                assert process.poll() is None
                time.sleep(0.05)
            else:
                raise AssertionError('no frames before interrupt deadline')
            process.send_signal(signal.SIGINT)
            stdout, _ = process.communicate(timeout=6)
            assert process.returncode == 0 and stdout == ''
            assert read(interrupted)[-1]['status'] == 'interrupted'
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
print('PASS real camera DMA, unsupported-DMA strict failure, timestamps, counters, snapshot, CLI validation and SIGINT cleanup')

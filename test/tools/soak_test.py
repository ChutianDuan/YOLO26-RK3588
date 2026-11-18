#!/usr/bin/python3
"""Bounded-memory hardware run. Standard library only; no system changes."""
import argparse
import json
import pathlib
import os
import subprocess
import time

from paths import DEFAULT_BUILD, TEST_ROOT

env = dict(os.environ)
for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
    env.pop(key, None)
parser = argparse.ArgumentParser()
parser.add_argument('--build-dir', '--build', dest='build_dir', type=pathlib.Path, default=DEFAULT_BUILD)
parser.add_argument('--frames', type=int, default=3000)
parser.add_argument('--input', type=pathlib.Path, help='Reuse an existing local video with exactly --frames frames')
parser.add_argument('--config', type=pathlib.Path, help='Explicit deployment config, including LK backend/scale')
parser.add_argument('--output-dir', type=pathlib.Path, default=TEST_ROOT / 'results/soak')
args = parser.parse_args()
if args.frames < 300:
    parser.error('--frames must be at least 300')
args.output_dir.mkdir(parents=True, exist_ok=False)
build = args.build_dir.resolve()
work = args.output_dir.resolve()
input_video = args.input.resolve() if args.input else work / 'input.avi'
if args.input:
    if not input_video.is_file():
        parser.error('--input must be an existing local video')
else:
    subprocess.run([str(build / 'core_test'), '--fixture', str(input_video), str(args.frames)], check=True, env=env)
samples = []
fd_samples = []
temperatures = []
npu_samples = []
start = time.monotonic()
with (work / 'stderr.log').open('w') as stderr, (work / 'stdout.log').open('w') as stdout:
    command = [str(build / 'yolo_run'), '--input', str(input_video), '--output', str(work / 'results.jsonl')]
    if args.config:
        command += ['--config', str(args.config.resolve())]
    process = subprocess.Popen(command, stderr=stderr, stdout=stdout, env=env)
    try:
        while process.poll() is None:
            try:
                status = pathlib.Path(f'/proc/{process.pid}/status').read_text()
                rss = next(int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:'))
                samples.append({'seconds': round(time.monotonic() - start, 3), 'rss_mb': rss / 1024})
            except (FileNotFoundError, StopIteration):
                pass
            try:
                fd_samples.append({'seconds': round(time.monotonic() - start, 3),
                                   'count': len(list(pathlib.Path(f'/proc/{process.pid}/fd').iterdir()))})
            except OSError:
                pass
            temps = {}
            for zone in pathlib.Path('/sys/class/thermal').glob('thermal_zone*'):
                try:
                    temps[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000
                except (OSError, ValueError):
                    pass
            temperatures.append(temps)
            try:
                npu_samples.append(pathlib.Path('/sys/kernel/debug/rknpu/load').read_text().strip())
            except OSError:
                pass
            if time.monotonic() - start > 600:
                raise TimeoutError('soak exceeded 600 seconds')
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError((work / 'stderr.log').read_text())
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
count = 0
last_frame = -1
summary = None
with (work / 'results.jsonl').open() as results:
    for line in results:
        row = json.loads(line)
        if row['type'] == 'frame':
            assert row['frame_index'] == last_frame + 1
            last_frame = row['frame_index']
            count += 1
        else:
            summary = row
assert count == args.frames and summary['frame_count'] == count
assert summary['max_pending'] <= 1 and summary['history_peak'] <= 64 and summary['pending_at_end'] == 0
assert summary['async_submissions'] == summary['async_corrections'] + summary['discarded_results']
# Ignore initialization and compare the second and last quarter. This detects
# sustained growth on this run, and does not certify unlimited runtime stability.
steady = [s['rss_mb'] for s in samples if s['seconds'] >= max(2, (time.monotonic() - start) * .2)]
assert len(steady) >= 8, 'run too short for memory observation; increase --frames'
quarter = max(1, len(steady) // 4)
early = sum(steady[:quarter]) / quarter
late = sum(steady[-quarter:]) / quarter
growth = late - early
steady_fds = [s['count'] for s in fd_samples if s['seconds'] >= max(2, (time.monotonic() - start) * .2)]
assert len(steady_fds) >= 8, 'run too short for FD observation'
fd_quarter = max(1, len(steady_fds) // 4)
early_fds = max(steady_fds[:fd_quarter])
late_fds = max(steady_fds[-fd_quarter:])
report = {
    'frames': count, 'summary': summary, 'rss_samples': samples,
    'steady_early_mean_mb': early, 'steady_late_mean_mb': late, 'steady_growth_mb': growth,
    'memory_pass': growth < 16, 'temperature_max_c': {key: max(t.get(key, 0) for t in temperatures) for t in temperatures for key in t},
    'npu_load_samples': npu_samples,
    'fd_samples': fd_samples, 'steady_early_peak_fds': early_fds,
    'steady_late_peak_fds': late_fds, 'fd_pass': late_fds <= early_fds,
}
(work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
assert report['memory_pass'], f'RSS grew by {growth:.2f} MiB; see {work / "report.json"}'
assert report['fd_pass'], f'FD count grew from {early_fds} to {late_fds}; see {work / "report.json"}'
print(json.dumps({key: report[key] for key in ('frames', 'steady_early_mean_mb', 'steady_late_mean_mb', 'steady_growth_mb', 'memory_pass', 'steady_early_peak_fds', 'steady_late_peak_fds', 'fd_pass')}))
print(f'Report: {work / "report.json"}')

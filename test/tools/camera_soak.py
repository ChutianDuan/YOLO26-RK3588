#!/usr/bin/python3
"""Run a finite camera validation and retain logs, PNG, JSONL and checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

from paths import SOURCE, DEFAULT_BUILD

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
parser.add_argument('--frames', type=int, default=3000)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if args.frames < 30:
    parser.error('--frames must be at least 30 for stability sampling')
source = SOURCE
args.output_dir.mkdir(parents=True, exist_ok=False)
destination = args.output_dir.resolve()
env = dict(os.environ)
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
paths = [args.build_dir.resolve() / 'camera_run', source.parent / 'yolo_model_rknn/yolo26s_800x608_int8.rknn',
         Path('/opt/rknn/runtime-2.5.0/lib/librknnrt.so'), Path('/opt/rockchip/librga/lib/librga.so')]
environment = {'uname': os.uname(), 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(destination / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
before = subprocess.run(['dmesg', '--time-format', 'iso'], capture_output=True, text=True)
(destination / 'kernel-before.log').write_text(before.stdout)
with (destination / 'pixel-validation.log').open('w') as log:
    subprocess.run([str(args.build_dir.resolve() / 'camera_validation'), '--output', str(destination / 'pixel-validation.json')],
                   env=env, stdout=log, stderr=log, check=True, timeout=30)
pixel_validation = json.loads((destination / 'pixel-validation.json').read_text())
with (destination / 'stdout.log').open('w') as stdout, (destination / 'stderr.log').open('w') as stderr:
    result = subprocess.run([str(paths[0]), '--io', 'dmabuf', '--max-frames', str(args.frames),
                             '--output', str(destination / 'results.jsonl'), '--snapshot', str(destination / 'snapshot.png')],
                            env=env, stdout=stdout, stderr=stderr, timeout=max(120, args.frames / 3 + 60))
after = subprocess.run(['dmesg', '--time-format', 'iso'], capture_output=True, text=True)
old_lines = set(before.stdout.splitlines())
new_lines = [line for line in after.stdout.splitlines() if line not in old_lines]
(destination / 'kernel-new.log').write_text('\n'.join(new_lines) + ('\n' if new_lines else ''))
rows = [json.loads(line) for line in (destination / 'results.jsonl').read_text().splitlines()]
frames = [row for row in rows if row['type'] == 'frame']
summary = rows[-1]
resources = summary['resources']
png = (destination / 'snapshot.png').read_bytes()
checks = {
    'process_success': result.returncode == 0 and summary['status'] == 'completed',
    'frame_count': len(frames) == summary['frame_count'] == summary['inference_count'] == args.frames,
    'sequences_and_timestamps': all(a['sequence'] < b['sequence'] and a['timestamp_ms'] < b['timestamp_ms'] for a,b in zip(frames,frames[1:])),
    'source_and_model_sizes': all((f['width'],f['height'],f['model_width'],f['model_height']) == (800,600,800,608) for f in frames),
    'bbox_in_source': all(0 <= d['bbox'][0] < d['bbox'][2] <= 800 and 0 <= d['bbox'][1] < d['bbox'][3] <= 600 for f in frames for d in f['detections']),
    'copy_accounting': summary['application_upload_bytes'] == 0,
    'direct_pixel_validation': pixel_validation['pixels_within_tolerance'] and pixel_validation['frames'] == 8,
    'snapshot_800x608': png[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',png[16:24]) == (800,608),
    'single_snapshot': summary['snapshot_conversion_count'] == 1,
    'stable_fds': resources['first_fd'] == resources['last_fd'] == resources['min_fd'] == resources['max_fd'],
    'rss_growth_below_16mb': resources['last_rss_mb'] - resources['first_rss_mb'] < 16,
    'no_new_kernel_errors': not any(('rga' in line.lower() or 'rknpu' in line.lower() or 'uvc' in line.lower()) and
                                  any(word in line.lower() for word in ('error','timeout','fault','failed','reset')) for line in new_lines),
}
(destination / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'checks':checks,'summary':summary}, indent=2))
sys.exit(0 if all(checks.values()) else 1)

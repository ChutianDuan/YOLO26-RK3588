#!/usr/bin/python3
"""Serial camera DMA benchmark and soak. Requires exclusive camera access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=workspace/'yolo_run/build-rk3588/camera_live')
    parser.add_argument('--before', type=Path, help='Preserved pre-change camera_live binary')
    parser.add_argument('--config', type=Path, default=workspace/'yolo_run/config/camera_coco640.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--soak-frames', type=int, default=3000)
    parser.add_argument('--compare-stabilization', action='store_true', help='Compare --stabilize off/on using the same binary')
    args = parser.parse_args()
    if args.compare_stabilization and args.before:
        parser.error('use either --before or --compare-stabilization')
    if args.frames < 30 or args.soak_frames < 30 or args.repeats < 1:
        parser.error('frames/soak-frames must be >=30 and repeats >=1')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    baseline = args.binary if args.compare_stabilization else args.before
    report = {'comparison': 'stabilize_off_on' if args.compare_stabilization else 'binaries', 'config': str(args.config.resolve()), 'runs': [], 'binary_sha256': {}}
    for name, binary in [('after', args.binary), ('before', baseline)]:
        if binary:
            report['binary_sha256'][name] = hashlib.sha256(binary.read_bytes()).hexdigest()

    def save():
        (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')

    def run(name, binary, frames, direct):
        output = root/(name+'.jsonl')
        command = [str(binary.resolve()), '--config', str(args.config.resolve()), '--preview', 'off',
                   '--decoder', 'mpp', '--npu-mode', 'split', '--max-frames', str(frames), '--output', str(output)]
        if args.compare_stabilization:
            command += ['--stabilize', 'off' if name.startswith('before-') else 'on']
        with (root/(name+'.stderr.log')).open('x') as stderr, (root/(name+'.stdout.log')).open('x') as stdout:
            result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=frames/10+60)
        rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        summary = rows[-1] if rows and rows[-1]['type'] == 'summary' else {}
        images = [r for r in rows if r['type'] == 'frame']
        resources = summary.get('resources', {})
        # The first output can precede the first RKNN import on the other workers.
        # Compare steady-state samples after 30 frames, retaining startup in JSONL.
        steady = [r for r in rows if r['type'] == 'resource' and r['frame_count'] >= 30]
        checks = {
            'completed': result.returncode == 0 and summary.get('status') == 'completed',
            'frame_count': len(images) == summary.get('frame_count') == frames,
            'monotonic_frames': all(a['sequence'] < b['sequence'] and a['timestamp_ms'] < b['timestamp_ms']
                                    for a, b in zip(images, images[1:])),
            'stable_fds': bool(steady) and len({r['fd_count'] for r in steady}) == 1,
            'bounded_rss': bool(steady) and max(r['rss_mb'] for r in steady)-steady[0]['rss_mb'] < 16,
        }
        if args.compare_stabilization:
            checks['output_fps_29'] = summary.get('output_fps',0) >= 29
        if direct or args.compare_stabilization:
            checks['capture_dma'] = bool(rows) and rows[0].get('capture_io') == 'dmabuf' and rows[0].get('packet_pool') == 8
            checks['copy_zero'] = summary.get('source', {}).get('compressed_copy_bytes') == 0
            checks['pool_stable'] = summary.get('source', {}).get('packet_pool_dropped') == 0 and summary.get('decoded_pool_dropped') == 0
        report['runs'].append({'name': name, 'command': command, 'checks': checks, 'metadata': rows[0] if rows else {}, 'summary': summary})
        save()
        print(name, checks, flush=True)
        if not all(checks.values()):
            raise RuntimeError('camera validation failed; see '+str(root))

    before_kernel = subprocess.run(['dmesg'], capture_output=True, text=True)
    for i in range(args.repeats):
        order = [('before', baseline), ('after', args.binary)]
        for name, binary in (order if i % 2 == 0 else reversed(order)):
            if binary:
                run(f'{name}-{i+1}', binary, args.frames, name == 'after')
    run('after-soak', args.binary, args.soak_frames, True)
    after_kernel = subprocess.run(['dmesg'], capture_output=True, text=True)
    old = set(before_kernel.stdout.splitlines())
    new = [line for line in after_kernel.stdout.splitlines() if line not in old]
    (root/'kernel-new.log').write_text('\n'.join(new)+'\n')
    report['new_hardware_errors'] = [line for line in new if
        any(device in line.lower() for device in ('uvc', 'mpp', 'rga', 'rknpu')) and
        any(word in line.lower() for word in ('error', 'fault', 'timeout', 'failed'))]
    report['kernel_log_accessible'] = before_kernel.returncode == after_kernel.returncode == 0
    report['medians'] = {}
    for name in ('before', 'after'):
        runs = [r['summary'] for r in report['runs'] if r['name'].startswith(name+'-') and not r['name'].endswith('soak')]
        if runs:
            report['medians'][name] = {
                'cpu_percent': statistics.median(r['cpu_percent'] for r in runs),
                'inference_fps': statistics.median(r['inference_fps'] for r in runs),
                'output_fps': statistics.median(r['output_fps'] for r in runs),
                'output_latency_p95_ms': statistics.median(r['timing']['output_latency']['p95_ms'] for r in runs),
            }
    report['passed'] = not report['new_hardware_errors'] and all(all(r['checks'].values()) for r in report['runs'])
    save()
    if not report['passed']:
        raise RuntimeError('new hardware errors; see '+str(root))


if __name__ == '__main__':
    main()

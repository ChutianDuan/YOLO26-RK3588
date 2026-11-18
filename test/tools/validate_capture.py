#!/usr/bin/python3
"""Serial 150-frame YUYV DMA and 300-frame MJPEG DMA split regression."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from paths import DEFAULT_BUILD
from summarize_metrics import percentile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    binary = args.build_dir.resolve()
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    before = subprocess.run(['dmesg', '--time-format', 'iso'], capture_output=True, text=True)
    (work / 'kernel-before.log').write_text(before.stdout or before.stderr)
    report = {'date_shanghai': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              'kind': 'capture-short-regression', 'runs': []}
    cases = [('yuyv-dmabuf-150', 'camera_run', 150, ['--io', 'dmabuf']),
             ('mjpeg-split-300', 'camera_live', 300, ['--preview', 'off', '--decoder', 'mpp', '--npu-mode', 'split'])]
    for name, executable, count, options in cases:
        output = work / (name + '.jsonl')
        command = [str(binary / executable), *options, '--max-frames', str(count), '--output', str(output)]
        if executable == 'camera_run':
            command += ['--snapshot', str(work / (name + '.png'))]
        with (work / (name + '.stderr.log')).open('x') as stderr, (work / (name + '.stdout.log')).open('x') as stdout:
            process = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=120)
        rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        frames = [r for r in rows if r['type'] == 'frame']
        summary = rows[-1] if rows and rows[-1]['type'] == 'summary' else {}
        checks = {'completed': process.returncode == 0 and summary.get('status') == 'completed',
                  'frame_count': len(frames) == summary.get('frame_count') == count,
                  'monotonic_frames': all(a['sequence'] < b['sequence'] and a['timestamp_ms'] < b['timestamp_ms']
                                          for a, b in zip(frames, frames[1:])),
                  'stdout_empty': (work / (name + '.stdout.log')).stat().st_size == 0}
        if executable == 'camera_run' and summary:
            checks['copy_accounting'] = summary['application_upload_bytes'] == 0
            checks['no_runtime_image_diagnostics'] = not any(k in row for row in rows for k in
                ('validation_frames', 'validation_upload_bytes', 'validation_rgb_read_bytes', 'diagnostic_read_bytes',
                 'checksum_samples', 'checksum_changes', 'rgb_checksum'))
            checks['fd_stable'] = summary['resources']['first_fd'] == summary['resources']['last_fd']
            latencies = [f['latency_ms'] for f in frames]
            latency = {f'p{n}_ms': percentile(latencies, n / 100) for n in (50, 95, 99)}
        else:
            latency = None
            if summary:
                checks['compressed_copy_zero'] = summary['source']['compressed_copy_bytes'] == 0
                checks['bounded_queues'] = summary['result_queue_peak'] <= 3 and summary['json_queue_peak'] <= 64
                checks['detections_present'] = summary['inference_count'] > 0
                checks['no_pool_exhaustion'] = summary['decoded_pool_dropped'] == 0
                checks['causal_snapshots'] = all(f['correction_timestamp_ms'] is None or
                    f['correction_timestamp_ms'] <= f['timestamp_ms'] for f in frames)
        report['runs'].append({'name': name, 'command': command, 'summary': summary,
            'capture_to_result_percentiles': latency, 'checks': checks,
            'binary_sha256': hashlib.sha256((binary / executable).read_bytes()).hexdigest()})
        (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, checks, flush=True)
        if not all(checks.values()):
            raise RuntimeError('short regression failed; see ' + str(work))
    after = subprocess.run(['dmesg', '--time-format', 'iso'], capture_output=True, text=True)
    known = set(before.stdout.splitlines())
    added = [line for line in after.stdout.splitlines() if line not in known]
    (work / 'kernel-new.log').write_text('\n'.join(added) + ('\n' if added else ''))
    report['kernel_log_accessible'] = before.returncode == after.returncode == 0
    report['new_hardware_kernel_errors'] = [line for line in added if
        any(device in line.lower() for device in ('rknpu', 'rga', 'uvc', 'mpp')) and
        any(word in line.lower() for word in ('error', 'fault', 'failed', 'timeout', 'reset'))]
    report['passed'] = all(all(r['checks'].values()) for r in report['runs']) and not report['new_hardware_kernel_errors']
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

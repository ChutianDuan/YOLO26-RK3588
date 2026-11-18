#!/usr/bin/env python3
"""Real RK3588 integration: four distinct static sources, faults, EOF and signals."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

binary, reference, fault_binary, source_dir = map(Path, sys.argv[1:])
config = source_dir / 'config/video_multi.yaml'
bus = Path('/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg')
env = dict(os.environ)
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)


def command(args, expected=0, extra_env=None, timeout=90):
    result = subprocess.run(list(map(str, args)), env=extra_env or env, capture_output=True, text=True, timeout=timeout)
    detail = ''
    if '--output-dir' in args:
        path = Path(args[args.index('--output-dir')+1])/'summary.json'
        if path.exists():
            detail = path.read_text()
    assert result.returncode == expected, (args, result.returncode, result.stderr[-5000:], detail)
    return result


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def invoke(root, name, videos, *, record='off', fault=None, expected=0, extra=()):
    output = root / name
    args = [fault_binary if fault else binary, '--config', config, '--output-dir', output,
            '--preview', 'off', '--record', record, *extra]
    for video in videos:
        args += ['--input', video]
    run_env = dict(env)
    if fault:
        run_env['YOLO_MULTI_TEST_FAULT'] = fault
    command(args, expected=expected, extra_env=run_env)
    return output, json.loads((output / 'summary.json').read_text())


with tempfile.TemporaryDirectory(prefix='yolo_multi_test_') as temp:
    work = Path(temp)
    videos = []
    for i, (width, height, fps, seconds) in enumerate([(800,600,30,2), (640,480,15,3), (960,540,24,4), (320,240,10,5)]):
        path = work / f'channel{i}.avi'
        transform = f'scale={width}:{height}' + (',hflip' if i % 2 else '')
        transform += f',drawbox=x=0:y=0:w=48:h=48:color={("red", "green", "blue", "yellow")[i]}:t=fill'
        command(['/usr/bin/ffmpeg', '-v', 'error', '-nostdin', '-loop', '1', '-i', bus,
                 '-vf', transform, '-filter_threads', '1', '-r', fps, '-t', seconds,
                 '-c:v', 'mjpeg', '-q:v', '3', '-threads', '1', path])
        videos.append(path)
    fixed = command([reference, config, *videos])
    baseline = json.loads(fixed.stdout.strip().splitlines()[-1])
    assert len(baseline) == 4 and sum(len(b['detections']) for b in baseline) > 0
    assert len({json.dumps(b['detections']) for b in baseline}) > 1, 'fixtures must exercise result routing'
    output, summary = invoke(work, 'four', videos, record='on')
    assert summary['status'] == 'completed' and summary['inference_count'] > 0
    for i, stats in enumerate(summary['channels']):
        assert stats['accepted_results'] == stats['inference_count'] > 0
        assert stats['scheduler']['dispatched'] == stats['inference_count']
        assert stats['scheduler']['replaced'] + stats['inference_count'] == stats['decoded']
        assert not stats['scheduler']['busy'] and not stats['scheduler']['pending']
        assert stats['event_queue_peak'] <= 64 and stats['event_dropped'] == 0
        log = rows(output / f'stream_{i}/results.jsonl')
        detections = [row for row in log if row['type'] == 'detection']
        assert len(detections) == stats['inference_count']
        indices = [row['source_frame_index'] for row in detections]
        assert indices == sorted(set(indices))
        assert indices[-1] == stats['grabbed'] - 1, 'EOF did not drain final detection'
        for row in detections:
            assert row['channel_id'] == i and row['core_mask'] == 1 << row['worker']
            assert (row['width'], row['height']) == (baseline[i]['width'], baseline[i]['height'])
            expected = baseline[i]['detections']
            assert len(row['detections']) == len(expected), (i, row, expected)
            for actual, wanted in zip(row['detections'], expected):
                assert actual['class_id'] == wanted['class_id']
                assert abs(actual['score'] - wanted['score']) <= .01
                assert max(abs(a-b) for a,b in zip(actual['bbox'], wanted['bbox'])) <= 1
        frames = [row for row in log if row['type'] == 'frame']
        for row in frames:
            assert row['detection_source_frame_index'] <= row['source_frame_index']
            assert row['source_frame_index'] <= row['output_index']
            assert row['detection_age_ms'] is None or row['detection_age_ms'] >= 0
            assert all(t['track_uid'] == f'{i}:{t["track_id"]}' for t in row['tracks'])
            for track in row['tracks']:
                assert track['last_detection_source_frame_index'] <= row['detection_source_frame_index']
                assert abs(track['detection_age_ms']-(row['media_timestamp_ms']-track['last_detection_timestamp_ms'])) < .001
                assert 0 <= track['detection_age_ms'] <= 500.000001
                assert all(v >= 0 for v in track['center_stddev_px'])
        probe = command(['/usr/bin/ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                         'stream=width,height,nb_read_frames,r_frame_rate:format=duration', '-of', 'json',
                         output / f'stream_{i}/annotated.avi'])
        media = json.loads(probe.stdout)
        assert int(media['streams'][0]['nb_read_frames']) == stats['recorded_frames']
        assert stats['recorded_frames'] == stats['grabbed']
        assert abs(float(media['format']['duration']) - stats['grabbed']/stats['source_fps']) < .05
    # RKNN uses a lower source-time cadence; other source frames use native tracker prediction.
    prediction_out, prediction = invoke(work, 'prediction', videos, extra=['--detect-fps', '4'])
    assert prediction['tracking']=='bytetrack_native_kalman'
    for i, stats in enumerate(prediction['channels']):
        assert stats['detection_rate_skipped'] > 0
        assert stats['decoded'] == stats['detection_rate_skipped'] + stats['inference_count'] + stats['scheduler']['replaced']
        assert stats['tracking']['prediction_frames'] > stats['inference_count']
        frames = [r for r in rows(prediction_out/f'stream_{i}/results.jsonl') if r['type']=='frame']
        assert any(r['predicted_boxes'] and not r['detection_correction_applied'] for r in frames)
        assert any(r['detection_correction_applied'] for r in frames)
        assert all(r['tracks_source'] in ('kalman','empty') for r in frames)
        events = [r for r in rows(prediction_out/f'stream_{i}/results.jsonl') if r['type']=='detection']
        assert events[-1]['source_frame_index']==stats['grabbed']-1
    age_out, _ = invoke(work, 'prediction_age', videos[:1], extra=['--detect-fps','2','--track-max-age-ms','100'])
    age_frames = [r for r in rows(age_out/'stream_0/results.jsonl') if r['type']=='frame']
    assert any(r['tracks'] for r in age_frames) and any(not r['tracks'] for r in age_frames)
    assert all(t['detection_age_ms']<=100.000001 for r in age_frames for t in r['tracks'])
    command([binary, '--input', videos[0], '--output-dir', output, '--preview', 'off'], expected=1)
    for args in [[], ['--input', videos[0]] * 5 + ['--output-dir', work/'invalid'],
                 ['--input', videos[0], '--output-dir', work/'invalid', '--duration-seconds', 'nan'],
                 ['--input', work/'missing', '--output-dir', work/'invalid'],
                 ['--input', videos[0], '--output-dir', work/'invalid', '--record', 'yes'],
                 ['--input', videos[0], '--output-dir', work/'invalid', '--detect-fps', '-1'],
                 ['--input', videos[0], '--output-dir', work/'invalid', '--detect-fps', 'nan'],
                 ['--input', videos[0], '--output-dir', work/'invalid', '--track-max-age-ms', '1001']]:
        command([binary, *args], expected=1)
    headless = dict(env); headless.pop('DISPLAY',None); headless.pop('WAYLAND_DISPLAY',None)
    command([binary, '--input', videos[0], '--output-dir', work/'display', '--preview','on'], expected=1, extra_env=headless)
    _, limited = invoke(work, 'duration', videos, extra=['--duration-seconds', '.7'])
    assert limited['status'] == 'stopped' and limited['stop_reason'] == 'duration'
    _, writer = invoke(work, 'writer', videos[:2], fault='writer', expected=1)
    assert writer['channels'][0]['output_error'] and writer['channels'][1]['accepted_results'] > 0
    _, lag = invoke(work, 'lag', videos[:2], fault='record-lag', record='on', expected=1)
    assert lag['channels'][0]['record_error'] and lag['channels'][1]['recorded_frames'] > 0
    _, slow = invoke(work, 'slow', videos, fault='slow-infer')
    assert all(c['inference_count'] > 0 for c in slow['channels'])
    _, failed = invoke(work, 'infer_error', videos, fault='infer-error', expected=1)
    assert 'injected infer failure' in failed['fatal_error']
    _, decode = invoke(work, 'decode_error', videos[:2], fault='decode-error', expected=1)
    assert 'injected decode failure' in decode['channels'][0]['source_error']
    assert decode['channels'][1]['accepted_results'] > 0 and decode['channels'][1]['source_done']
    signal_out = work/'signal'
    args = [binary,'--output-dir',signal_out,'--preview','off','--record','off']
    for video in videos:
        args += ['--input',video]
    with (work/'signal.log').open('w') as logfile:
        process = subprocess.Popen(list(map(str,args)), env=env, stdout=logfile, stderr=logfile)
        deadline = time.monotonic()+30
        while not (signal_out/'metadata.json').exists() and process.poll() is None and time.monotonic()<deadline:
            time.sleep(.02)
        time.sleep(.35); process.send_signal(signal.SIGINT)
        assert process.wait(timeout=15) == 0
    interrupted = json.loads((signal_out/'summary.json').read_text())
    assert interrupted['stop_reason']=='signal' and all(not c['scheduler']['busy'] for c in interrupted['channels'])
    traffic = source_dir.parent/'test/rknn_video_compare_20261018_002d290d/video/002d290d-90f2bab2.mov'
    if traffic.exists():
        _, rotated = invoke(work,'rotated',[traffic],extra=['--duration-seconds','.6'])
        assert (rotated['channels'][0]['width'],rotated['channels'][0]['height']) == (1280,720)
    print('PASS: four distinct inputs equal serial inference, stable DMA allocations, independent tracking, mixed FPS/EOF, recording, CLI, faults, SIGINT and rotated MOV')

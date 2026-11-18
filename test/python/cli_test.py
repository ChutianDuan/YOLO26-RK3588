#!/usr/bin/python3
"""Hardware CLI integration, using only the system Python standard library."""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

binary, source, fixture_binary = map(pathlib.Path, sys.argv[1:])
bus = pathlib.Path('/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg')

def run(*args, ok=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True, timeout=45)
    assert (result.returncode == 0) == ok, (args, result.stderr)
    if ok:
        assert result.stdout == '', 'runtime diagnostics escaped to stdout'
    return result

with tempfile.TemporaryDirectory(prefix='yolo_cli_') as name:
    work = pathlib.Path(name)
    for precision in ('fp16', 'int8'):
        for method in ('rga',):
            model = source.parent / 'yolo_model_rknn' / f'yolo26s_800x608_{precision}.rknn'
            out, image = work / 'image.jsonl', work / 'image.jpg'
            run('--input', bus, '--model', model, '--preprocess', method, '--output', out, '--annotated', image)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            assert len(rows) == 2 and rows[0]['detections'] and rows[1]['frame_count'] == 1
            assert rows[1]['preprocess'] == method and image.read_bytes().startswith(b'\xff\xd8')
            subprocess.run([str(fixture_binary), '--verify-media', str(image), '1'], check=True, timeout=10)
    video = work / 'video.avi'
    subprocess.run([str(fixture_binary), '--fixture', str(video), '60'], check=True, timeout=15)
    config = work / 'config.yaml'
    text = (source / 'config.yaml').read_text()
    text = text.replace('../yolo_model_rknn/', str(source.parent / 'yolo_model_rknn') + '/')
    text = text.replace('deploy/classes.json', str(source / 'deploy/classes.json'))
    for asynchronous, scale in ((True, 1.0), (False, 1.0), (True, 0.5), (True, 0.75)):
        configured = text.replace('video_model_async: true', f'video_model_async: {str(asynchronous).lower()}')
        configured = re.sub(r'(?m)^lk_scale:.*$', f'lk_scale: {scale}', configured)
        config.write_text(configured)
        out = work / 'video.jsonl'
        run('--input', video, '--config', config, '--output', out, '--annotated', work / 'marked.avi')
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        summary = rows.pop()
        assert len(rows) == 60 and summary['frame_count'] == 60
        assert [r['frame_index'] for r in rows] == list(range(60))
        assert summary['pending_at_end'] == 0 and summary['max_pending'] <= 1 and summary['history_peak'] <= 64
        assert summary['async_submissions'] == summary['async_corrections'] + summary['discarded_results']
        assert summary['lk_preprocess'] == 'rga' and summary['lk_scale'] == scale
        assert summary['lk_width'] == round(rows[0]['width'] * scale)
        assert summary['lk_height'] == round(rows[0]['height'] * scale)
        subprocess.run([str(fixture_binary), '--verify-media', str(work / 'marked.avi'), '60'], check=True, timeout=10)
        if asynchronous:
            assert summary['async_submissions'] > 0
    run('--input', work / 'missing', ok=False)
    for removed in ('opencv', 'bad'):
        run('--input', bus, '--preprocess', removed, ok=False)
    run('--input', bus, '--output', bus, ok=False)
    assert bus.stat().st_size > 1000
    run('--input', bus, '--model', work / 'missing.rknn', ok=False)
print('PASS CLI image/model/backend matrix, video sync/async, JSONL, outputs, invalid arguments')

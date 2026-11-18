#!/usr/bin/python3
"""Compare the installed FP16/INT8 models through camera_live's split pipeline.

All decoding, RGA and inference use the existing native C++17 executable. Model
AP uses the same annotated JPEGs without dropping evaluation frames; latency
uses the full paced fixture at its original rate. No system tuning or installs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from paths import WORKSPACE, SOURCE, TEST_ROOT, DEFAULT_DATASET
from compare_rknn import score_detection_file
from validate_camera_live import quality, npu_average


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'p99_ms': None}
    def quantile(p):
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {'count': len(values), 'mean_ms': statistics.mean(values),
            'p50_ms': quantile(.5), 'p95_ms': quantile(.95), 'p99_ms': quantile(.99)}


def sample_frequencies():
    values = {}
    for path in Path('/sys/class/devfreq').glob('*npu*/cur_freq'):
        values[str(path)] = int(path.read_text())
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'):
        values[str(path)] = int(path.read_text())
    return values


def run_case(binary, fixture, model, config, work, name, env, source_count, fps, warmup):
    output = work / (name + '.jsonl')
    command = [str(binary), '--preview', 'off', '--stabilize', 'off', '--decoder', 'mpp', '--npu-mode', 'split',
               '--model', str(model), '--config', str(config), '--replay', str(fixture),
               '--output', str(output)]
    print('RUN', name, flush=True)
    save(work / (name + '.command.json'), command)
    samples = []
    started = time.monotonic()
    with (work / (name + '.stderr.log')).open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                samples.append({'elapsed_s': time.monotonic() - started, 'frequencies': sample_frequencies()})
                if time.monotonic() - started > 180:
                    raise TimeoutError(name)
                time.sleep(.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if process.returncode:
        raise RuntimeError(f'{name} exited {process.returncode}; inspect {name}.stderr.log')
    rows = read_rows(output)
    metadata, summary = rows[0], rows[-1]
    frames = [r for r in rows if r['type'] == 'frame']
    detections = [r for r in rows if r['type'] == 'detection']
    resources = [r for r in rows if r['type'] == 'resource']
    assert metadata['npu_mode'] == 'split' and metadata['workers'] == 3
    assert summary['type'] == 'summary' and summary['status'] == 'completed'
    assert all(a['sequence'] < b['sequence'] for a, b in zip(frames, frames[1:]))
    assert all(a['sequence'] < b['sequence'] for a, b in zip(detections, detections[1:]))
    assert summary['decoded_pool_dropped'] == 0
    assert summary['result_queue_peak'] <= 3 and summary['json_queue_peak'] <= 64
    assert all(r['correction_timestamp_ms'] is None or r['correction_timestamp_ms'] <= r['timestamp_ms'] for r in frames)
    worker_counts = Counter(r['worker'] for r in detections)
    assert set(worker_counts) == {0, 1, 2}, worker_counts
    record = {'name': name, 'output': str(output), 'command': command, 'summary': summary,
              'accepted_results_by_worker': dict(worker_counts), 'output_frames': len(frames),
              'npu_load_percent_sample_mean': npu_average(resources)}
    if warmup:
        # Original-rate fixture: its raw time starts at zero and preserves source cadence.
        origin = frames[0]['timestamp_ms'] - frames[0]['raw_timestamp_ms']
        first, last = warmup, source_count - 30
        low, high = origin + first * 1000 / fps, origin + last * 1000 / fps
        selected = [r for r in detections if first <= r['sequence'] < last]
        rendered = [r for r in frames if first <= r['sequence'] < last]
        duration = (high - low) / 1000
        steady_resources = [r for r in resources if first <= r['frame_count'] < last]
        record['steady'] = {
            'source_sequence_window': [first, last], 'window_seconds': duration,
            'detection_fps': sum(low <= r['completed_ms'] < high for r in detections) / duration,
            'output_fps': sum(low <= r['output_monotonic_ms'] < high for r in frames) / duration,
            'accepted_detection_source_coverage': len(selected) / (last - first),
            'infer': distribution([r['infer_ms'] for r in selected]),
            'source_to_detection': distribution([r['completed_ms'] - r['timestamp_ms'] for r in selected]),
            'output_latency': distribution([r['output_latency_ms'] for r in rendered]),
            'correction_age': distribution([r['correction_age_ms'] for r in rendered if r['correction_age_ms'] is not None]),
            'preprocess': distribution([r['preprocess_ms'] for r in selected]),
            'postprocess': distribution([r['postprocess_ms'] for r in selected]),
            'npu_load_percent_sample_mean': npu_average(steady_resources),
            'steady_fd_min': min(r['fd_count'] for r in steady_resources),
            'steady_fd_max': max(r['fd_count'] for r in steady_resources),
            'steady_rss_first_mib': steady_resources[0]['rss_mb'],
            'steady_rss_last_mib': steady_resources[-1]['rss_mb'],
        }
    save(work / (name + '.frequencies.json'), samples)
    save(work / (name + '.measurement.json'), record)
    print('DONE', name, 'infer_fps', round(record.get('steady', {}).get('detection_fps', summary['inference_fps']), 2),
          'infer_ms', round(summary['timing']['infer']['mean_ms'], 2), flush=True)
    return record


def make_accuracy_fixture(work, original, indices):
    directory = work / 'annotated_fixture'
    directory.mkdir()
    rows = read_rows(original)
    metadata, source = rows[0], {r['sequence']: r for r in rows[1:]}
    selected = sorted(indices)
    # Retain three unscored tail frames so the last annotated inference can finish.
    tail = [i for i in sorted(source) if i > selected[-1]][:3]
    if len(tail) != 3:
        raise ValueError('fixture needs three source frames after the final annotation')
    fixture = directory / 'manifest.jsonl'
    with fixture.open('x') as stream:
        stream.write(json.dumps({**metadata, 'negotiated_fps': 5.0, 'purpose': 'same-frame AP only; 200ms spacing plus 3 tail frames'}) + '\n')
        for position, index in enumerate(selected + tail):
            filename = str(index) + '.jpg'
            (directory / filename).symlink_to((original.parent / source[index]['file']).resolve())
            stream.write(json.dumps({'file': filename, 'sequence': index, 'timestamp_ms': position * 200.0}) + '\n')
    return fixture


def export_accuracy(raw, output, indices, manifest):
    found = {}
    for row in read_rows(raw):
        if row['type'] == 'detection' and row['sequence'] in indices:
            index = row['sequence']
            if index in found:
                raise ValueError('duplicate annotation inference')
            boxes = []
            for item in row['detections']:
                b = item['bbox']
                boxes.append({**item, 'bbox': [b[0] * 1280 / 800, b[1] * 720 / 600,
                                              b[2] * 1280 / 800, b[3] * 720 / 600]})
            found[index] = {'type': 'frame', 'frame_index': index, 'width': 1280, 'height': 720,
                            'is_detection_frame': True, 'detections': boxes}
    if set(found) != indices:
        raise ValueError(f'AP requires every annotation: missing {sorted(indices - set(found))}')
    with output.open('x') as stream:
        for index in sorted(found):
            stream.write(json.dumps(found[index]) + '\n')
    return len(found)


def aggregate(runs):
    def average(path):
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(value)
        return statistics.mean(values)
    return {'repeats': len(runs),
            **{key: average(['steady', key]) for key in ('detection_fps', 'output_fps', 'accepted_detection_source_coverage')},
            **{key: {metric: average(['steady', key, metric]) for metric in ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms')}
               for key in ('infer', 'source_to_detection', 'output_latency', 'correction_age', 'preprocess', 'postprocess')},
            'peak_rss_mib': average(['summary', 'resources', 'max_rss_mb']),
            'whole_run_cpu_percent': average(['summary', 'cpu_percent']),
            'max_temperature_c': max(r['summary']['resources']['max_temperature_c'] for r in runs),
            'tracking': {key: average(['tracking', key]) for key in ('idf1', 'mota', 'id_switches', 'precision', 'recall', 'f1')
                         if key in runs[0]['tracking']}}


def render(work, report):
    a, b = report['aggregates']['fp16'], report['aggregates']['int8']
    qa, qb = report['detection_quality']['fp16'], report['detection_quality']['int8']
    lines = ['# split 三 Worker：FP16 / INT8 当前板端对比', '', '时间：' + report['environment']['date_shanghai'], '',
             '单路视频、3 个固定 Worker / 独立 Context，分别绑定 NPU Core0/1/2；MPP → NV12 DMA-BUF → RGA → RKNN，统一 ByteTrack/Kalman，关闭预览但执行画框。', '',
             '## 相同图像的模型检测精度', '',
             '同一 BDD100K 训练序列的 202 个标注时刻；使用现有 800×600 JPEG 夹具，MPP 解码后 letterbox 到 800×608。候选 conf≥0.001、最多 300 框；框还原到 1280×720，沿用原评测器的 crowd/distractor 忽略规则和 AP101。AP 仅平均有普通真值的六类，工作点 conf=0.25、IoU=0.5。',
             '为排除速度与丢帧影响，AP 专项以 200ms 间隔提交相同标注图，确认 FP16/INT8 都完成 202/202 次推理；其 FPS 不作性能比较。', '',
             '| 指标 | FP16 | INT8 |', '|---|---:|---:|']
    for label, path in [('mAP50', ['map50']), ('mAP50–95', ['map50_95']),
                        ('Precision @0.25', ['operating_point', 'precision']),
                        ('Recall @0.25', ['operating_point', 'recall']),
                        ('F1 @0.25', ['operating_point', 'f1']),
                        ('人体 AP50', ['per_class', 'person', 'ap50'])]:
        x, y = qa, qb
        for key in path:
            x, y = x[key], y[key]
        lines.append(f'| {label} | {x:.2%} | {y:.2%} |')
    lines += ['', '## 原始约 30 FPS 输入的延迟和吞吐', '',
              'FP16/INT8 各三遍，每轮交换先后顺序，实际为 FP16→INT8、INT8→FP16、FP16→INT8；所有硬件进程串行。各遍完整回放 1210 帧；默认 conf=0.25。稳态排除源序号前 90 帧和末尾 30 帧，按固定源时间窗内被接受的检测结果/完成输出数计算 FPS；表内均值与分位数均为各遍统计量的算术平均，不是合并样本分位数。', '',
              '| 指标 | FP16 split | INT8 split |', '|---|---:|---:|']
    rows = [('推理阶段均值 ms', a['infer']['mean_ms'], b['infer']['mean_ms']),
            ('推理阶段 P95 ms', a['infer']['p95_ms'], b['infer']['p95_ms']),
            ('源时间到检测结果均值 ms', a['source_to_detection']['mean_ms'], b['source_to_detection']['mean_ms']),
            ('源时间到检测结果 P95 ms', a['source_to_detection']['p95_ms'], b['source_to_detection']['p95_ms']),
            ('输出所用检测数据年龄 P95 ms', a['correction_age']['p95_ms'], b['correction_age']['p95_ms']),
            ('源时间到无屏画框完成 P95 ms', a['output_latency']['p95_ms'], b['output_latency']['p95_ms']),
            ('有效检测结果更新 FPS', a['detection_fps'], b['detection_fps']),
            ('视频输出 FPS', a['output_fps'], b['output_fps']),
            ('源帧检测覆盖率 %', a['accepted_detection_source_coverage'] * 100, b['accepted_detection_source_coverage'] * 100),
            ('每遍峰值 RSS 的均值 MiB', a['peak_rss_mib'], b['peak_rss_mib']),
            ('整段进程 CPU %（单核=100%）', a['whole_run_cpu_percent'], b['whole_run_cpu_percent']),
            ('采样最高温度 °C', a['max_temperature_c'], b['max_temperature_c'])]
    lines += [f'| {label} | {x:.2f} | {y:.2f} |' for label, x, y in rows]
    lines += ['', '`infer_ms` 包含输入绑定、Runtime 执行和输出获取/转换，不是纯 NPU 硬件时间；另有独立后处理计时。有效检测更新 FPS 只统计通过时间水位检查的结果，迟到结果不用于更新轨迹。输出线程不等待检测，所以输出 FPS 和无屏输出延迟不能代表检测更新速度。检测数据年龄才体现当前显示轨迹使用了多久以前的观测。INT8 受约 30 FPS 输入限制，本实验没有测出其最大检测吞吐。', '',
              '## 实时轨迹效果', '',
              '对各遍原始速度输出的 tracks 评分；缺失标注时刻按空预测计，预测框恢复到原图坐标，IoU=0.5。以下也是各遍均值，调度与检测覆盖率差异属于整条实时链路效果，不能全部归因于模型量化。', '',
              '| 指标 | FP16 split | INT8 split |', '|---|---:|---:|']
    for label, key in [('IDF1', 'idf1'), ('MOTA', 'mota'), ('ID 切换次数', 'id_switches')]:
        x, y = a['tracking'][key], b['tracking'][key]
        lines.append(f'| {label} | {x:.4f} | {y:.4f} |')
    lines += ['', '## 逐遍性能', '', '| 运行 | 检测 FPS | 推理均值 ms | 检测结果 P95 ms | 数据年龄 P95 ms | 输出帧数 |', '|---|---:|---:|---:|---:|---:|']
    for run in report['performance_runs']:
        s = run['steady']
        lines.append(f'| {run["name"]} | {s["detection_fps"]:.2f} | {s["infer"]["mean_ms"]:.2f} | {s["source_to_detection"]["p95_ms"]:.2f} | {s["correction_age"]["p95_ms"]:.2f} | {run["output_frames"]} |')
    lines += ['', '## 复现与边界', '',
              '```bash', 'cd /home/yolo', '/usr/bin/python3 test/tools/compare_split_precision.py \\',
              '  --output-dir test/results/split-precision-NEW --repeats 3', '```', '',
              '输出目录必须全新。命令、模型/视频/GT/夹具/二进制哈希、逐帧结果、逐遍计量、频率采样和评测 JSON 均保存在本目录。评测仅复用原 AP101 的精度包络及有界 IoU 缓存加速，运行时自检 24 组确定性等价用例，未修改材料评测器。',
              '保持系统现有频率策略，未锁频、修改设备权限/驱动/内核或部署服务。真实设备参与 MPP/RGA/NPU 运算；输入为可重复的录像 JPEG 回放，并非本次真实摄像头采集。实际屏幕、曝光到显示、多路独立摄像头和长期热稳定性未测。',
              '这是一条定向选择的 BDD100K 训练序列，不是独立工业测试集。JPEG 拉伸及压缩条件与历史原视频检测精度不同，不能把两组 AP 直接当成同一次实验。当前 INT8 split 默认配置保持不变。', '']
    (work / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--binary', type=Path, default=SOURCE / 'build-rk3588/camera_live')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--fixture', type=Path, default=TEST_ROOT / 'validation/camera_live_20261018/traffic_fixture/manifest.jsonl')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / 'executed_compare_split_precision.py').write_bytes(Path(__file__).read_bytes())
    (work / 'performance_config.snapshot.yaml').write_bytes((SOURCE / 'config.yaml').read_bytes())
    binary, pack, fixture = args.binary.resolve(), args.dataset.resolve(), args.fixture.resolve()
    manifest = json.loads((pack / 'manifest.json').read_text())
    fixture_rows = read_rows(fixture)
    source_count = manifest['video']['decoded_frame_count']
    if len(fixture_rows) - 1 != source_count:
        raise ValueError('full fixture frame count differs from dataset')
    if sha(pack / manifest['video']['path']) != manifest['video']['sha256']:
        raise ValueError('source video hash mismatch')
    if sha(pack / manifest['ground_truth']['path']) != manifest['ground_truth']['sha256']:
        raise ValueError('GT hash mismatch')
    models = {p: WORKSPACE / 'yolo_model_rknn' / f'yolo26s_800x608_{p}.rknn' for p in ('fp16', 'int8')}
    for precision, path in models.items():
        if sha(path) != manifest['rknn'][precision]['sha256']:
            raise ValueError('model hash mismatch: ' + precision)
    fixture_hashes = {row['file']: sha(fixture.parent / row['file']) for row in fixture_rows[1:]}
    save(work / 'fixture-jpeg-sha256.json', fixture_hashes)
    environment = {'date_shanghai': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   'platform': platform.platform(), 'binary_sha256': sha(binary),
                   'model_sha256': {p: sha(m) for p, m in models.items()},
                   'fixture_manifest_sha256': sha(fixture), 'dataset_manifest_sha256': sha(pack / 'manifest.json'),
                   'video_sha256': manifest['video']['sha256'], 'gt_sha256': manifest['ground_truth']['sha256'],
                   'script_sha256': sha(Path(__file__)),
                   'runtime_library_sha256': {str(p): sha(p) for p in [Path('/opt/rknn/runtime-2.5.0/lib/librknnrt.so'), Path('/opt/rockchip/librga/lib/librga.so'), Path('/opt/rockchip/mpp/lib/librockchip_mpp.so')]}}
    save(work / 'environment.json', environment)
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    dependencies = subprocess.run(['/usr/bin/ldd', str(binary)], env=env, capture_output=True, text=True, check=True).stdout
    if any(s in dependencies.lower() for s in ('conda', 'miniforge', 'not found')):
        raise ValueError('unexpected runtime dependencies')
    (work / 'ldd.log').write_text(dependencies)
    default_config = (SOURCE / 'config.yaml').read_text()
    config = work / 'accuracy.yaml'
    config.write_text(default_config.replace('labels_path: deploy/classes.json', 'labels_path: ' + json.dumps(str(SOURCE / 'deploy/classes.json')))
                      .replace('model_path: ../yolo_model_rknn/yolo26s_800x608_fp16.rknn', 'model_path: ' + json.dumps(str(models['fp16'])))
                      .replace('conf_threshold: 0.25', 'conf_threshold: 0.001'))
    runs = []
    for repeat in range(args.repeats):
        for precision in (('fp16', 'int8') if repeat % 2 == 0 else ('int8', 'fp16')):
            record = run_case(binary, fixture, models[precision], SOURCE / 'config.yaml', work,
                              f'performance_{precision}_{repeat + 1}', env, source_count, manifest['video']['fps'], 90)
            record['precision'] = precision
            runs.append(record)
            save(work / 'performance-checkpoint.json', runs)
    indices = {row['frame_index'] for row in json.loads((pack / 'frame_map.json').read_text())}
    annotated_fixture = make_accuracy_fixture(work, fixture, indices)
    accuracy_runs, predictions = {}, {}
    for precision in ('fp16', 'int8'):
        accuracy_runs[precision] = run_case(binary, annotated_fixture, models[precision], config, work,
                                             'accuracy_' + precision, env, source_count, 5.0, 0)
        predictions[precision] = work / ('accuracy_' + precision + '.predictions.jsonl')
        count = export_accuracy(Path(accuracy_runs[precision]['output']), predictions[precision], indices, manifest)
        print('AP coverage', precision, count, '/', len(indices), flush=True)
    # Finish all device measurement before CPU-heavy scoring to avoid interference.
    detection_quality = {}
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(score_detection_file, str(pack), str(path)): p for p, path in predictions.items()}
        for future in as_completed(futures):
            precision = futures[future]
            detection_quality[precision], metadata = future.result()
            save(work / ('quality-model-' + precision + '.json'), {'metrics': detection_quality[precision], 'input': metadata})
            print('SCORED model', precision, 'mAP50', round(detection_quality[precision]['map50'] * 100, 2), flush=True)
    for run in runs:
        print('SCORE tracks', run['name'], flush=True)
        run['tracking'] = quality(Path(run['output']), pack)
        save(work / (run['name'] + '.tracking.json'), run['tracking'])
    report = {'environment': environment, 'performance_runs': runs, 'accuracy_runs': accuracy_runs,
              'detection_quality': detection_quality,
              'aggregates': {p: aggregate([r for r in runs if r['precision'] == p]) for p in models}}
    save(work / 'report.json', report)
    render(work, report)
    print('REPORT', work / 'report.md', flush=True)


if __name__ == '__main__':
    main()

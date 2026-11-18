#!/usr/bin/python3
"""Serial RK3588 split benchmarks; existing native SDKs, read-only monitoring.

The paced source measures deployed video behavior, not maximum NPU throughput.
The parent samples only the child process; parsing/reporting happens between runs.
"""
import argparse
from collections import Counter
import csv
import datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

from compare_split_precision import distribution, sha
from paths import WORKSPACE, SOURCE, TEST_ROOT, DEFAULT_BUILD

MODELS = {
    'fp16': WORKSPACE / 'yolo_model_rknn/yolo26s_800x608_fp16.rknn',
    'int8': WORKSPACE / 'yolo_model_rknn/yolo26s_800x608_int8.rknn',
    'hybrid': WORKSPACE / 'yolo_model_rknn/rk_kl_divergence_hybrid_13.rknn',
}
SHANGHAI = datetime.timezone(datetime.timedelta(hours=8))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def native_build_commands(binary, native_test):
    groups = {}
    for executable, target in [(binary, 'camera_live'), (native_test, 'live_npu_test')]:
        root = executable.parent.parent if executable.parent.name == 'bin' else executable.parent
        groups.setdefault(root, []).append(target)
    return [['/usr/bin/cmake', '--build', str(root), '--target', *targets, '-j', '4']
            for root, targets in groups.items()]


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    result = distribution(values)
    return {**{k.removesuffix('_ms'): v for k, v in result.items()},
            'min': min(values) if values else None, 'max': max(values) if values else None}


def numeric_fields(text):
    result = {}
    for line in text.splitlines():
        key, sep, value = line.partition(':')
        if sep:
            try:
                result[key] = float(value.split()[0])
            except (ValueError, IndexError):
                pass
    return result


def make_long_fixture(original, destination, seconds):
    source = rows(original)
    metadata, frames = source[0], source[1:]
    fps = metadata['negotiated_fps']
    destination.mkdir()
    for item in frames:
        name = item['file']
        if Path(name).name != name:
            raise ValueError('fixture file must be a basename')
        (destination / name).symlink_to((original.parent / name).resolve())
    count = math.ceil(seconds * fps) + 1
    manifest = destination / 'manifest.jsonl'
    with manifest.open('x') as stream:
        stream.write(json.dumps({**metadata, 'purpose': 'single-process continuous split soak',
                                 'requested_seconds': seconds}) + '\n')
        for index in range(count):
            stream.write(json.dumps({'file': frames[index % len(frames)]['file'],
                                     'sequence': index, 'timestamp_ms': index * 1000 / fps}) + '\n')
    return manifest, count


class Monitor:
    def __init__(self, pid):
        self.proc = Path('/proc') / str(pid)
        self.previous_cpu = None
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.errors = Counter()
        self.frequencies = {**{f'cpu_{p.parent.name}_khz': p for p in
                              Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')},
                            **{'npu_hz': p for p in Path('/sys/class/devfreq').glob('*npu*/cur_freq')}}
        self.temperatures = {}
        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
            kind = self.read(zone / 'type')
            self.temperatures[f'temp_{kind.strip() if kind else zone.name}_c'] = zone / 'temp'

    def read(self, path):
        try:
            return path.read_text()
        except (OSError, ValueError) as error:
            self.errors[f'{path}: {type(error).__name__}'] += 1
            return None

    def number(self, path, divisor=1):
        value = self.read(path)
        try:
            return float(value.strip()) / divisor if value is not None else None
        except ValueError:
            self.errors[f'{path}: invalid number'] += 1
            return None

    def dma_buffers(self, fd_paths):
        # fdinfo may expose exporter/size directly. bufinfo supplies sizes by inode
        # on kernels without these fields. Deduplicate aliases of the same object.
        objects = {}
        inodes = set()
        visible = False
        for fd in fd_paths:
            info = self.read(self.proc / 'fdinfo' / fd.name)
            if info is None:
                continue
            fields = numeric_fields(info)
            inode = fields.get('ino')
            if inode is not None:
                inodes.add(int(inode))
            if 'exp_name:' in info and 'size' in fields and inode is not None:
                objects[int(inode)] = int(fields['size'])
                visible = True
        raw = self.read(Path('/sys/kernel/debug/dma_buf/bufinfo'))
        if raw is not None:
            visible = True
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) >= 6 and parts[0].isdigit() and parts[5].isdigit():
                    inode = int(parts[5])
                    if inode in inodes:
                        objects[inode] = int(parts[0])
        return {'dmabuf_mib': sum(objects.values()) / 1048576 if visible else None,
                'dmabuf_objects': len(objects) if visible else None}

    def sample(self, slow):
        now = time.monotonic()
        data = {'monotonic_s': now, 'cpu_percent': None}
        status = self.read(self.proc / 'status')
        fields = numeric_fields(status or '')
        for name, key in [('rss_mib', 'VmRSS'), ('hwm_mib', 'VmHWM'), ('swap_status_mib', 'VmSwap')]:
            data[name] = fields[key] / 1024 if key in fields else None
        raw = self.read(self.proc / 'stat')
        if raw:
            fields_stat = raw.rsplit(')', 1)[1].split()
            cpu = (int(fields_stat[11]) + int(fields_stat[12])) / self.ticks
            if self.previous_cpu and now > self.previous_cpu[0]:
                data['cpu_percent'] = (cpu - self.previous_cpu[1]) * 100 / (now - self.previous_cpu[0])
            self.previous_cpu = now, cpu
        try:
            fds = list((self.proc / 'fd').iterdir())
            data['fd_count'] = len(fds)
        except OSError:
            fds = []
            data['fd_count'] = None
        load = self.read(Path('/sys/kernel/debug/rknpu/load'))
        cores = dict(re.findall(r'Core([012]):\s*(\d+)%', load or ''))
        for core in range(3):
            data[f'npu_core{core}_percent'] = int(cores[str(core)]) if str(core) in cores else None
        for key, path in self.frequencies.items():
            data[key] = self.number(path)
        for key, path in self.temperatures.items():
            data[key] = self.number(path, 1000)
        if slow:
            smaps = numeric_fields(self.read(self.proc / 'smaps_rollup') or '')
            data['pss_mib'] = smaps['Pss'] / 1024 if 'Pss' in smaps else None
            data['private_mib'] = (smaps.get('Private_Clean', 0) + smaps.get('Private_Dirty', 0)) / 1024 if smaps else None
            data['swap_mib'] = smaps['Swap'] / 1024 if 'Swap' in smaps else None
            data.update(self.dma_buffers(fds))
        return data


def resource_stats(samples):
    keys = sorted({key for row in samples for key in row} - {'monotonic_s', 'elapsed_s'})
    return {key: stats([r.get(key) for r in samples]) for key in keys}


def fit_slope(samples, key):
    points = [(r['monotonic_s'], r[key]) for r in samples if r.get(key) is not None]
    if len(points) < 2:
        return None
    mean_x, mean_y = statistics.mean(p[0] for p in points), statistics.mean(p[1] for p in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator * 60 if denominator else None


def analyze(events, samples, source_count, fps, kind):
    metadata = events[0]
    summary = events[-1] if events[-1].get('type') == 'summary' else {}
    frames = [r for r in events if r['type'] == 'frame']
    detections = [r for r in events if r['type'] == 'detection']
    worker_counts = Counter(r['worker'] for r in detections)
    checks = {
        'completed': summary.get('status') == 'completed',
        'split_three_workers': metadata.get('npu_mode') == 'split' and metadata.get('workers') == 3,
        'all_workers_participated': set(worker_counts) == {0, 1, 2},
        'frame_count_matches': bool(summary) and summary.get('frame_count') == len(frames),
        'accepted_count_matches': bool(summary) and summary.get('accepted_results') == len(detections),
        'inference_accounting': bool(summary) and summary.get('inference_count') ==
            summary.get('accepted_results', -1) + summary.get('late_results_dropped', -1),
        'source_count_matches': summary.get('source', {}).get('dequeued') == source_count,
        'frame_sequence_monotonic': all(a['sequence'] < b['sequence'] for a, b in zip(frames, frames[1:])),
        'detection_sequence_monotonic': all(a['sequence'] < b['sequence'] for a, b in zip(detections, detections[1:])),
        'bounded_queues': bool(summary) and summary.get('result_queue_peak', 4) <= 3 and summary.get('json_queue_peak', 65) <= 64,
        'no_decoded_pool_exhaustion': summary.get('decoded_pool_dropped') == 0,
        'no_future_corrections': all(r['correction_timestamp_ms'] is None or
                                   r['correction_timestamp_ms'] <= r['timestamp_ms'] for r in frames),
    }
    result = {'metadata': metadata, 'summary': summary, 'checks': checks,
              'accepted_results_by_worker': dict(worker_counts), 'resources_whole_process': resource_stats(samples)}
    if not frames:
        return result
    origin = frames[0]['timestamp_ms'] - frames[0]['raw_timestamp_ms']
    start_sequence = 90 if kind == 'short' else math.ceil(30 * fps)
    end_sequence = source_count - 30
    low, high = origin + start_sequence * 1000 / fps, origin + end_sequence * 1000 / fps
    duration = (high - low) / 1000
    selected = [r for r in detections if start_sequence <= r['sequence'] < end_sequence]
    rendered = [r for r in frames if start_sequence <= r['sequence'] < end_sequence]
    steady_samples = [r for r in samples if low <= r['monotonic_s'] * 1000 < high]
    steady = {'source_sequence_window': [start_sequence, end_sequence], 'window_seconds': duration,
              'effective_detection_fps': sum(low <= r['completed_ms'] < high for r in detections) / duration,
              'output_fps': sum(low <= r['output_monotonic_ms'] < high for r in frames) / duration,
              'source_detection_coverage': len(selected) / (end_sequence - start_sequence),
              'resources': resource_stats(steady_samples)}
    for key in ['preprocess_ms', 'infer_ms', 'postprocess_ms']:
        steady[key] = stats([r[key] for r in selected])
    steady['source_to_detection_ms'] = stats([r['completed_ms'] - r['timestamp_ms'] for r in selected])
    steady['output_latency_ms'] = stats([r['output_latency_ms'] for r in rendered])
    steady['correction_age_ms'] = stats([r['correction_age_ms'] for r in rendered])
    result['steady'] = steady
    if kind == 'soak':
        result['source_span_seconds'] = (source_count - 1) / fps
        first = [r for r in steady_samples if r['monotonic_s'] * 1000 < low + 60000]
        last = [r for r in steady_samples if r['monotonic_s'] * 1000 >= high - 60000]
        growth = {}
        for key in ['rss_mib', 'pss_mib', 'private_mib', 'dmabuf_mib', 'fd_count']:
            a = [r[key] for r in first if r.get(key) is not None]
            b = [r[key] for r in last if r.get(key) is not None]
            growth[key] = {'first_minute_median': statistics.median(a) if a else None,
                           'last_minute_median': statistics.median(b) if b else None,
                           'delta': statistics.median(b) - statistics.median(a) if a and b else None,
                           'slope_per_minute': fit_slope(steady_samples, key)}
        result['memory_growth'] = growth
        minutes = []
        for minute in range(math.ceil(result['source_span_seconds'] / 60)):
            a, b = origin + minute * 60000, min(origin + (minute + 1) * 60000, origin + (source_count - 1) * 1000 / fps)
            if b - a < 1000:
                continue
            resources = [r for r in samples if a <= r['monotonic_s'] * 1000 < b]
            completed = [r for r in detections if a <= r['completed_ms'] < b]
            minutes.append({'minute': minute + 1, 'duration_s': (b - a) / 1000,
                            'effective_detection_fps': len(completed) * 1000 / (b - a),
                            'resources': resource_stats(resources)})
        result['minutes'] = minutes
    return result


def run_case(binary, config, fixture, model, work, name, env, count, fps, kind, requested_seconds):
    output = work / f'{name}.jsonl'
    command = [str(binary), '--config', str(config), '--model', str(model), '--replay', str(fixture),
               '--preview', 'off', '--stabilize', 'off', '--decoder', 'mpp', '--npu-mode', 'split', '--detect-fps', '0', '--output', str(output)]
    save(work / f'{name}.command.json', command)
    print('RUN', name, datetime.datetime.now(SHANGHAI).isoformat(), flush=True)
    started = time.monotonic()
    samples, error = [], None
    with (work / f'{name}.stderr.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        monitor = Monitor(process.pid)
        next_slow, next_progress = started, started + 30
        deadline = started + max(180, requested_seconds + 120)
        try:
            while process.poll() is None:
                now = time.monotonic()
                sample = monitor.sample(now >= next_slow)
                sample['elapsed_s'] = sample['monotonic_s'] - started
                samples.append(sample)
                if now >= next_slow:
                    next_slow = now + 1
                if now >= next_progress:
                    print('PROGRESS', name, round(now - started), 's', 'rss_mib', sample['rss_mib'],
                          'npu', [sample[f'npu_core{i}_percent'] for i in range(3)], flush=True)
                    next_progress = now + 30
                if now > deadline:
                    raise TimeoutError(f'run exceeded {deadline - started:.1f}s')
                time.sleep(max(0, .25 - (time.monotonic() - now)))
        except Exception as exc:
            error = str(exc)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    elapsed = time.monotonic() - started
    save(work / f'{name}.resources.json', samples)
    save(work / f'{name}.sampling-errors.json', dict(monitor.errors))
    fields = sorted({key for sample in samples for key in sample})
    with (work / f'{name}.resources.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    record = {'name': name, 'model': model.name, 'kind': kind, 'command': command,
              'pid': process.pid, 'wall_seconds': elapsed, 'returncode': process.returncode,
              'error': error, 'sampling_errors': dict(monitor.errors), 'expected_source_frames': count}
    try:
        record.update(analyze(rows(output), samples, count, fps, kind))
        if kind == 'soak':
            record['checks']['continuous_600s_or_requested'] = (
                record['source_span_seconds'] >= requested_seconds and
                record['summary'].get('elapsed_ms', 0) / 1000 >= requested_seconds)
        record['checks']['process_exit_zero'] = process.returncode == 0 and error is None
        record['status'] = 'passed' if all(record['checks'].values()) else 'failed'
    except Exception as exc:
        record.update(status='failed', analysis_error=str(exc))
    save(work / f'{name}.measurement.json', record)
    print('DONE', name, record['status'], 'inferences', record.get('summary', {}).get('inference_count'),
          'effective_fps', record.get('steady', {}).get('effective_detection_fps'), flush=True)
    return record


def preflight(binary, model, work, key, env):
    command = [str(binary), '--model', str(model)]
    save(work / f'preflight_{key}.command.json', command)
    print('PREFLIGHT', key, flush=True)
    try:
        with (work / f'preflight_{key}.stdout.log').open('x') as out, (work / f'preflight_{key}.stderr.log').open('x') as err:
            result = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=120)
        raw = (work / f'preflight_{key}.stdout.log').read_text()
        parsed = None
        for match in re.finditer(r'\[', raw):
            try:
                value, _ = json.JSONDecoder().raw_decode(raw[match.start():])
                if isinstance(value, list) and len(value) == 5 and all(isinstance(r, dict) and 'core_mask' in r for r in value):
                    parsed = value
            except ValueError:
                pass
        record = {'passed': result.returncode == 0 and parsed is not None,
                  'returncode': result.returncode, 'core_checks': parsed}
    except Exception as exc:
        record = {'passed': False, 'error': str(exc)}
    save(work / f'preflight_{key}.json', record)
    return record


def value(run, *path):
    obj = run
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def fmt(number, digits=2):
    return '不可用' if number is None else f'{number:.{digits}f}'


def render(work, report):
    lines = ['# 三模型 split 板端实测报告', '', f'开始时间（北京时间）：{report["environment"]["date_shanghai"]}', '',
             '编译检查：' + ('通过' if report['environment']['compile_passed'] else '失败') + '。计量回归及默认模型固定输入检查见 regression-ctest.log。', '',
             '单路相同交通 JPEG 回放，MPP → NV12 DMA-BUF → RGA letterbox → RGB 800×608 → RKNN；'
             '三个独立 Context 分别绑定 Core0/1/2，保留 ByteTrack/Kalman 与无屏画框。conf=0.25，detect_fps=0。', '',
             '| 简称 | 模型文件 | 文件大小 MiB |', '|---|---|---:|']
    for key, model in report['environment']['models'].items():
        lines.append(f'| {key} | `{Path(model["file"]).name}` | {model["bytes"] / 1048576:.2f} |')
    lines += ['',
             '## 验证状态', '', '| 模型 | 固定输入跨核心一致性 | 三轮短测通过 | 连续长测通过 |', '|---|---|---|---|']
    for key in MODELS:
        short = [r for r in report['runs'] if r['key'] == key and r['kind'] == 'short']
        soak = [r for r in report['runs'] if r['key'] == key and r['kind'] == 'soak']
        lines.append(f'| {key} | {"通过" if report["preflight"][key]["passed"] else "失败"} | '
                     f'{sum(r["status"] == "passed" for r in short)}/{report["repeats"]} | '
                     f'{"通过" if soak and soak[0]["status"] == "passed" else "未通过/未运行"} |')
    lines += ['', '一致性检查在同一张已安装 bus 夹具上，对默认核选择、三核与各单核各运行三次；'
              '检查有限数值、类别、候选数、坐标误差≤1px、置信度误差≤0.01。'
              '它验证执行契约和跨核一致性，不能证明工业场景精度。模型属性、输出类型、zp/scale 见 preflight 日志。', '',
              '## 连续长测主要结果', '',
              '| 模型 | 实际推理/输入帧数 | 稳态有效检测 FPS | RSS均值 MiB | PSS均值 MiB | CPU均值 % | NPU0/1/2均值 % | NPU最高温度 °C |',
              '|---|---:|---:|---:|---:|---:|---|---:|']
    for run in report['runs']:
        if run['kind'] != 'soak':
            continue
        s = run.get('summary', {})
        resources = run.get('steady', {}).get('resources', {})
        cpu = value(resources, 'cpu_percent', 'mean')
        loads = '/'.join(fmt(value(resources, f'npu_core{i}_percent', 'mean')) for i in range(3))
        peak = value(run, 'resources_whole_process', 'temp_npu-thermal_c', 'max')
        lines.append(f'| {run["key"]} | {s.get("inference_count", "不可用")}/{run["expected_source_frames"]} | '
                     f'{fmt(value(run,"steady","effective_detection_fps"))} | {fmt(value(resources,"rss_mib","mean"))} | '
                     f'{fmt(value(resources,"pss_mib","mean"))} | {fmt(cpu)} | {loads} | {fmt(peak)} |')
    lines += ['', 'CPU单核=100%；内存列为进程指标，RSS与PSS有重叠，不相加。有效检测FPS受约30.107 FPS输入限制。', '',
              '## 三轮短测：检测速度和资源', '',
              '每轮完整回放相同 1210 帧；顺序轮转，所有硬件进程串行。稳态排除前90帧及末尾30帧。'
              '下表为成功运行的逐遍统计量均值 [最小值, 最大值]，不是合并样本分位数。', '',
              '| 指标 | FP16 | INT8 | Hybrid |', '|---|---:|---:|---:|']
    metrics = [
        ('全程实际推理 FPS', ('summary', 'inference_fps')),
        ('稳态有效检测 FPS', ('steady', 'effective_detection_fps')),
        ('稳态视频输出 FPS', ('steady', 'output_fps')),
        ('源帧检测覆盖率（0~1）', ('steady', 'source_detection_coverage')),
        ('推理阶段均值 ms', ('steady', 'infer_ms', 'mean')),
        ('推理阶段 P95 ms', ('steady', 'infer_ms', 'p95')),
        ('源帧→检测结果 P95 ms', ('steady', 'source_to_detection_ms', 'p95')),
        ('输出所用检测数据年龄 P95 ms', ('steady', 'correction_age_ms', 'p95')),
        ('视频输出延迟 P95 ms', ('steady', 'output_latency_ms', 'p95')),
        ('稳态 RSS 均值 MiB', ('steady', 'resources', 'rss_mib', 'mean')),
        ('全程采样 RSS 峰值 MiB', ('resources_whole_process', 'rss_mib', 'max')),
        ('进程 RSS 高水位 MiB', ('resources_whole_process', 'hwm_mib', 'max')),
        ('稳态 PSS 均值 MiB', ('steady', 'resources', 'pss_mib', 'mean')),
        ('稳态私有内存均值 MiB', ('steady', 'resources', 'private_mib', 'mean')),
        ('稳态可识别 DMA-BUF 均值 MiB', ('steady', 'resources', 'dmabuf_mib', 'mean')),
        ('全程 Swap 峰值 MiB', ('resources_whole_process', 'swap_mib', 'max')),
        ('稳态 CPU 均值 %（单核=100%）', ('steady', 'resources', 'cpu_percent', 'mean')),
    ]
    for core in range(3):
        for label, stat in [('均值', 'mean'), ('P95', 'p95'), ('峰值', 'max')]:
            metrics.append((f'NPU Core{core} 负载{label} %', ('steady', 'resources', f'npu_core{core}_percent', stat)))
    metrics += [('稳态 NPU 频率均值 MHz', ('steady', 'resources', 'npu_hz', 'mean')),
                ('采样 NPU 最高温度 °C', ('resources_whole_process', 'temp_npu-thermal_c', 'max'))]
    aggregates = {}
    for label, path in metrics:
        cells = []
        for key in MODELS:
            vals = [value(r, *path) for r in report['runs'] if r['key'] == key and r['kind'] == 'short' and r['status'] == 'passed']
            vals = [v / 1e6 if path[-2:-1] == ('npu_hz',) else v for v in vals if v is not None]
            aggregates.setdefault(key, {})[label] = stats(vals)
            cells.append(f'{fmt(statistics.mean(vals))} [{fmt(min(vals))}, {fmt(max(vals))}]' if vals else '不可用')
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    report['short_aggregates'] = aggregates
    lines += ['', '## 逐遍计数与丢帧', '',
              '| 运行 | 状态 | 输入 | 解码 | 实际推理 | 有效检测 | 输出 | Core0/1/2有效结果 | 推理待处理丢帧 | 迟到丢弃 |',
              '|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for run in report['runs']:
        s = run.get('summary', {})
        counts = run.get('accepted_results_by_worker', {})
        cells = [run['name'], run['status'], str(s.get('source', {}).get('dequeued', '不可用')),
                 *[str(s.get(k, '不可用')) for k in ['decoded_count', 'inference_count', 'accepted_results', 'frame_count']],
                 '/'.join(str(counts.get(i, counts.get(str(i), 0))) for i in range(3)),
                 str(s.get('inference_pending_dropped', '不可用')), str(s.get('late_results_dropped', '不可用'))]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '完整各阶段丢帧、退出丢弃和队列峰值保存在各遍 measurement.json；'
              '检测框数量不等于检测帧数，包含零目标的已完成推理也计一帧。', '',
              '## 延迟分布（稳态，毫秒）', '', '| 运行 | 阶段 | 均值 | P50 | P95 | P99 |', '|---|---|---:|---:|---:|---:|']
    for run in report['runs']:
        for key in ['preprocess_ms', 'infer_ms', 'postprocess_ms', 'source_to_detection_ms', 'output_latency_ms', 'correction_age_ms']:
            s = run.get('steady', {}).get(key, {})
            lines.append(f'| {run["name"]} | {key} | ' + ' | '.join(fmt(s.get(k)) for k in ['mean', 'p50', 'p95', 'p99']) + ' |')
        decode = run.get('summary', {}).get('timing', {}).get('decode', {})
        lines.append(f'| {run["name"]} | decode（全程，原生1ms直方图） | ' +
                     ' | '.join(fmt(decode.get(k + '_ms')) for k in ['mean', 'p50', 'p95', 'p99']) + ' |')
    lines += ['', '## 连续长测与内存趋势', '',
              '每模型单进程、单套 Context 连续运行；前30秒预热，末尾30帧排除于稳态。'
              '比较稳态首尾各一分钟的中位数，并给出整个稳态的线性回归斜率。', '',
              '| 模型 | 源时间跨度 s | 稳态有效检测 FPS | 资源 | 首分钟中位数 | 尾分钟中位数 | 增量 | 每分钟斜率 |',
              '|---|---:|---:|---|---:|---:|---:|---:|']
    for run in report['runs']:
        if run['kind'] != 'soak':
            continue
        for key, growth in run.get('memory_growth', {}).items():
            lines.append(f'| {run["key"]} | {fmt(run.get("source_span_seconds"))} | '
                         f'{fmt(value(run,"steady","effective_detection_fps"))} | {key} | ' +
                         ' | '.join(fmt(growth.get(k), 4) for k in ['first_minute_median', 'last_minute_median', 'delta', 'slope_per_minute']) + ' |')
    lines += ['', '### 长测资源分布（稳态）', '',
              '| 模型 | 指标 | 均值 | P50 | P95 | P99 | 最小 | 最大 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        if run['kind'] != 'soak':
            continue
        for key, data in run.get('steady', {}).get('resources', {}).items():
            lines.append(f'| {run["key"]} | {key} | ' +
                         ' | '.join(fmt(data.get(k), 4) for k in ['mean', 'p50', 'p95', 'p99', 'min', 'max']) + ' |')
    lines += ['', '### 长测逐分钟', '',
              '| 模型/分钟 | 有效检测 FPS | RSS均值 MiB | PSS均值 MiB | FD最小/最大 | NPU0/1/2均值 % | NPU MHz | NPU温度峰值 °C |',
              '|---|---:|---:|---:|---|---|---:|---:|']
    for run in report['runs']:
        for minute in run.get('minutes', []):
            r = minute['resources']
            cell = lambda key, metric='mean': fmt(value(r, key, metric))
            hz = value(r, 'npu_hz', 'mean')
            lines.append(f'| {run["key"]}/{minute["minute"]} | {fmt(minute["effective_detection_fps"])} | '
                         f'{cell("rss_mib")} | {cell("pss_mib")} | {cell("fd_count","min")}/{cell("fd_count","max")} | ' +
                         '/'.join(cell(f'npu_core{i}_percent') for i in range(3)) +
                         f' | {fmt(hz / 1e6 if hz is not None else None)} | {cell("temp_npu-thermal_c","max")} |')
    lines += ['', '## 口径、边界与复现', '',
              '- 全程推理 FPS 来自原生 inference_count/elapsed；稳态有效检测 FPS 按固定时间窗内完成且被接受的结果计数。覆盖率按源序号计算。迟到结果不更新轨迹。',
              '- infer_ms 包含输入绑定、Runtime 执行及输出获取/转换，不是纯 NPU 硬件时间。输出异步运行，因此输出 FPS 不代表检测速度。',
              '- 约30.107 FPS的输入限制了本次测量；不估算最大吞吐。逐事件延迟只覆盖有效接受的结果；decode保留原生全程直方图。',
              '- RSS/PSS/私有内存来自/proc；VmHWM为采到的进程高水位。DMA-BUF按进程FD引用的inode去重，仅覆盖可识别导出对象，可能遗漏RKNN驱动内部内存；0表示未识别到对象，不表示硬件不占内存。这些口径有重叠，禁止直接相加。',
              '- CPU单核100%；各NPU核心负载分别报告，来自驱动窗口采样。CPU/NPU频率及各热区温度原始值均保留于CSV。系统采用原有动态频率策略，未锁频。',
              '- 缺失监控值保持null/不可用，采样错误另存。没有业务FPS/延迟门槛，不宣称性能达标；10分钟结果不构成长期无泄漏保证。',
              '- 本次使用真实MPP/RGA/NPU执行录像回放；未测真实摄像头、显示曝光延迟、最大吞吐、完整mAP或多路视频。',
              '- 未修改SDK、内核、设备树、驱动、引导、设备权限、系统服务或生产默认配置；测试会暂时占用NPU/RGA/MPP并增加功耗与温度。', '',
              '```bash', 'cd /home/yolo/test', 'cmake --preset rk3588-release', 'cmake --build --preset rk3588-release',
              '/usr/bin/python3 tools/benchmark_split_models.py --output-dir results/split_models_NEW --repeats 3 --soak-seconds 600', '```', '',
              '命令、模型/输入/二进制/配置/SDK哈希、源码快照、逐遍JSONL、measurement.json、resources.csv/json均在本目录。'
              'compile-verification.log记录编译检查；report.json包含全部验证状态、资源分布和逐分钟数据。', '']
    failed = [r['name'] for r in report['runs'] if r['status'] != 'passed']
    if failed:
        lines += ['失败运行：' + '、'.join(failed), '']
    (work / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--soak-seconds', type=float, default=600)
    parser.add_argument('--binary', type=Path, default=DEFAULT_BUILD / 'camera_live')
    parser.add_argument('--preflight-binary', type=Path, default=DEFAULT_BUILD / 'live_npu_test')
    parser.add_argument('--fixture', type=Path, default=TEST_ROOT / 'validation/camera_live_20261018/traffic_fixture/manifest.jsonl')
    args = parser.parse_args()
    if args.repeats < 1 or not math.isfinite(args.soak_seconds) or args.soak_seconds <= 31:
        parser.error('repeats must be positive; soak-seconds must be finite and >31')
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ['LD_LIBRARY_PATH', 'LD_PRELOAD']:
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    binary, native_test, fixture = args.binary.resolve(), args.preflight_binary.resolve(), args.fixture.resolve()
    source = rows(fixture)
    fps, count = source[0]['negotiated_fps'], len(source) - 1
    if count <= 120:
        raise ValueError('short source needs >120 frames')
    input_hashes = {r['file']: sha(fixture.parent / r['file']) for r in source[1:]}
    save(work / 'fixture-jpeg-sha256.json', input_hashes)
    long_fixture, long_count = make_long_fixture(fixture, work / 'soak_fixture', args.soak_seconds)
    # Preserve configuration in a new location with explicit existing asset paths.
    config_text = (SOURCE / 'config.yaml').read_text()
    config = work / 'config.snapshot.yaml'
    config.write_text(config_text.replace('labels_path: deploy/classes.json',
                                         'labels_path: ' + json.dumps(str(SOURCE / 'deploy/classes.json')))
                      .replace('model_path: ../yolo_model_rknn/yolo26s_800x608_fp16.rknn',
                               'model_path: ' + json.dumps(str(MODELS['fp16']))))
    for path in [Path(__file__), TEST_ROOT / 'cpp/live_npu_test.cpp', TEST_ROOT / 'tools/compare_split_precision.py']:
        (work / ('executed_' + path.name)).write_bytes(path.read_bytes())
    dependencies = subprocess.run(['/usr/bin/ldd', str(binary)], env=env, capture_output=True, text=True, check=True).stdout
    (work / 'ldd.log').write_text(dependencies)
    if any(s in dependencies.lower() for s in ['conda', 'miniforge', 'not found']):
        raise RuntimeError('unexpected native library dependencies')
    build_root = binary.parent.parent if binary.parent.name == 'bin' else binary.parent
    with (work / 'compile-verification.log').open('x') as build_log:
        for command in native_build_commands(binary, native_test):
            build_log.write(json.dumps(command) + '\n')
            build_log.flush()
            build_result = subprocess.run(command, env={**env, 'PATH': '/usr/bin:/bin'},
                                          stdout=build_log, stderr=subprocess.STDOUT)
            if build_result.returncode:
                raise RuntimeError('native build verification failed')
    (work / 'baseline-meminfo.txt').write_text(Path('/proc/meminfo').read_text())
    baseline = subprocess.run(['/usr/bin/ps', '-eo', 'pid,comm,pcpu', '--sort=-pcpu'],
                              capture_output=True, text=True, check=True)
    (work / 'baseline-processes.txt').write_text(baseline.stdout)
    for filename in ['CMakeCache.txt', 'compile_commands.json']:
        if (build_root / filename).is_file():
            (work / filename).write_bytes((build_root / filename).read_bytes())
    sdk = [Path('/opt/rknn/runtime-2.5.0/lib/librknnrt.so'), Path('/opt/rockchip/librga/lib/librga.so'),
           Path('/opt/rockchip/mpp/lib/librockchip_mpp.so')]
    environment = {'date_shanghai': datetime.datetime.now(SHANGHAI).isoformat(), 'platform': platform.platform(),
                   'models': {k: {'file': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for k, p in MODELS.items()},
                   'binary': str(binary), 'binary_sha256': sha(binary), 'preflight_binary_sha256': sha(native_test),
                   'original_config_sha256': sha(SOURCE / 'config.yaml'), 'config_sha256': sha(config),
                   'fixture_sha256': sha(fixture), 'soak_fixture_sha256': sha(long_fixture),
                   'classes_sha256': sha(SOURCE / 'deploy/classes.json'),
                   'sdk_sha256': {str(p): sha(p) for p in sdk}, 'script_sha256': sha(Path(__file__)),
                   'source_fps': fps, 'short_source_frames': count, 'soak_source_frames': long_count,
                   'npu_governor': Path('/sys/class/devfreq/fdab0000.npu/governor').read_text().strip(),
                   'npu_driver': Path('/sys/kernel/debug/rknpu/version').read_text().strip(),
                   'compile_passed': True}
    save(work / 'environment.json', environment)
    report = {'environment': environment, 'repeats': args.repeats, 'soak_seconds': args.soak_seconds,
              'preflight': {}, 'runs': []}
    for key, model in MODELS.items():
        report['preflight'][key] = preflight(native_test, model, work, key, env)
        save(work / 'checkpoint.json', report)
    order = list(MODELS)
    for repeat in range(args.repeats):
        rotated = order[repeat % 3:] + order[:repeat % 3]
        for key in rotated:
            if not report['preflight'][key]['passed']:
                continue
            run = run_case(binary, config, fixture, MODELS[key], work, f'short_{key}_{repeat + 1}', env, count, fps, 'short', (count - 1) / fps)
            run['key'] = key
            report['runs'].append(run)
            save(work / 'checkpoint.json', report)
    for key, model in MODELS.items():
        if not report['preflight'][key]['passed']:
            continue
        run = run_case(binary, config, long_fixture, model, work, f'soak_{key}', env, long_count, fps, 'soak', args.soak_seconds)
        run['key'] = key
        report['runs'].append(run)
        save(work / 'checkpoint.json', report)
    report['measurement_finished_shanghai'] = datetime.datetime.now(SHANGHAI).isoformat()
    render(work, report)
    save(work / 'report.json', report)
    success = all(p['passed'] for p in report['preflight'].values()) and all(r['status'] == 'passed' for r in report['runs'])
    print('REPORT', work / 'report.md', 'all_passed', success, flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

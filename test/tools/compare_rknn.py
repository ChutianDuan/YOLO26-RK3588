#!/usr/bin/python3
"""Paired RK3588 performance/quality experiments against the supplied BDD pack.

Uses native C++ for all media/NPU work and the pack's verified standard-library
evaluator. No SDK installation, Python inference engine or system tuning.
"""
import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import hashlib
import importlib
import functools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from paths import SOURCE, TEST_ROOT, DEFAULT_BUILD, DEFAULT_DATASET, resolve_artifact, load_history


def ap101_envelope(tp_flags, fp_flags, ground_truth_count):
    """Exactly the reference's 101 levels, using monotone recall/suffix maxima."""
    if ground_truth_count <= 0:
        return 0.0
    recalls, precisions = [], []
    true_positives = false_positives = 0
    for tp, fp in zip(tp_flags, fp_flags):
        true_positives += tp
        false_positives += fp
        recalls.append(true_positives / ground_truth_count)
        precisions.append(true_positives / max(1, true_positives + false_positives))
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(precisions[index] if index < len(precisions) else 0.0
               for index in (bisect.bisect_left(recalls, level / 100.0) for level in range(101))) / 101.0


def accelerate_evaluator(evaluator, verbose=True):
    metrics = sys.modules[evaluator.detection_metrics.__module__]
    original_ap = getattr(metrics, '_reference_ap101', metrics.ap101)
    original_iou = getattr(metrics, '_reference_box_iou', metrics.box_iou)
    metrics._reference_ap101, metrics._reference_box_iou = original_ap, original_iou
    rng = random.Random(3588)
    checks = 0
    for size in (0, 1, 2, 10, 100, 1000):
        for ground_truth in (0, 1, 7, 123):
            flags = [rng.randrange(2) for _ in range(size)]
            opposite = [1 - value for value in flags]
            assert ap101_envelope(flags, opposite, ground_truth) == original_ap(flags, opposite, ground_truth)
            checks += 1
    cached_iou = functools.lru_cache(maxsize=1_000_000)(original_iou)
    evaluator.box_iou = metrics.box_iou = cached_iou
    metrics.ap101 = ap101_envelope
    if verbose:
        print(f'AP101 exact parity: {checks} deterministic checks; bounded IoU memoization enabled', flush=True)
    return cached_iou


def score_detection_file(pack_string, path_string):
    pack = Path(pack_string)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(pack / 'tools'))
    evaluator = importlib.import_module('compare')
    evaluator.ROOT = pack
    cache = accelerate_evaluator(evaluator, verbose=False)
    manifest, labels, crowds = evaluator.load_context()
    predicted, metadata = evaluator.load_predictions(Path(path_string), manifest, set(labels))
    result = evaluator.score_predictions(predicted, labels, crowds)
    cache.cache_clear()
    return result, metadata


def sha(path):
    digest = hashlib.sha256()
    with resolve_artifact(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_summary(path):
    summary = None
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['type'] == 'summary':
                if summary is not None:
                    raise ValueError('multiple summaries')
                summary = row
    if summary is None:
        raise ValueError(f'Incomplete run: {path}')
    return summary


def observed_run(binary, pack, work, case, environment):
    prefix = work / case['name']
    output, ready = prefix.with_suffix('.jsonl'), prefix.with_suffix('.ready')
    config = prefix.with_suffix('.yaml')
    config.write_text('\n'.join([
        'model_path: ' + json.dumps(case['model']),
        'labels_path: ' + json.dumps(str(pack / 'classes.json')),
        'input_width: 800', 'input_height: 608', 'num_classes: 10',
        'conf_threshold: ' + ('.001' if case['mode'] == 'detection' else '.25'),
        'class_conf_thresholds: []', 'preprocess: ' + case['preprocess'], 'use_letterbox: true',
        'lk_preprocess: ' + case.get('lk_preprocess', 'rga'),
        'lk_scale: ' + str(case.get('lk_scale', 1.0)),
        'video_detect_fps: ' + ('0' if case['mode'] == 'detection' else '4'),
        'video_model_async: ' + str(case.get('async', False)).lower(),
        'video_stride_mode: ' + ('dynamic' if case.get('async') else 'fixed'),
        'opencv_threads: 2', 'inference_timeout_ms: 60000',
    ]) + '\n')
    command = [str(binary), '--config', str(config), '--input', str(pack / 'video/002d290d-90f2bab2.mov'),
               '--output', str(output), '--mode', case['mode'], '--frame-map', str(pack / 'frame_map.json'),
               '--ready-file', str(ready)]
    print('START ' + case['name'], flush=True)
    started = time.monotonic()
    samples = []
    with prefix.with_suffix('.stderr.log').open('w') as stderr, prefix.with_suffix('.stdout.log').open('w') as stdout:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        try:
            while proc.poll() is None:
                if ready.exists():
                    try:
                        status = Path(f'/proc/{proc.pid}/status').read_text()
                        rss = next(int(line.split()[1]) / 1024 for line in status.splitlines() if line.startswith('VmRSS:'))
                        sample = {'seconds': time.monotonic() - started, 'rss_mib': rss}
                        for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
                            sample[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000
                        sample['cpu_frequency_khz'] = {p.name: int((p / 'scaling_cur_freq').read_text())
                                                       for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*')}
                        sample['npu_frequency_hz'] = {p.name: int((p / 'cur_freq').read_text())
                                                      for p in Path('/sys/class/devfreq').glob('*npu*')}
                        samples.append(sample)
                    except (OSError, StopIteration, ValueError):
                        pass
                if time.monotonic() - started > 900:
                    raise TimeoutError(case['name'])
                time.sleep(.05)
            if proc.returncode:
                raise RuntimeError(prefix.with_suffix('.stderr.log').read_text())
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    if not samples:
        raise ValueError('No runtime RSS samples')
    if prefix.with_suffix('.stdout.log').stat().st_size:
        raise ValueError('SDK/application diagnostics escaped stderr')
    result = dict(case, output=str(output), config=str(config), command=command,
                  summary=read_summary(output), rss_samples=samples,
                  mean_rss_mib=statistics.mean(s['rss_mib'] for s in samples),
                  model_bytes=Path(case['model']).stat().st_size,
                  prediction_sha256=sha(output), config_sha256=sha(config))
    summary = result['summary']
    if summary['timing']['decode']['count'] != 1210:
        raise ValueError('Incomplete decoding or source-frame mismatch')
    if case['mode'] == 'detection':
        if summary['frame_count'] != 202 or summary['inference_count'] != 202:
            raise ValueError('Detection subset mismatch')
    else:
        if summary['frame_count'] != 1210 or summary['history_peak'] > 64 or summary['max_pending'] > 1 or summary['pending_at_end']:
            raise ValueError('Pipeline bounds/frame count mismatch')
        if any(summary['timing'][key]['count'] != 1210 for key in ('flow', 'tracker')):
            raise ValueError('Invalid per-frame flow/ByteTrack denominators')
        if summary['async_submissions'] != summary['async_corrections'] + summary['discarded_results']:
            raise ValueError('Missing asynchronous result')
    write_json(prefix.with_suffix('.measurement.json'), result)
    print(f"DONE {case['name']}: infer={summary['timing']['infer']['mean_ms']:.3f} ms, RSS={result['mean_rss_mib']:.2f} MiB", flush=True)
    return result


def pipeline_quality(path, manifest, labels, crowds, evaluator):
    predictions = {}
    seen = set()
    out_of_frame = 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['type'] != 'frame':
                continue
            index = row['frame_index']
            if not isinstance(index, int) or index in seen or index != len(seen):
                raise ValueError('Duplicate or nonconsecutive pipeline frame')
            seen.add(index)
            if (row['width'], row['height']) != (1280, 720):
                raise ValueError('Pipeline rotation/original coordinates mismatch')
            if index not in labels:
                continue
            predictions[index] = []
            for item in row['tracks']:
                cls, score, box = item['class_id'], item['score'], tuple(item['bbox'])
                if not isinstance(cls, int) or not 0 <= cls < 10 or len(box) != 4 or not all(math.isfinite(v) for v in (*box, score)):
                    raise ValueError('Invalid track')
                if item['class_name'] != manifest['classes'][cls] or not 0 <= score <= 1:
                    raise ValueError('Track class/score mismatch')
                # The supplied tracking evaluator preserves complete Kalman
                # boxes. Unlike detector outputs, tracks may extend outside
                # the image; use their full area in IoU, without clipping.
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError('Invalid track coordinates')
                if cls not in manifest['annotated_class_ids'] or score < .25:
                    continue
                if box[0] < 0 or box[1] < 0 or box[2] > 1280 or box[3] > 720:
                    out_of_frame += 1
                predictions[index].append(evaluator.ObjectRecord(index, str(item['track_id']), item['class_name'], box, score))
    if len(seen) != 1210 or set(predictions) != set(labels):
        raise ValueError('Missing evaluation frames')
    candidates, ignored = evaluator.suppress_ignored(predictions, labels, crowds, .5)
    pooled, per_class = evaluator.detection_metrics(labels, candidates, .5)
    return {'operating_point': {k: v for k, v in pooled.items() if k != 'map50'},
            'per_class': {name: {k: v for k, v in row.items() if k != 'ap50'} for name, row in per_class.items()},
            'ignored_predictions': ignored, 'prediction_field': 'tracks', 'confidence': .25, 'iou': .5,
            'out_of_frame_tracks_evaluated': out_of_frame, 'track_box_policy': 'unclipped, matching supplied tracking evaluator'}


def aggregate(runs):
    stages = {}
    for stage in ('decode', 'preprocess', 'infer', 'postprocess', 'flow', 'tracker'):
        count = sum(r['summary']['timing'][stage]['count'] for r in runs)
        total = sum(r['summary']['timing'][stage]['total_ms'] for r in runs)
        stages[stage] = total / count if count else 0
    samples = [s['rss_mib'] for r in runs for s in r['rss_samples']]
    result = {'stages_mean_ms': stages, 'mean_rss_mib': statistics.mean(samples),
              'model_mb': runs[0]['model_bytes'] / 1e6,
              'processing_fps': sum(r['summary']['frame_count'] for r in runs) * 1000 / sum(r['summary']['elapsed_ms'] for r in runs),
              'inference_count_mean': statistics.mean(r['summary']['inference_count'] for r in runs),
              'recall': statistics.mean(r['quality']['operating_point']['recall'] for r in runs),
              'precision': statistics.mean(r['quality']['operating_point']['precision'] for r in runs),
              'f1': statistics.mean(r['quality']['operating_point']['f1'] for r in runs),
              'person_recall': statistics.mean(r['quality']['per_class']['person']['recall'] for r in runs),
              'person_f1': statistics.mean(r['quality']['per_class']['person']['f1'] for r in runs)}
    return result


def render(report, work):
    totals, detector = report['aggregates'], report['detection_quality']
    a, b = totals['fp16'], totals['int8']
    lines = ['# RK3588：YOLO26 FP16 / INT8 与 LK 对比实验', '',
             '日期：' + report['date_shanghai'], '',
             '数据：BDD100K train 的 002d290d-90f2bab2，1210 帧、202 个标注帧、1280×720 显示方向。标注帧 n 对应源帧 n×6。',
             '普通真值 4400 个，覆盖 6 类；人体 2587 个。crowd/distractor 按材料包规则忽略，未标注的交通灯/标志不计入精度。', '',
             '## 板端主对比', '',
             f"FP16 和 INT8 各完成 {report['repeats']} 次整段视频运行，交替执行。RGA 预处理、检测目标 4 FPS、异步与动态步长、LK、ByteTrack；每个进程预热 5 次后开始测量，配置与部署默认配置隔离。",
             '纯模型 AP 独立在 202 个标注帧推理，保存 conf≥0.001 候选；主表 AP 使用 RGA。LK 召回率/F1 读取整段流程的 tracks，在标注帧以 conf≥0.25、IoU≥0.50 统计。', '',
             'P/R/F1 在 MOT 可标注的 0～7 类统计，六类在本片段有普通真值；无真值的受支持类别预测仍计误报。F1 使用逐帧同类别框的一对一 IoU 匹配，未计算轨迹 ID 指标。', '',
             '跟踪评测与材料原跟踪评测器一致：Kalman 框可以延伸至画面外，保留完整预测框计算 IoU，不裁剪后再统计；原始预测文件保持不变。', '',
             '| 指标 | FP16＋LK＋ByteTrack | INT8＋LK＋ByteTrack |', '|---|---:|---:|']
    def row(label, x, y, fmt):
        lines.append(f'| {label} | {fmt(x)} | {fmt(y)} |')
    ms = lambda x: f'{x:.3f} ms'
    pct = lambda x: f'{100*x:.2f}%'
    for name, key in [('模型推理平均', 'infer'), ('预处理平均', 'preprocess'), ('后处理平均', 'postprocess')]:
        row(name, a['stages_mean_ms'][key], b['stages_mean_ms'][key], ms)
    row('模型文件大小', a['model_mb'], b['model_mb'], lambda x: f'{x:.2f} MB')
    row('运行期平均 RSS', a['mean_rss_mib'], b['mean_rss_mib'], lambda x: f'{x:.2f} MiB')
    for label, key in [('模型 mAP50（RGA）', 'map50'), ('模型 mAP50–95（RGA）', 'map50_95')]:
        row(label, detector['fp16_rga'][key], detector['int8_rga'][key], pct)
    for label, key in [('单模型＋LK 召回率（MOT 微平均）', 'recall'), ('单模型＋LK F1（MOT 微平均）', 'f1'),
                       ('单模型＋LK 人体召回率', 'person_recall'), ('单模型＋LK 人体 F1', 'person_f1')]:
        row(label, a[key], b[key], pct)
    row('视频处理 FPS（含 LK 帧）', a['processing_fps'], b['processing_fps'], lambda x: f'{x:.2f}')
    row('每遍实际推理次数', a['inference_count_mean'], b['inference_count_mean'], lambda x: f'{x:.2f}')
    lines += ['', f"INT8 模型调用加速 {a['stages_mean_ms']['infer']/b['stages_mean_ms']['infer']:.2f} 倍；整流程 FPS 提高 {(b['processing_fps']/a['processing_fps']-1)*100:.2f}%。LK 与解码仍占主线程主要耗时。", '']
    lines += ['', '## 阶段耗时', '', '| 阶段 | FP16 | INT8 |', '|---|---:|---:|']
    for label, key in [('Decode：视频读取与解码', 'decode'), ('Preprocess：CPU 复制/同步＋RGA', 'preprocess'),
                       ('YOLO：设置输入＋NPU 执行＋获取输出', 'infer'), ('LK Flow：灰度转换＋光流＋运动补偿', 'flow'),
                       ('ByteTrack：检测/光流轨迹更新', 'tracker'), ('Postprocess：解析＋双阶段 Top-K，无 NMS', 'postprocess')]:
        row(label, a['stages_mean_ms'][key], b['stages_mean_ms'][key], ms)
    lines += ['', 'Decode/LK/ByteTrack 为每个解码帧的平均；Preprocess/YOLO/Postprocess 为每次推理的平均。次数不同且异步重叠，不能将各行相加作为每帧墙钟时间。',
              'RSS 每 50 ms 采样，排除模型加载与 5 次预热，按实际运行样本合并求平均；MB=10⁶ 字节，MiB=2²⁰ 字节。',
              'RGA 预处理包含 OpenCV Mat 到 DMA-BUF 的复制与 cache sync，RKNN 使用 DMA-BUF FD 输入。跟踪运行保留全部 10 类；精度统计按 MOT 标注范围筛选。模型输出的框数量也会影响 LK 工作量。',
              '动态异步调度会使两个模型的检测帧与推理次数不同；主表是当前流程对比，不能将跟踪差异全部归因于量化。', '',
              '## 与材料中的 FP32 精度基线对齐', '',
              '本板只有 FP16 和 INT8 RKNN，不能把 FP16 标成 FP32。FP32 是材料包中另一台主机预先保存的结果；不使用其耗时/RSS冒充板端性能，也没有 FP32＋LK 数据。', '',
              '| 模型 / 预处理 | mAP50 | mAP50–95 | 人体 AP50 | 人体 AP50–95 | 人体召回率 @0.25 | 人体 F1 @0.25 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, metrics in [('主机历史 FP32 / OpenCV', report['fp32_reference']),
                          ('RKNN FP16 / RGA', detector['fp16_rga']), ('RKNN INT8 / RGA', detector['int8_rga'])]:
        person = metrics['per_class']['person']
        op = person['operating_point']
        values = [metrics['map50'], metrics['map50_95'], person['ap50'], person['ap50_95'], op['recall'], op['f1']]
        lines.append('| ' + name + ' | ' + ' | '.join(pct(x) for x in values) + ' |')
    lines += ['', 'AP 为材料评测器的 101 点插值，mAP50–95 为 10 个 IoU 阈值平均，仅对有普通真值的六类取平均；不是官方 COCO 全套评测，也不是十类独立验证集 mAP。',
              '主机历史基线与板端 RGA 插值及 INT8 分数并列值可能使结果不同。', '',
              '## 固定相同检测帧的对照', '',
              '另各跑一遍同步、固定步长 8 的 RGA＋LK＋ByteTrack，两个模型均在相同源帧检测，排除异步速度引起的检测频率差异。', '',
              '| 模型 | 推理次数 | MOT Recall | MOT F1 | 人体 Recall | 人体 F1 |', '|---|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        if run['group'] != 'controlled':
            continue
        q = run['quality']; op = q['operating_point']; person = q['per_class']['person']
        lines.append(f"| {run['precision'].upper()} | {run['summary']['inference_count']} | {pct(op['recall'])} | {pct(op['f1'])} | {pct(person['recall'])} | {pct(person['f1'])} |")
    lines += ['', '## 每次运行与复现', '', '| 运行 | 帧数 | 推理次数 | 推理均值 ms | 平均 RSS MiB | MOT Recall | MOT F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        if run['mode'] != 'pipeline':
            continue
        s, q = run['summary'], run['quality']['operating_point']
        lines.append(f"| {run['name']} | {s['frame_count']} | {s['inference_count']} | {s['timing']['infer']['mean_ms']:.3f} | {run['mean_rss_mib']:.2f} | {pct(q['recall'])} | {pct(q['f1'])} |")
    max_temp = max((s.get('npu-thermal', 0) for r in report['runs'] for s in r['rss_samples']), default=0)
    reference = read_summary(Path(report['dataset']) / 'baseline/fp32_predictions.jsonl')
    lines += ['', f"主机 FP32 基线使用 ONNX Runtime {reference['runtime_version']}、OpenCV {reference['opencv_version']}、{reference['threads']} 个推理线程；板端为系统 OpenCV 4.10.0、RKNN Runtime 2.5.0，OpenCV 线程数为 2。跨主机精度对照也包含运行库与预处理实现差异。", '',
              f'测量期间 NPU 温度峰值：{max_temp:.2f} °C。保持系统默认频率策略与默认 NPU 核心策略，没有设置性能模式、改驱动或启动服务。',
              '视频、GT、模型及材料包文件均校验哈希；已有 FP32 指标通过同一评测器重新计算并对齐。板端测量进程串行执行，没有同时争抢 NPU。',
              '全部硬件测量完成后才计算精度，最多 4 个固定 CPU 进程；仅缓存 IoU 并用等价精度包络加速原 AP101。24 组确定性用例完全一致、原评测器 4 项回归通过、FP32 全部指标逐字段一致；原评测器文件未修改。',
              '双阶段 Top-K 按材料包 yolo26_deploy.py 对齐；[Ultralytics Detect 实现](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)同样先选 anchor，再选 anchor/class 对；未加入 NMS。', '',
              '```bash', 'cd /home/yolo/test', 'cmake --build --preset rk3588-release',
              f'/usr/bin/python3 tools/compare_rknn.py --dataset {report["dataset"]} --repeats {report["repeats"]} --output-dir validation/bdd_compare_NEW', '```', '',
              f'[完整机器报告及逐类指标]({work / "report.json"})；目录中保留每次配置、命令、预测 JSONL、RSS/温度采样及 stderr。',
              '单个定向选择的训练序列不代表独立测试集泛化；本报告不宣称满足未指定的工业精度、吞吐量或延迟指标。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--from-dir', type=Path, help='Existing experiment for --report-only; output must be new')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error('--repeats must be 1..10')
    pack, work = args.dataset.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(pack / 'tools'))
    sys.dont_write_bytecode = True
    evaluator = importlib.import_module('compare')
    evaluator.ROOT = pack
    manifest, labels, crowds = evaluator.load_context()
    binary = args.build_dir.resolve() / 'benchmark_video'
    if args.report_only and not args.from_dir:
        parser.error('--report-only requires --from-dir')
    work.mkdir(parents=True, exist_ok=False)
    if not args.report_only:
        check = subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=pack, capture_output=True, text=True, check=True)
        (work / 'dataset-integrity.log').write_text(check.stdout)
        models = {}
        for precision in ('fp16', 'int8'):
            models[precision] = str((SOURCE.parent / 'yolo_model_rknn' / f'yolo26s_800x608_{precision}.rknn').resolve())
            if sha(models[precision]) != manifest['rknn'][precision]['sha256']:
                raise ValueError('RKNN model hash differs from comparison pack')
        environment = dict(os.environ)
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
            environment.pop(key, None)
        experiment = {'dataset': str(pack), 'repeats': args.repeats,
                      'date_shanghai': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                      'platform': platform.platform(), 'binary_sha256': sha(binary),
                      'models': {p: {'path': path, 'sha256': sha(path)} for p, path in models.items()},
                      'source_sha256': {str(p.relative_to(TEST_ROOT.parent)): sha(p) for p in [SOURCE / 'model/yolo_postprocess.cpp', SOURCE / 'video/video_inference.cpp', TEST_ROOT / 'tools/benchmark_video.cpp', Path(__file__)]},
                      'runs': []}
        cases = []
        for method in ('rga',):
            for precision in ('fp16', 'int8'):
                cases.append({'name': f'detection_{precision}_{method}', 'precision': precision, 'preprocess': method,
                              'mode': 'detection', 'group': 'detection', 'model': models[precision]})
        for repetition in range(args.repeats):
            order = ('fp16', 'int8') if repetition % 2 == 0 else ('int8', 'fp16')
            for precision in order:
                cases.append({'name': f'pipeline_{precision}_{repetition+1}', 'precision': precision, 'preprocess': 'rga',
                              'mode': 'pipeline', 'group': 'primary', 'model': models[precision], 'async': True})
        for precision in ('fp16', 'int8'):
            cases.append({'name': f'controlled_{precision}', 'precision': precision, 'preprocess': 'rga',
                          'mode': 'pipeline', 'group': 'controlled', 'model': models[precision], 'async': False})
        write_json(work / 'experiment.json', experiment)
        for case in cases:
            experiment['runs'].append(observed_run(binary, pack, work, case, environment))
            write_json(work / 'experiment.json', experiment)
    else:
        experiment = load_history(args.from_dir / 'experiment.json')
        experiment['regenerated_from'] = str(args.from_dir.resolve())
    cached_iou = accelerate_evaluator(evaluator)
    print('Scoring FP32 and four detector variants with four bounded CPU workers...', flush=True)
    jobs = {'FP32': pack / 'baseline/fp32_predictions.jsonl'}
    jobs.update({run['name']: Path(run['output']) for run in experiment['runs'] if run['mode'] == 'detection'})
    scored = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(score_detection_file, str(pack), str(path)): name for name, path in jobs.items()}
        for task in as_completed(tasks):
            name = tasks[task]
            scored[name] = task.result()
            print('Scored ' + name, flush=True)
            write_json(work / 'detection-quality-checkpoint.json', scored)
    fp32, fp32_metadata = scored['FP32']
    expected = json.loads((pack / 'baseline/fp32_metrics.json').read_text())['models']['FP32']
    for key in ('map50', 'map50_95', 'operating_point', 'per_class'):
        if fp32[key] != expected[key]:
            raise ValueError('FP32 evaluator parity failed: ' + key)
    cached_iou.cache_clear()
    report = dict(experiment, fp32_reference=fp32, fp32_reference_input=fp32_metadata,
                  dataset_manifest=manifest, report_generator_sha256=sha(Path(__file__)),
                  evaluator_validation={'exact_ap101_cases': 24, 'fp32_all_metric_fields_match': True,
                                        'iou_cache_maxsize': 1_000_000}, detection_quality={})
    for run in report['runs']:
        print('Scoring ' + run['name'], flush=True)
        if sha(run['output']) != run['prediction_sha256']:
            raise ValueError('Recorded prediction has changed')
        if run['mode'] == 'detection':
            run['quality'] = scored[run['name']][0]
            report['detection_quality'][run['precision'] + '_' + run['preprocess']] = run['quality']
        else:
            run['quality'] = pipeline_quality(run['output'], manifest, labels, crowds, evaluator)
        cached_iou.cache_clear()
    report['aggregates'] = {p: aggregate([r for r in report['runs'] if r['group'] == 'primary' and r['precision'] == p]) for p in ('fp16', 'int8')}
    write_json(work / 'report.json', report)
    markdown = render(report, work)
    (work / 'report.md').write_text(markdown)
    print('REPORT ' + str(work / 'report.md'), flush=True)


if __name__ == '__main__':
    main()

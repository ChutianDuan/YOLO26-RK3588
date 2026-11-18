#!/usr/bin/python3
"""Compare RGA grayscale scales for CPU LK, or read historical CPU/RGA reports."""
import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import statistics
import sys

from paths import SOURCE, DEFAULT_BUILD, load_history
from compare_rknn import accelerate_evaluator, observed_run, pipeline_quality, sha, write_json


def render(runs, work, pack, binary_hash=None, provenance=None):
    lines = ['# LK 的 RGA / DMA-BUF 对比实验', '',
             '日期：' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), '',
             '数据：BDD100K 002d290d-90f2bab2，1210 帧、1280×720，202 个标注帧。',
             '系统 OpenCV 2 线程，检测预处理均为 RGA；模型、SDK、系统频率策略不变。每个进程先预热 5 次。',
             'fixed：同步、固定步长 8，恰好 152 次检测；dynamic：4 FPS 目标、异步动态步长。所有硬件运行串行执行。',
             'Flow 为每个解码帧的平均，包含灰度准备（CPU→DMA 复制、Cache Sync、RGA 同步等待）、特征/金字塔/双向 LK、运动估计、异步框补偿；ByteTrack 单独计时。',
             '所有配置均使用 CPU 执行 LK；RGA 输出的灰度图通过 mmap 直接供 CPU 读取，没有再复制灰度图，也未实现端到端零拷贝。', '',
             '实现：RGA3 将 BGR DMA-BUF 转为 NV12，LK 直接读取 Y 平面；宽高取最近偶数、stride 对齐 16，前后帧固定双缓冲并持有 DMA READ 租约。',
             'RK3588 的 Y400 和 color fill 依赖 RGA2，system heap 的高地址页可能导致 no core match。最终实现避免 Y400，检测 padding 仅首次/几何变化时由 CPU 初始化为 114；两路 RGA 任务均通过单任务选项使用 RGA3，NPU 核心策略不变。', '',
             '| 调度 | 模型 | LK 图像来源 / 比例 | 次数 | Flow ms/帧 | 灰度准备 ms/帧 | FPS | RSS MiB | Recall | F1 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    groups = {}
    for run in runs:
        groups.setdefault((run['schedule'], run['precision'], run['lk_preprocess'], run['lk_scale']), []).append(run)
    aggregates = []
    for (schedule, precision, method, scale), rows in groups.items():
        frames = sum(r['summary']['frame_count'] for r in rows)
        def stage(key):
            return sum(r['summary']['timing'][key]['total_ms'] for r in rows) / frames
        details = {key: sum(r['summary']['flow_detail'][key]['total_ms'] for r in rows) / frames
                   for key in rows[0]['summary']['flow_detail']}
        fps = frames * 1000 / sum(r['summary']['elapsed_ms'] for r in rows)
        recall = statistics.mean(r['quality']['operating_point']['recall'] for r in rows)
        f1 = statistics.mean(r['quality']['operating_point']['f1'] for r in rows)
        rss = statistics.mean(s['rss_mib'] for r in rows for s in r['rss_samples'])
        row = dict(schedule=schedule, precision=precision, method=method, scale=scale, repeats=len(rows),
                   flow_ms=stage('flow'), prepare_ms=details['prepare'], fps=fps, recall=recall, f1=f1,
                   rss_mib=rss, detail_ms=details)
        aggregates.append(row)
        lines.append(f'| {schedule} | {precision} | {method} / {scale:g} | {len(rows)} | {row["flow_ms"]:.3f} | {row["prepare_ms"]:.3f} | {fps:.2f} | {rss:.2f} | {recall:.2%} | {f1:.2%} |')
    lines += ['', '## Flow 内部分项（ms/解码帧，跳过的阶段计 0）', '',
              '| 调度 / 模型 / 图像 | 准备 | 帧差 | 特征 | 金字塔 | 正向 LK | 反向 LK | 质量检查 | 运动估计 | 框补偿 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregates:
        values = ' | '.join(f'{row["detail_ms"][key]:.3f}' for key in
                            ('prepare','frame_diff','features','pyramid','forward','backward','quality','motion','compensation'))
        lines.append(f'| {row["schedule"]}/{row["precision"]}/{row["method"]}×{row["scale"]:g} | {values} |')
    lines += ['', '## 评测边界与原始记录', '',
              'Recall/F1：材料包同一评测器，tracks，conf≥0.25、IoU≥0.5；GT=4400，按原规则忽略 crowd/distractor，保留完整 Kalman 框。未计算 IDF1/HOTA，不能仅据 F1 宣称轨迹 ID 精度不变。',
              '缩放后框坐标、目标位移和背景平移均按实际 X/Y 比例还原至原视频坐标。特征间距、窗口、前后向误差阈值仍使用跟踪图像像素，缩放会改变跟踪行为。',
              '模型未修改；本次没有重新计算模型 mAP。动态调度的检测帧可能变化，固定调度用于隔离这种影响。', '',
              'CPU/NPU 保持默认 ondemand 策略，没有锁频或绑核。单次 Flow 与 FPS 见下表；重复运行有波动，不能将跨时段差异完全归因于 RGA。后续补充运行保留了 CPU/NPU 频率采样，早期主对照没有频率时间序列。', '',
              '| 运行 | Flow ms/帧 | FPS | F1 | 检测次数 | 跳过提交 | 队列上限 | 历史上限 | EOF 待处理 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        s = run['summary']
        lines.append(f'| {run["name"]} | {s["timing"]["flow"]["mean_ms"]:.3f} | {s["processing_fps"]:.2f} | {run["quality"]["operating_point"]["f1"]:.2%} | {s["inference_count"]} | {s["skipped_detections"]} | {s["max_pending"]} | {s["history_peak"]} | {s["pending_at_end"]} |')
    lines += ['', '## 异步结果回传帧间隔', '',
              '间隔为结果应用帧号减去推理源帧号。缩小跟踪图后处理帧速提高，单次推理可能跨过更多源帧；同时检测时机、LK 点和误差阈值的物理尺度也在变化，不能仅凭该实验确定 F1 变化的单一原因。', '',
              '| 运行 | 校正次数 | 平均源帧间隔 | 最大源帧间隔 | 丢弃结果 |', '|---|---:|---:|---:|---:|']
    for run in runs:
        if run['schedule'] != 'dynamic':
            continue
        ages = []
        with Path(run['output']).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['type'] == 'frame' and row['corrected_from_frame_index'] >= 0:
                    ages.append(row['frame_index'] - row['corrected_from_frame_index'])
        assert len(ages) == run['summary']['async_corrections'] and all(0 <= age <= 64 for age in ages)
        run['correction_age_frames'] = {'count':len(ages), 'mean':statistics.mean(ages) if ages else 0, 'max':max(ages,default=0)}
        lines.append(f'| {run["name"]} | {len(ages)} | {statistics.mean(ages) if ages else 0:.2f} | {max(ages,default=0)} | {run["summary"]["discarded_results"]} |')
    fixed = [r for r in runs if r['schedule'] == 'fixed']
    for precision in {r['precision'] for r in fixed}:
        reference = None
        for run in fixed:
            if run['precision'] != precision:
                continue
            detections = []
            for line in Path(run['output']).read_text().splitlines():
                frame = json.loads(line)
                if frame['type'] == 'frame' and frame['is_detection_frame']:
                    detections.append((frame['frame_index'], frame['detections']))
            assert len(detections) == 152 and [i for i, _ in detections] == list(range(0,1210,8))
            if reference is None:
                reference = detections
            else:
                assert reference == detections, 'fixed detector outputs changed between LK cases'
    lines += ['', '固定调度各配置的 152 个检测源帧及原始检测框逐字段一致；差异来自 LK 图像与后续跟踪。', '',
              '完整原始 JSONL、配置、stderr、50ms RSS/温度采样、哈希见同目录 *.measurement.json 与 report.json。', '',
              '```bash', 'cd /home/yolo/test', 'cmake --build --preset rk3588-release',
              f'/usr/bin/python3 tools/compare_lk_rga.py --dataset {pack} --output-dir validation/lk_rga_NEW --repeats 2', '```', '']
    write_json(work / 'report.json', {'aggregates': aggregates, 'runs': runs, 'dataset': str(pack),
                                    'binary_sha256': binary_hash, 'regenerated_from': provenance})
    (work / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:20]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--schedule', choices=('fixed', 'dynamic'), default='fixed')
    parser.add_argument('--variants', nargs='+', default=['rga:1', 'rga:0.5'],
                        metavar='rga:SCALE', help='RGA grayscale scales in [0.25, 1]')
    parser.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--from-dir', type=Path)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--merge', nargs='+', type=Path, help='Merge completed report directories without running hardware')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not (args.merge or args.report_only):
        for variant in args.variants:
            method, _, scale = variant.partition(':')
            try:
                if method != 'rga' or not 0.25 <= float(scale) <= 1:
                    raise ValueError
            except ValueError:
                parser.error(f'unsupported live variant {variant!r}: use rga:SCALE in [0.25, 1]')
    work, pack = args.output_dir.resolve(), args.dataset.resolve()
    work.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTHONHOME'):
        environment.pop(key, None)
    environment['PATH'] = '/usr/bin:/bin'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    checkpoint = work / 'experiment.json'
    if args.merge:
        runs = []
        seen = set()
        for directory in args.merge:
            saved = load_history(directory / 'report.json')
            if saved['dataset'] != str(pack):
                raise ValueError('Cannot merge different datasets')
            for run in saved['runs']:
                if run['name'] in seen or sha(run['output']) != run['prediction_sha256'] or sha(run['config']) != run['config_sha256']:
                    raise ValueError('Duplicate run or changed artifact')
                seen.add(run['name'])
                runs.append(run)
        render(runs,work,pack, provenance=[str(d.resolve()) for d in args.merge])
        return
    if args.report_only:
        if not args.from_dir:
            parser.error('--report-only requires --from-dir')
        runs = load_history(args.from_dir / 'experiment.json')
    else:
        if checkpoint.exists():
            parser.error('output directory already contains experiment.json; use a new directory or --report-only')
        runs = []
        for repeat in range(args.repeats):
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                method, scale = variant.split(':')
                for precision in ('fp16','int8') if repeat % 2 == 0 else ('int8','fp16'):
                    case = dict(name=f'{args.schedule}_{precision}_{method}_{scale.replace(".","p")}_{repeat+1}', precision=precision,
                                model=str(SOURCE.parent/'yolo_model_rknn'/f'yolo26s_800x608_{precision}.rknn'),
                                mode='pipeline', preprocess='rga', lk_preprocess=method, lk_scale=float(scale),
                                schedule=args.schedule, **{'async': args.schedule == 'dynamic'})
                    runs.append(observed_run(args.build_dir.resolve()/'benchmark_video',pack,work,case,environment))
                    write_json(checkpoint,runs)
    # Only score labels once all performance measurements have finished.
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(pack/'tools'))
    evaluator = importlib.import_module('compare')
    evaluator.ROOT = pack
    cache = accelerate_evaluator(evaluator)
    manifest, labels, crowds = evaluator.load_context()
    for run in runs:
        run['quality'] = pipeline_quality(run['output'],manifest,labels,crowds,evaluator)
        cache.cache_clear()
    render(runs,work,pack, None if args.report_only else sha(args.build_dir.resolve()/'benchmark_video'),
           str(args.from_dir.resolve()) if args.report_only else None)


if __name__ == '__main__':
    main()

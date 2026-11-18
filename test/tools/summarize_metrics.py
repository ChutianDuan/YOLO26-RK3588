#!/usr/bin/python3
"""Summarize recorded experiments without opening devices or rewriting evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from paths import TEST_ROOT, WORKSPACE


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


def collect():
    sources = {}

    def read(name):
        path = TEST_ROOT / name
        raw = path.read_bytes()
        sources[name] = {'path': str(path.relative_to(WORKSPACE)), 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)

    yuyv = read('validation/camera_20261018/soak3000/summary.json')
    raw_path = TEST_ROOT / 'validation/camera_20261018/soak3000/results.jsonl'
    latencies = []
    with raw_path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['type'] == 'frame':
                latencies.append(item['latency_ms'])
    with raw_path.open('rb') as stream:
        sources['yuyv_frames'] = {'path': str(raw_path.relative_to(WORKSPACE)),
                                 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    yuyv['capture_to_result_percentiles_ms'] = {key: percentile(latencies, q) for key, q in
                                               (('p50', .5), ('p95', .95), ('p99', .99))}
    live = read('validation/camera_live_20261018/experiment/report.json')
    comparison = read('validation/bdd_compare_20261018/report.json')
    lk = read('validation/lk_rga_20261018/report.json')
    dma = read('validation/rknn_dma_input/report.json')
    detector = {'host_fp32_opencv': comparison['fp32_reference'], **comparison['detection_quality']}
    compact = {}
    for name, value in detector.items():
        compact[name] = {k: value[k] for k in ('map50', 'map50_95', 'operating_point', 'per_class')}
    return {'schema_version': 1, 'kind': 'historical_evidence_summary', 'date_shanghai': '2026-10-18',
            'units': {'time': 'ms', 'rss': 'MiB (including historical *_mb field names)',
                      'cpu': 'one core = 100%', 'npu': 'mean of driver snapshots, not time-integrated utilization'},
            'sources': sources, 'yuyv_3000': yuyv, 'mjpeg_live': live,
            'model_quality': compact, 'historical_standard_input_pipeline': comparison['aggregates'],
            'lk_aggregates': lk['aggregates'],
            'dma_soak': {k: dma[k] for k in ('summary', 'steady_early_mean_mb', 'steady_late_mean_mb',
                        'steady_growth_mb', 'memory_pass', 'steady_early_peak_fds', 'steady_late_peak_fds', 'fd_pass')},
            'boundary': 'Historical runs from different revisions; no new inference or accuracy measurement is performed by this script.'}


def render(data, destination=None):
    destination = destination or TEST_ROOT / "reports"
    lines = ['# 已有实测指标汇总', '',
             '日期：2026-10-18（Asia/Shanghai）。本文件由 `test/tools/summarize_metrics.py` 从原始机器记录生成；未重新执行推理。', '',
             'CPU 一个核为 100%，RSS 使用 MiB；NPU 为驱动快照采样均值。不同阶段、版本和输入分组展示，不能拼接成一次实验。', '',
             '## YUYV：800×600、15 FPS、INT8、单 Context', '']
    y = data['yuyv_3000']
    lines += ['| 指标 | 值 |', '|---|---:|', f"| 输出帧 / 推理次数 | {y['frame_count']} / {y['inference_count']} |",
              f"| 时长 s / 处理 FPS | {y['elapsed_ms']/1000:.3f} / {y['processing_fps']:.4f} |",
              f"| RGA / RKNN / 后处理均值 ms | {y['timing']['preprocess']['mean_ms']:.3f} / {y['timing']['infer']['mean_ms']:.3f} / {y['timing']['postprocess']['mean_ms']:.3f} |",
              f"| 采集到结果 P50 / P95 / P99 ms | " + ' / '.join(f'{v:.3f}' for v in y['capture_to_result_percentiles_ms'].values()) + ' |',
              f"| 稳态应用层上传 B / 历史启动参考上传 B | {y['application_upload_bytes']} / {y['validation_upload_bytes']} |",
              f"| CPU % / 采样 RSS 首尾 MiB | {y['cpu_percent']:.2f} / {y['resources']['first_rss_mb']:.2f} → {y['resources']['last_rss_mb']:.2f} |",
              f"| FD 最小/最大 / 温度 °C | {y['resources']['min_fd']}/{y['resources']['max_fd']} / {y['resources']['max_temperature_c']:.3f} |",
              f"| 主动丢旧帧 / 驱动序号缺口 / 损坏帧 | {y['stale_dropped']} / {y['driver_sequence_gaps']} / {y['corrupt_dropped']} |", '',
              '本组为迁移前含启动像素校验和 checksum 的历史版本。采集时间戳到结果的延迟不等于曝光到显示延迟。', '',
              '## MJPEG：800×600、30 FPS、MPP、INT8、无屏画框', '',
              '| 运行 | 输出/检测 FPS | 输出 P95 ms | 校正年龄 P95 ms | CPU % | RSS 峰值 MiB | 温度 °C |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in data['mjpeg_live']['records']:
        s = r['summary']; t = s['timing']; resources = s['resources']
        lines.append(f"| {r['name']} | {s['output_fps']:.2f}/{s['inference_fps']:.2f} | {t['output_latency']['p95_ms']:.0f} | {t['correction_age']['p95_ms']:.0f} | {s['cpu_percent']:.1f} | {resources['max_rss_mb']:.2f} | {resources['max_temperature_c']:.2f} |")
    lines += ['', '| 摄像头模式 | NPU Core0/1/2 采样均值 % | 包/检测/输出槽丢旧帧 | 结果/JSON 队列峰值 | 驱动缺口 |',
              '|---|---:|---:|---:|---:|']
    for r in data['mjpeg_live']['records']:
        if not r['name'].startswith('camera-'):
            continue
        s=r['summary']; n=r['npu_load_percent_sample_mean']
        lines.append(f"| {r['name']} | " + '/'.join(f'{n[str(i)]:.1f}' for i in range(3)) +
                     f" | {s['packet_pending_dropped']}/{s['inference_pending_dropped']}/{s['output_pending_dropped']} | {s['result_queue_peak']}/{s['json_queue_peak']} | {s['source']['driver_sequence_gaps']} |")
    lines += ['', '输出延迟到无屏画框完成；校正年龄描述输出所用检测数据的新鲜程度。900 帧同夹具回放的校正年龄 P95 为 default 105、all 136、split 73 ms，选择 split。三个真实摄像头运行各 3000 帧。', '',
              '## 模型精度：BDD100K 单条训练序列', '',
              '1210 源帧、202 标注时刻、普通真值 4400（人体 2587）。AP 对有普通真值的六类取平均；候选下限 0.001，工作点 conf=0.25/IoU=0.5。FP32 来自另一台主机。', '',
              '| 模型/预处理 | mAP50 | mAP50–95 | P@0.25 | R@0.25 | F1@0.25 | 人体 AP50 | 人体 F1@0.25 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,q in data['model_quality'].items():
        op=q['operating_point']; person=q['per_class']['person']
        values=[q['map50'],q['map50_95'],op['precision'],op['recall'],op['f1'],person['ap50'],person['operating_point']['f1']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2%}' for v in values)+' |')
    lines += ['', '该组板端数据来自外部 FD 输入改造之前的标准输入版本；保留为历史精度证据，不标作当前代码重新测得的 mAP。不是十类独立测试集或官方 COCO 全套评测。逐类 AP、P/R/F1 和计数见 metrics.json。', '',
              '## 历史文件流水线与 LK', '',
              '| 模型 | 推理均值 ms | Flow ms/解码帧 | 处理 FPS | 平均 RSS MiB | 跟踪 Recall / F1 |', '|---|---:|---:|---:|---:|---:|']
    for name,q in data['historical_standard_input_pipeline'].items():
        lines.append(f"| {name} | {q['stages_mean_ms']['infer']:.3f} | {q['stages_mean_ms']['flow']:.3f} | {q['processing_fps']:.2f} | {q['mean_rss_mib']:.2f} | {q['recall']:.2%}/{q['f1']:.2%} |")
    lines += ['', '上表为旧标准输入、RGA 检测预处理、4 FPS 目标检测、异步动态调度，各模型三次运行。推理阶段含输入提交和输出获取；处理 FPS 含 LK 帧。', '',
              '| LK 调度/模型/来源/比例 | 重复次数 | Flow ms/帧 | FPS | F1 | RSS MiB |', '|---|---:|---:|---:|---:|---:|']
    for q in data['lk_aggregates']:
        lines.append(f"| {q['schedule']}/{q['precision']}/{q['method']}/{q['scale']:g} | {q['repeats']} | {q['flow_ms']:.3f} | {q['fps']:.2f} | {q['f1']:.2%} | {q['rss_mib']:.2f} |")
    lines += ['', 'RGA 只负责灰度准备，LK 特征、金字塔和双向光流仍在 CPU。固定调度各 152 次相同源帧检测；异步检测时机不同。缩图收益不能直接归因于 RGA 算法加速。', '',
              '## 实时跟踪：JPEG 拉伸 800×600 后还原坐标评分', '',
              '| 模式 | IDF1 | MOTA | ID 切换 | 匹配 IoU | 中心误差均值/P95 px | 标注帧覆盖 |', '|---|---:|---:|---:|---:|---:|---:|']
    for mode,q in data['mjpeg_live']['quality'].items():
        lines.append(f"| {mode} | {q['idf1']:.4f} | {q['mota']:.4f} | {q['id_switches']} | {q['mean_matched_iou']:.4f} | {q['mean_matched_center_error_px']:.2f}/{q['p95_matched_center_error_px']:.2f} | {q['output_labeled_frames']}/{q['labeled_frames']} |")
    d=data['dma_soak']; s=d['summary']
    lines += ['', '该组预处理和调度与 LK 离线实验不同；中心误差仅统计已匹配框。当前 IDF1 较低、MOTA 为负，仍需改善模型精度与跟踪，不能作为工业精度达标证据。', '',
              '## 外部 FD 输入后的合成视频资源检查', '',
              f"3000 帧 320×240 合成视频、FP16、RGA＋LK：输出 {s['frame_count']} 帧、推理 {s['inference_count']} 次、处理 {s['processing_fps']:.3f} FPS。",
              f"稳态 RSS {d['steady_early_mean_mb']:.3f} → {d['steady_late_mean_mb']:.3f} MiB，变化 {d['steady_growth_mb']:.3f} MiB；FD 峰值 {d['steady_early_peak_fds']} → {d['steady_late_peak_fds']}。",
              '此 FPS 包含 LK 帧，只说明本次输入和时长下的资源表现；未证明无限时长稳定性。', '',
              '## 数据来源与校验', '', '| 来源 | SHA-256 |', '|---|---|']
    for q in data['sources'].values():
        path=q['path']; link=os.path.relpath(WORKSPACE / path, destination)
        lines.append(f"| [{path}]({link}) | `{q['sha256']}` |")
    guide = os.path.relpath(TEST_ROOT / 'README.md', destination)
    index = os.path.relpath(TEST_ROOT / 'reports/README.md', destination)
    lines += ['', f'历史报告的旧绝对路径保持原文，工具通过 `paths.py` 在读取时映射；重跑见 [测试指南]({guide})，各版本证据见 [报告索引]({index})。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=TEST_ROOT / 'reports')
    args = parser.parse_args()
    # Outputs are generated summaries only. Never write within historical evidence.
    destination = args.output_dir.resolve()
    protected = [TEST_ROOT / 'validation', TEST_ROOT / 'camera', TEST_ROOT / 'archive', TEST_ROOT / 'rknn_video_compare_20261018_002d290d']
    if any(destination == p or p in destination.parents for p in protected):
        parser.error('output directory must be outside archived evidence')
    data = collect()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'metrics.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (destination / 'metrics.md').write_text(render(data, destination))
    print(destination / 'metrics.md')


if __name__ == '__main__':
    main()

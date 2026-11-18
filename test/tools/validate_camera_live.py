#!/usr/bin/python3
"""Serial native RK3588 replay/soak and BDD tracking validation. No tuning/install."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

from paths import SOURCE, DEFAULT_BUILD, DEFAULT_DATASET

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def npu_average(resources):
    values = {str(core): [] for core in range(3)}
    for row in resources:
        for core, percent in re.findall(r'Core([012]):\s*(\d+)%', row.get('npu_load','')):
            values[core].append(int(percent))
    return {core: statistics.mean(samples) if samples else None for core,samples in values.items()}

def run(binary, work, name, args, env):
    output, log = work / (name + '.jsonl'), work / (name + '.log')
    # Scoring uses source-image coordinates; keep the baseline full field of view.
    command = [str(binary), '--preview', 'off', '--stabilize', 'off', '--output', str(output), *map(str, args)]
    print('RUN', name, flush=True)
    with log.open('x') as f:
        p = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300)
    if p.returncode:
        raise RuntimeError(f'{name} failed ({p.returncode}); see {log}')
    rows = read(output)
    assert rows[-1]['type'] == 'summary' and rows[-1]['status'] == 'completed'
    frames = [r for r in rows if r['type'] == 'frame']
    detections = [r for r in rows if r['type'] == 'detection']
    resources = [r for r in rows if r['type'] == 'resource']
    assert len(frames) == rows[-1]['frame_count'] and frames
    assert all(a['sequence'] < b['sequence'] for a,b in zip(frames,frames[1:]))
    assert all(a['sequence'] < b['sequence'] for a,b in zip(detections,detections[1:]))
    assert all(r['correction_timestamp_ms'] is None or r['correction_timestamp_ms'] <= r['timestamp_ms'] for r in frames)
    assert all(t['last_match_age_ms'] <= 500 for r in frames for t in r['tracks'])
    steady = [r for r in resources if r['frame_count'] >= 150]
    summary = rows[-1]
    checks = {
        'bounded_queues': summary['result_queue_peak'] <= 3 and summary['json_queue_peak'] <= 64,
        'no_decoded_pool_exhaustion': summary['decoded_pool_dropped'] == 0,
        'no_timestamp_drops': summary['source'].get('timestamp_dropped',0) == 0,
        'fd_stable_after_warmup': not steady or max(r['fd_count'] for r in steady) == min(r['fd_count'] for r in steady),
        'rss_growth_under_4mb_after_warmup': not steady or steady[-1]['rss_mb'] - steady[0]['rss_mb'] < 4,
    }
    record = dict(name=name, command=command, output=str(output), summary=summary, checks=checks,
                  npu_load_percent_sample_mean=npu_average(resources))
    save(work / (name + '.measurement.json'), record)
    print('DONE', name, 'frames',len(frames),'output_fps',round(summary['output_fps'],2),
          'inference_fps',round(summary['inference_fps'],2),'correction_p95_ms',summary['timing']['correction_age']['p95_ms'],flush=True)
    if not all(checks.values()):
        raise RuntimeError(f'{name} checks failed: {checks}')
    return record

def quality(output, pack):
    sys.path.insert(0, str(pack / 'tools'))
    sys.dont_write_bytecode = True
    compare = importlib.import_module('compare')
    evaluator = importlib.import_module('evaluate_bdd100k_tracking')
    compare.ROOT = pack
    manifest, labels, crowds = compare.load_context()
    predictions = {index: [] for index in labels}
    present = set()
    for row in read(output):
        if row.get('type') != 'frame':
            continue
        index = row['sequence']
        if index not in labels:
            continue
        present.add(index)
        for item in row['tracks']:
            cls = item['class_id']
            if cls not in manifest['annotated_class_ids'] or item['score'] < .25:
                continue
            b = item['bbox']
            box = (b[0]*1280/800,b[1]*720/600,b[2]*1280/800,b[3]*720/600)
            predictions[index].append(evaluator.ObjectRecord(index,str(item['track_id']),manifest['classes'][cls],box,item['score']))
    candidates, ignored = compare.suppress_ignored(predictions, labels, crowds, .5)
    metrics = evaluator.tracking_metrics(labels,candidates,.5,manifest['video']['fps']/6)
    errors = []
    for index, objects in labels.items():
        for a,b,iou in evaluator.match_frame(objects,candidates.get(index,[]),.5):
            gt, pred = objects[a].box,candidates[index][b].box
            errors.append((((gt[0]+gt[2]-pred[0]-pred[2])/2)**2 + ((gt[1]+gt[3]-pred[1]-pred[3])/2)**2)**.5)
    metrics.update(labeled_frames=len(labels), output_labeled_frames=len(present),
        missing_labeled_frames_scored_as_empty=len(labels)-len(present), ignored_predictions=ignored,
        mean_matched_center_error_px=statistics.mean(errors) if errors else None,
        p95_matched_center_error_px=evaluator.percentile(errors,.95),
        geometry='JPEG stretched to 800x600; predictions restored to original 1280x720 for GT',
        boundary='single existing BDD training sequence; not an independent industrial accuracy evaluation')
    return metrics

def render(work, records, selected, qualities):
    rows = ['# camera_live 板端验证', '', '时间：'+datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(), '',
        'V4L2 800×600 MJPEG，请求 30 FPS；MPP→NV12→RGA→YOLO26 INT8；无屏输出仍执行 BGR 转换与画框。',
        '防抖关闭以保持原图坐标。实采 MJPEG 轮换 DMA-BUF，复制量见各次运行的 compressed_copy_bytes；延迟统计到板端输出。', '',
        '| 运行 | 输出帧 | 源 FPS | 输出 FPS | 检测 FPS | 输出延迟 P95 ms | 校正数据年龄 P95 ms | CPU % | 峰值 RSS MiB | 温度 °C |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        s = record['summary']
        rows.append(f'| {record["name"]} | {s["frame_count"]} | {s["source"]["observed_source_fps"]:.2f} | {s["output_fps"]:.2f} | '
            f'{s["inference_fps"]:.2f} | {s["timing"]["output_latency"]["p95_ms"]:.0f} | {s["timing"]["correction_age"]["p95_ms"]:.0f} | '
            f'{s["cpu_percent"]:.1f} | {s["resources"]["max_rss_mb"]:.2f} | {s["resources"]["max_temperature_c"]:.2f} |')
    rows += ['', '| 摄像头运行 | NPU Core0 / Core1 / Core2 采样均值 % |', '|---|---:|']
    for record in records:
        if record['name'].startswith('camera-'):
            averages = record.get('npu_load_percent_sample_mean',{})
            values = ' / '.join(f'{averages[str(i)]:.1f}' if averages.get(str(i)) is not None else 'n/a' for i in range(3))
            rows.append(f'| {record["name"]} | {values} |')
    rows += ['', 'NPU 值来自每 30 帧读取的驱动负载快照，以上为这些样本的均值，不是对整个运行区间的精确积分。',
        '', f'推荐默认模式：`{selected}`。选择依据为同一夹具实时回放的校正数据年龄 P95；差距不足 5% 时保留 Context 较少的模式。',
        '分位数为固定直方图统计，分辨率 1 ms；≥4096 ms 进入溢出桶，最大值另行记录。', '',
        '## 交通视频预测框与 ID', '', '| 模式 | IDF1 | MOTA | ID 切换 | 平均匹配 IoU | 中心误差均值 / P95 px | 标注帧输出 / 总数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,q in qualities.items():
        rows.append(f'| {mode} | {q["idf1"]:.4f} | {q["mota"]:.4f} | {q["id_switches"]} | {q["mean_matched_iou"]:.4f} | '
            f'{q["mean_matched_center_error_px"]:.2f} / {q["p95_matched_center_error_px"]:.2f} | {q["output_labeled_frames"]} / {q["labeled_frames"]} |')
    rows += ['', 'BDD100K 现有训练序列，1210 帧、202 个标注时刻。JPEG 拉伸至 800×600，评分前将框还原至 1280×720；'
        '按既有评测器忽略 crowd/distractor，IoU≥0.5，缺失输出帧按空预测评分。误差仅统计已匹配目标，不代表总体定位误差。',
        '该预处理和实时调度与既有 LK 离线基线不同，不能将指标差异完全归因于 Kalman 或 NPU 模式。', '',
        '## 验证边界', '', '每种 NPU 模式真实摄像头运行 3000 个输出帧；源≥29.5 FPS 时要求输出≥29 FPS。'
        '源帧率不足则如实记录，未修改曝光、锁频、驱动或系统配置。FD/RSS 检查从第 150 个输出帧开始。',
        '实际显示、物理拔插、驱动卡死恢复、自动重连、多路和工业场景精度仍未验证。', '',
        '逐帧数据、命令、队列/丢帧/复制字节、阶段时延、资源采样和检查见同目录 JSONL、measurement.json 与 report.json。', '']
    (work/'report.md').write_text('\n'.join(rows))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir', type=Path, default=DEFAULT_BUILD)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--traffic-fixture', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    p.add_argument('--soak-frames', type=int, default=3000)
    args = p.parse_args()
    work = args.output_dir.resolve(); work.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ('LD_LIBRARY_PATH','LD_PRELOAD'):
        env.pop(key,None)
    binary = args.build_dir.resolve()/'camera_live'
    environment = {'date_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'platform':platform.platform(),'binary_sha256':digest(binary),'fixture_sha256':digest(args.fixture),
        'model_sha256':digest(SOURCE.parent/'yolo_model_rknn/yolo26s_800x608_int8.rknn'),
        'mpp_sha256':digest(Path('/opt/rockchip/mpp/lib/librockchip_mpp.so')),
        'rga_sha256':digest(Path('/opt/rockchip/librga/lib/librga.so')),
        'rknn_sha256':digest(Path('/opt/rknn/runtime-2.5.0/lib/librknnrt.so'))}
    save(work/'environment.json',environment)
    with (work/'npu-equivalence.log').open('x') as f:
        subprocess.run([str(args.build_dir.resolve()/'live_npu_test')],stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=90)
    records=[]; replay=[]
    for mode in ('default','all','split'):
        record = run(binary,work,'replay-'+mode,['--replay',args.fixture.resolve(),'--npu-mode',mode],env)
        records.append(record); replay.append(record)
    best = min(replay,key=lambda r:r['summary']['timing']['correction_age']['p95_ms'])
    best_p95 = best['summary']['timing']['correction_age']['p95_ms']
    eligible = [r for r in replay if r['summary']['timing']['correction_age']['p95_ms'] < best_p95 / .95]
    chosen = min(eligible,key=lambda r: (3 if r['summary']['npu_mode']=='split' else 1,
        r['summary']['timing']['correction_age']['p95_ms'],r['summary']['npu_mode']!='default'))
    selected = chosen['summary']['npu_mode']
    qualities={}
    for mode in dict.fromkeys(('default',selected)):
        record=run(binary,work,'traffic-'+mode,['--replay',args.traffic_fixture.resolve(),'--npu-mode',mode],env)
        records.append(record); qualities[mode]=quality(Path(record['output']),args.dataset.resolve())
        save(work/('quality-'+mode+'.json'),qualities[mode])
    for mode in ('default','all','split'):
        record=run(binary,work,'camera-'+mode,['--npu-mode',mode,'--max-frames',args.soak_frames],env)
        assert record['summary']['frame_count']==args.soak_frames
        source_fps=record['summary']['source']['observed_source_fps']
        record['performance_status']='pass' if source_fps>=29.5 and record['summary']['output_fps']>=29 else (
            'source_below_29.5_fps' if source_fps<29.5 else 'output_below_29_fps')
        records.append(record)
    save(work/'report.json',dict(selected_npu_mode=selected,records=records,quality=qualities,environment=environment))
    render(work,records,selected,qualities)
    if any(r.get('performance_status') == 'output_below_29_fps' for r in records):
        raise RuntimeError('output below 29 FPS with sufficient source FPS; see report.json')
    print('SELECTED',selected,flush=True)

if __name__=='__main__':
    main()

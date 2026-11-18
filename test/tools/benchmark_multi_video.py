#!/usr/bin/env python3
"""Serial hardware runs: three/four channels, recording cost and ten-minute soak."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def distribution(values):
    values = sorted(v for v in values if isinstance(v,(int,float)) and math.isfinite(v))
    if not values:
        return None
    return dict(count=len(values), mean=statistics.mean(values), min=values[0], max=values[-1],
                **{name: values[min(len(values)-1,math.ceil(len(values)*p)-1)] for name,p in [('p50',.5),('p95',.95),('p99',.99)]})


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def analyze(output, seconds):
    summary = json.loads((output/'summary.json').read_text())
    metadata = json.loads((output/'metadata.json').read_text())
    resource = list(rows(output/'resources.jsonl'))
    steady = [r for r in resource if 3000 <= r['elapsed_ms'] <= seconds*1000-1000]
    result = dict(summary=summary, steady={}, resources={})
    if len(steady) >= 2:
        first,last = steady[0],steady[-1]
        span = (last['elapsed_ms']-first['elapsed_ms'])/1000
        rates = [(b['inference_count']-a['inference_count'])/span for a,b in zip(first['channels'],last['channels'])]
        result['steady'].update(detection_fps_by_channel=rates, total_detection_fps=sum(rates),
                               min_max_fairness=min(rates)/max(rates) if max(rates) else 0,
                               span_seconds=span,
                               output_fps_by_channel=[(b['output_frames']-a['output_frames'])/span for a,b in zip(first['channels'],last['channels'])])
    epoch = metadata['epoch_monotonic_ms']
    metrics = []
    for channel in summary['channels']:
        timings = {key:[] for key in ['queue_wait_ms','preprocess_ms','infer_ms','postprocess_ms','source_to_result_ms','output_latency_ms','result_age_ms']}
        for row in rows(output/f'stream_{channel["channel_id"]}'/'results.jsonl'):
            time_ms = row.get('completed_monotonic_ms',row.get('emitted_monotonic_ms',0))-epoch
            if not 3000 <= time_ms <= seconds*1000-1000:
                continue
            if row['type']=='detection':
                for key in timings:
                    if key in row:
                        timings[key].append(row[key])
            elif row['type']=='frame':
                timings['output_latency_ms'].append(row['emitted_monotonic_ms']-epoch-row['media_timestamp_ms'])
                if row.get('detection_age_ms') is not None:
                    timings['result_age_ms'].append(row['detection_age_ms'])
        metrics.append({key:distribution(values) for key,values in timings.items()})
    result['steady']['timing_by_channel'] = metrics
    for key in ['cpu_percent','rss_mb','pss_mb','fd_count','npu_frequency_hz']:
        result['resources'][key] = distribution([r.get(key) for r in steady])
    result['resources']['npu_load_by_core'] = [distribution([r['npu_load_percent'][i] for r in steady if r.get('npu_load_percent') is not None]) for i in range(3)]
    names = set().union(*(r['temperature_c'] for r in steady))
    result['resources']['temperature_c'] = {name:distribution([r['temperature_c'].get(name) for r in steady]) for name in sorted(names)}
    result['per_minute'] = []
    for minute in range(math.ceil(seconds/60)):
        samples = [r for r in resource if minute*60000 <= r['elapsed_ms'] < (minute+1)*60000]
        if len(samples) < 2:
            continue
        a,b = samples[0],samples[-1]
        span = (b['elapsed_ms']-a['elapsed_ms'])/1000
        rates = [(y['inference_count']-x['inference_count'])/span for x,y in zip(a['channels'],b['channels'])]
        result['per_minute'].append(dict(minute=minute+1,detection_fps=rates,
            rss_mb=distribution([r['rss_mb'] for r in samples]),fd_count=distribution([r['fd_count'] for r in samples])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--binary',type=Path,default=ROOT/'test/build-rk3588/bin/yolo_multi')
    parser.add_argument('--input',type=Path,default=ROOT/'test/rknn_video_compare_20261018_002d290d/video/002d290d-90f2bab2.mov')
    parser.add_argument('--config',type=Path,default=ROOT/'yolo_run/config/video_multi.yaml')
    parser.add_argument('--short-seconds',type=float,default=20)
    parser.add_argument('--soak-seconds',type=float,default=600)
    parser.add_argument('--detect-fps',type=float,default=0)
    parser.add_argument('--track-max-age-ms',type=float,default=500)
    args = parser.parse_args()
    if args.short_seconds < 8 or args.soak_seconds < 8:
        parser.error('runs must be at least 8 seconds')
    output = args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ)
    for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:
        env.pop(key,None)
    long_input = output/'soak_input.mov'
    ffmpeg = ['/usr/bin/ffmpeg','-v','error','-nostdin','-stream_loop','-1','-i',str(args.input),
              '-t',str(max(args.short_seconds,args.soak_seconds)+5),'-map','0:v:0','-c','copy',str(long_input)]
    with (output/'fixture.log').open('w') as log:
        subprocess.run(ffmpeg,check=True,stdout=log,stderr=log,env=env,timeout=120)
    files = [args.binary,args.input,args.config,ROOT/'yolo_model_rknn/yolo26s_800x608_int8.rknn',
             Path('/opt/rknn/runtime-2.5.0/lib/librknnrt.so'),Path('/opt/rockchip/librga/lib/librga.so')]
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),host=os.uname()._asdict() if hasattr(os.uname(),'_asdict') else list(os.uname()),
                    hashes={str(p.resolve()):digest(p) for p in files},fixture_command=ffmpeg,
                    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'yolo_run/video').glob('multi*') if p.is_file()})
    patterns = ['yolo_run/video/multi*','yolo_run/video/mjpeg_recorder.*','yolo_run/video/fair_scheduler.h',
                'yolo_run/image/image_processing.*','yolo_run/config/video_multi.yaml',
                'yolo_run/config/app_config.*','yolo_run/model/yolo_engine.*','yolo_run/tracking/byte_tracker.*',
                'yolo_run/CMakeLists.txt','test/CMakeLists.txt','test/cpp/multi*','test/tools/multi_reference.cpp',
                'test/python/multi_video_test.py','test/support/multi_fault_main.cpp','test/tools/benchmark_multi_video.py']
    source_files = sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()})
    manifest['source_snapshot_sha256'] = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        destination = output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        manifest['source_snapshot_sha256'][str(relative)] = digest(path)
    manifest['compiler_version'] = subprocess.check_output(['/usr/bin/g++','--version'],text=True).splitlines()[0]
    for build_commands in [args.binary.parent/'compile_commands.json', args.binary.parent.parent/'compile_commands.json']:
        if build_commands.exists():
            shutil.copy2(build_commands,output/'compile_commands.json')
            break
    (output/'linked_libraries.txt').write_text(subprocess.check_output(['/usr/bin/ldd',str(args.binary.resolve())],text=True))
    save(output/'manifest.json',manifest)
    report = dict(runs={},hardware_runs_serial=True,steady_excludes_first_seconds=3,
                  tracking='bytetrack_native_kalman',detect_fps_limit=args.detect_fps,
                  track_max_age_ms=args.track_max_age_ms,
                  scope='OpenCV CPU file decode -> BGR upload -> RGA -> three independent RKNN contexts; no end-to-end zero-copy claim')
    cases = [(f'{channels}ch_record_{record}',channels,record,args.short_seconds) for channels in [3,4] for record in ['off','on']]
    cases.append(('4ch_soak',4,'off',args.soak_seconds))
    for name,channels,record,seconds in cases:
        destination = output/name
        cmd = [str(args.binary.resolve()),'--config',str(args.config.resolve()),'--output-dir',str(destination),
               '--preview','off','--record',record,'--duration-seconds',str(seconds),
               '--detect-fps',str(args.detect_fps),'--track-max-age-ms',str(args.track_max_age_ms)]
        for _ in range(channels):
            cmd += ['--input',str(long_input)]
        print(f'Start {name}: {channels} channels, record={record}, {seconds}s',flush=True)
        save(output/f'{name}.command.json',cmd)
        with (output/f'{name}.log').open('w') as log:
            run = subprocess.run(cmd,stdout=log,stderr=log,env=env,timeout=seconds+90)
        result = analyze(destination,seconds)
        result.update(exit_code=run.returncode,record=record,channels=channels,seconds=seconds)
        result['passed'] = run.returncode==0 and all(c['inference_count']>0 and c['event_dropped']==0 and not c['scheduler']['busy'] and not c['scheduler']['pending'] for c in result['summary']['channels'])
        report['runs'][name] = result
        save(output/'report.json',report)
        print(f'Finished {name}: status={result["summary"]["status"]}, steady FPS={result["steady"].get("detection_fps_by_channel")}',flush=True)
    lines = ['# 四路视频 / 三个 NPU Worker 板端验证','',
             '本报告测量当前输入负载下的吞吐；软件解码、录制和动态频率均可能影响结果，不外推理论最大吞吐。', '',
             '| 运行 | 状态 | 每路稳态检测 FPS | 每路输出 FPS | 最小/最大检测 FPS | CPU均值 % | RSS峰值 MiB | 三核NPU均值 % |',
             '|---|---|---|---|---:|---:|---:|---|']
    def fmt(value):
        return '不可用' if value is None else f'{value:.2f}'
    for name,r in report['runs'].items():
        res=r['resources']; steady=r['steady']
        loads='/'.join(fmt(v['mean'] if v else None) for v in res['npu_load_by_core'])
        lines.append(f'| {name} | {r["summary"]["status"]} | '+'/'.join(map(fmt,steady.get('detection_fps_by_channel',[])))+' | '+
            '/'.join(map(fmt,steady.get('output_fps_by_channel',[])))+f' | {fmt(steady.get("min_max_fairness"))} | '+
            f'{fmt(res["cpu_percent"]["mean"] if res["cpu_percent"] else None)} | {fmt(res["rss_mb"]["max"] if res["rss_mb"] else None)} | {loads} |')
    lines += ['', '各路 P50/P95/P99、派发份额、解码/队列丢帧、每分钟内存/FD、温度和原始计数见 report.json、resources.jsonl 与各路 results.jsonl。',
              'CPU 单核=100%；RSS/PSS 指标有重叠。NPU负载为驱动采样窗口读数。关闭预览的结果不代表桌面显示已验收。',
              '输入检测覆盖率接近100%且没有待检测替换时，应视为输入受限；有替换时仍须结合解码丢帧、CPU与NPU负载判断瓶颈。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return 0 if all(r['passed'] for r in report['runs'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())

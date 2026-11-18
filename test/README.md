# 测试与实验指南

核对日期：2026-11-18（Asia/Shanghai）。测试工程直接编译当前 `../yolo_run` 源码，保留 **28 项有效回归**；旧日期的测试只要仍覆盖现有接口就继续保留。实验与长测工具不属于 CTest 自动发现范围。

## 构建与分组

```bash
cd /home/yolo
bash scripts/dev.sh test
```

日常入口构建全部测试程序，再按 `cpu` 标签串行执行 **9 项纯软件回归**。生产与测试构建都采用系统 C++17 / GCC / OpenCV、现有 Rockchip Targets、Ninja 并发 4；测试二进制在 `test/build-rk3588/bin/`。

```bash
cd /home/yolo/test
cmake --preset rk3588-release
cmake --build --preset rk3588-release
ctest --preset rk3588-release -N             # 全部 28 项
ctest --preset rk3588-release -L cpu         # 9 项，无硬件访问
ctest --preset rk3588-release -L hardware    # 19 项，真实硬件，串行
ctest --preset rk3588-release               # 完整回归，串行
```

`cpu` 测试仍会链接既有 SDK，并可能使用临时文件、软件视频编解码或模拟子进程；无需摄像头/NPU/RGA/MPP 设备。完整回归需要独占相应硬件。先检查设备占用，不为测试停止已有关键服务、修改设备权限或更换驱动。推流回归只管理自己启动的测试进程。

| CTest 名称 | 分组 | 覆盖范围 |
|---|---|---|
| `stabilization_core` | cpu | 平移、掩码、EMA、失锁/回中、时间与边界 |
| `coco_core` | cpu | COCO 配置、模型布局、TopK、坐标还原 |
| `historical_paths` | cpu | 旧证据路径映射、命令来源与原文件保留；使用临时夹具，不依赖历史运行目录 |
| `split_benchmark_accounting` | cpu | 检测计数、统计窗口、资源采样、长夹具和构建路径 |
| `live_core` | cpu | 时间、轨迹、不可变快照与预测契约 |
| `multi_scheduler` / `multi_prediction` / `multi_recorder` | cpu | 公平派发/租约、Kalman 时序、MJPG 录像/帧率/写盘错误 |
| `stream_publisher` | cpu | 发布器看门狗、参数、模拟子进程恢复 |
| `stabilization_media` | hardware | 真实 RGA ROI、stride、letterbox、灰度双槽 |
| `coco_hardware` / `coco_live` / `coco_stream` | hardware | COCO FP16 的真实 RGA/NPU 等价、MPP 回放、配置优先级和摄像头 RTSP |
| `camera_validation` / `camera_preprocess` / `camera_dma` / `camera_cli` | hardware | 真实/合成颜色、stride、DMA 轮换/池压力、损坏包、初始化失败、租约与严格失败 |
| `live_npu` / `live_media` / `live_cli` | hardware | 三核心模式对照、MPP 注册/帧池、背压/乱序/写出失败、CLI 和防抖 ROI 因果性 |
| `stream_media` / `stream_timeout` / `stream_rtsp` | hardware | MPP H.264 像素/PTS/包长、超时隔离、RTSP/WHEP 和自建服务恢复 |
| `core` / `hardware` / `cli` / `yolo_perf` | hardware | 文件链路/RGA/LK、CPU参考与FD推理等价、输出/参数、Runtime profiling |
| `multi_video` | hardware | 四路实际 NPU/串行一致性、检测频率、混合帧率、录制与故障退出 |

测试和部署细节：[COCO640](../yolo_run/deploy/COCO640.md) · [防抖](../yolo_run/deploy/STABILIZATION.md) · [推流](../yolo_run/deploy/STREAMING.md) · [四路视频](../yolo_run/video/MULTI_VIDEO.md)。

## 目录与数据保留

仓库提供回归源码、评测器、模型及汇总报告。实拍素材、BDD100K 原视频与标注、逐帧原始输出和备份由板端单独保存。下表及历史报告中的这些路径指本地数据。重跑依赖固定视频的硬件回归或精度对照时，将自行取得的素材按评测目录的 `manifest.json` 与 `SHA256SUMS` 核对并放回对应路径。

| 目录 | 用途 |
|---|---|
| `cpp/`、`python/` | CTest 当前回归源码 |
| `support/` | 故障注入、CPU 参考、夹具录制；正式应用不链接测试支持 |
| `tools/` | 有效的像素/短链路、评测、基准与长测工具 |
| `reports/` | 按日期记录的版本验收；见 [报告索引](reports/README.md) |
| `results/` | 被忽略的运行产物，部分仍被报告引用或用作回放输入 |
| `validation/`、`camera/` | 历史证据、样本与夹具；不是待执行测试目录 |
| `rknn_video_compare_20261018_002d290d/` | 固定视频、标注、FP32 基线和外部评测器，保留完整性 |

旧混合构建归档和迁移校验脚本已删除。旧路径映射清单与源码快照留在 `archive/` 供历史追溯。新实验写入 `results/`，使用此前不存在的输出目录。

## 像素、摄像头与防抖

所有输出目录/文件须全新。以下从 `test/` 执行；`results/` 不存在时先 `mkdir -p results`。`--build-dir` 指向含程序的目录，测试构建为 `build-rk3588/bin/`，生产构建为 `../yolo_run/build-rk3588/`。

```bash
cd /home/yolo/test
mkdir -p results
run_dir=$(mktemp -d "$PWD/results/pixel.XXXXXXXX")
./build-rk3588/bin/camera_validation --frames 8 --output "$run_dir/validation.json"
/usr/bin/python3 -B tools/camera_soak.py --frames 150 --output-dir results/camera-NEW
/usr/bin/python3 -B tools/validate_capture.py \
  --build-dir /home/yolo/yolo_run/build-rk3588 --output-dir results/capture-NEW
/usr/bin/python3 -B tools/benchmark_dma_capture.py \
  --compare-stabilization --output-dir results/stabilization-NEW
```

`camera_validation` 独立验证真实颜色和补边；正式应用不做 checksum 诊断。`validate_capture.py` 串行运行 150 帧 YUYV 与 300 帧 MJPEG，保存哈希、逐帧结果、资源和内核增量。`benchmark_dma_capture.py` 交替测试防抖开关各三轮 300 帧，再开启防抖连续 3000 帧；不加 `--compare-stabilization` 时用于采集复制/资源检查，可用 `--before` 指定保留的旧二进制。

可重复硬件防抖演示：`./build-rk3588/bin/stabilization_test --artifacts results/stabilization-demo-NEW`。灰度估计、实际 ROI、坐标还原和现场背景不足的验证边界见 [防抖验收](reports/stabilization_validation.md)。

## 夹具、评测与长测

```bash
cd /home/yolo/test
./build-rk3588/bin/live_fixture capture results/camera-fixture-NEW 900
./build-rk3588/bin/live_fixture video \
  rknn_video_compare_20261018_002d290d/video/002d290d-90f2bab2.mov \
  results/traffic-fixture-NEW
./build-rk3588/bin/camera_live --preview off --stabilize off \
  --replay results/camera-fixture-NEW/manifest.jsonl --output results/replay-NEW.jsonl
```

| 工具 | 用途与关键参数 |
|---|---|
| `soak_test.py` | 文件链路长测：`--frames 3000 --input validation/soak/input.avi --output-dir results/file-soak-NEW` |
| `compare_rknn.py` | 原视频 FP16/INT8 检测与文件链路：`--repeats 3 --output-dir results/bdd-NEW` |
| `compare_lk_rga.py` | 按 LK 灰度尺度比较，当前实跑支持 `rga:1`、`rga:0.5` 等，默认固定调度：`--dataset rknn_video_compare_20261018_002d290d --repeats 2 --output-dir results/lk-NEW` |
| `validate_camera_live.py` | 三模式回放/真实摄像头：`--fixture results/camera-fixture-NEW/manifest.jsonl --traffic-fixture results/traffic-fixture-NEW/manifest.jsonl --soak-frames 3000 --output-dir results/live-NEW` |
| `compare_split_precision.py` | 同帧 FP16/INT8 精度与实时性能分开测：`--output-dir results/split-precision-NEW --repeats 3` |
| `benchmark_split_models.py` | FP16/INT8/Hybrid 性能及连续资源：`--output-dir results/split-models-NEW --repeats 3 --soak-seconds 600` |
| `benchmark_multi_video.py` | 1～4 路本地视频资源/调度；命令见 [四路说明](../yolo_run/video/MULTI_VIDEO.md#验证命令) |
| `stream_soak.py` | 独立发布/推流长测；命令见 [推流说明](../yolo_run/deploy/STREAMING.md#验证与长测) |

表中 Python 工具均用 `/usr/bin/python3 -B tools/工具名.py ...`，不安装新推理依赖。`compare_split_precision.py`、`benchmark_split_models.py` 的默认夹具是保留的 `validation/camera_live_20261018/traffic_fixture/manifest.jsonl`。精度/三模式对照显式关闭防抖，使预测保持完整原图坐标；避免默认防抖改变历史比较条件。

**评测已知限制**：`validate_camera_live.py` 的 tracking 指标适配尚需修正非单位标注 stride 的时间索引；连续漏检、fragment 和 jitter 等依赖相邻标注时刻的字段受影响，不作为新的跟踪精度验收。详见 [后续路线](../docs/development-roadmap.md)。Hybrid 的同帧 mAP 仍待补齐。

## Runtime profiling

```bash
cd /home/yolo/test
ctest --preset rk3588-release -R '^yolo_perf$'
./build-rk3588/bin/yolo_perf_test \
  --model ../yolo_model_rknn/yolo26s_800x608_int8.rknn \
  --warmup 3 --iterations 20 --output-dir results/yolo-perf-NEW
```

`--help` 列出 `--image`、`--config`、`--core default|0|1|2|all` 等参数。默认使用 `config.yaml` 的 FP16 和官方 `bus.jpg`；先测关闭采集的独立 Context，再测启用 `RKNN_FLAG_COLLECT_PERF_MASK` 的诊断 Context。固定输入串行 FPS 不含解码/预处理，诊断采集会影响性能；逐层表和 Runtime 运行耗时不能当作纯 NPU 或视频端到端 FPS。接口与检测等价验证见 [profiling 报告](reports/yolo_perf_validation.md)。

## 离线历史结果与完整性

```bash
cd /home/yolo/test
/usr/bin/python3 -B tools/summarize_metrics.py
/usr/bin/python3 -B tools/compare_rknn.py --report-only \
  --from-dir validation/bdd_compare_20261018 --output-dir results/bdd-regenerated-NEW
cd rknn_video_compare_20261018_002d290d
sha256sum -c SHA256SUMS
/usr/bin/python3 -B -m unittest discover -s tools -p test_compare.py -v
```

汇总脚本只重生成 `reports/metrics.json`、`metrics.md`；不重新推理。LK 历史合并使用 `compare_lk_rga.py --merge`，具体选项见 `--help`。历史 `yolo_run/validation/` → `test/validation/`、`Camera/` → `test/camera/` 的路径通过 `tools/paths.py` 在读取时映射，不重写原始命令与哈希。

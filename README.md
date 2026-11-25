# RK3588 工业视觉与 YOLO26 部署

C++17 板端工程，运行于 RK3588 ARM64 / Linux 6.6.89-kdev，复用 RKNN Runtime 2.5.0、librga 1.10.6、现有 MPP 和系统 OpenCV 4.10.0。模型转换在另一台 Linux 主机完成。

板端源码、RKNN 权重和测试工具在本仓库维护。主机端训练、导出与工作台见 [VisionTrack](https://github.com/ChutianDuan/Yolo)。

## 实验结果

**单路 COCO640 实摄**在 RK3588 上使用 INT8、两个 NPU Context 和压缩包 FIFO，整个进程限定在 A76 核上。三轮各 60 秒，共采集 5,402 帧，每帧都完成检测、跟踪校正和画面输出。检测和画面的 P99 延迟分别为 70.41 ms 和 26.20 ms，均取三轮中位数。[实验结果索引](profiling/results/README.md)

| 想看什么 | 入口 |
|---|---|
| 比较 COCO INT8 与 FP16、单核与多核，以及 10 分钟长稳结果 | [144 轮回放对照](profiling/results/comparison.md) |
| 在一帧上查看采集、排队、推理和画面输出的先后关系 | [离线交互报告](profiling/examples/replay/report.html) |
| 在自己的板上重复摄像头或回放实验 | [配置与复现命令](profiling/README.md#重现实验) |
| 查找其他模型精度、多路视频、防抖和推流实验 | [历史性能与精度](docs/performance.md) · [验收报告索引](test/reports/README.md) |

程序用法、架构与部署见 [项目文档索引](docs/README.md)。

## 当前能力与默认值

| 正式程序 | 输入 | 默认行为 |
|---|---|---|
| `yolo_run` | 本地图片 / 视频 | 800×608、10 类 FP16、RGA；视频异步检测、LK 与 ByteTrack，检测目标 4 FPS |
| `camera_run` | USB YUYV 800×600，请求 15 FPS | 800×608、10 类 INT8；DMA-BUF 采集、RGA、单 Context 逐帧检测，无跟踪 |
| `camera_live` | USB MJPEG 800×600，请求 30 FPS，或 JPEG 夹具回放 | MPP 解码、INT8、split 三个独立 Context、ByteTrack/Kalman；平移防抖和本地预览默认开启，推流默认关闭 |
| `yolo_multi` | 1～4 路本地视频文件 | INT8、三核固定 Worker、公平派发；每路独立预测/校正和输出，不运行 LK |

- 生产预处理统一使用 RGA / DMA-BUF；YUYV 不再提供 `copy/auto`，实时解码不再提供 OpenCV 分支。文件视频仍使用 CPU OpenCV/FFmpeg 解码并上传 BGR；CPU 像素/推理参考保留在测试中。
- COCO80 / 640×640 配置默认加载 FP16。复现实验中的 INT8 时，同时传入该配置和 `--model yolo_model_rknn/yolo26s_coco_int8.rknn`，沿用相同的标签、输入尺寸和 letterbox。[COCO640 配置](yolo_run/config/camera_coco640.yaml)
- 实时防抖默认保留 720×540 ROI、输出 800×600；检测、跟踪、预览和推流共享防抖画布坐标。`--stabilize off` 使用完整原图，JSONL schema v2 保存坐标类型和逐帧 ROI，详见 [防抖说明](yolo_run/deploy/STABILIZATION.md)。
- 推流链路为 C++ MPP H.264 → MPEG-TS/本机 UDP → 独立 FFmpeg → 独立 MediaMTX → RTSP/WebRTC。当前支持这些协议的输出；RTSP 输入和 H.264 输入硬解仍是后续工作。

## 环境与构建

```bash
cd /home/yolo
bash scripts/dev.sh doctor
bash scripts/dev.sh build
bash scripts/dev.sh test
```

生产产物在 `yolo_run/build-rk3588/`，包含上表四个程序。测试可执行文件在 `test/build-rk3588/bin/`，直接复用正式源码。原生 C++17 / Ninja 构建并发 4，C++ 链接系统 OpenCV/libstdc++ 和现有 Rockchip Targets。

日常 `test` 按 `cpu` 标签执行 **10 项无需摄像头、NPU、RGA/MPP 硬件的回归**，包括 COCO 契约、防抖算法、录像、调度和发布器模拟。完整套件 **30 项**另含实际硬件测试，需独占设备并串行执行：

```bash
ctest --test-dir /home/yolo/test/build-rk3588 --output-on-failure -j1
```

Python / 子 shell 使用 `bash scripts/dev.sh shell`，默认复用 `rk3588-infer`；标准库脚本用 `bash scripts/dev.sh --role system python ...`。新终端不自动激活环境。依赖规格和精确锁在 [env/](env/environment.yml)，详见 [环境管理](docs/environment.md)。

## 四个运行入口

先构建，并在项目根目录执行一次 `mkdir -p test/results`。以下示例都从 `/home/yolo` 执行。新目录避免覆盖输出，`max-frames` 到达后正常退出；持续运行时可用 Ctrl+C 收尾。设备检查使用 `v4l2-ctl --list-devices` 和 `fuser /dev/video1 /dev/rga /dev/mpp_service`，设备枚举以实机为准。

### YUYV 逐帧检测

```bash
run_dir=$(mktemp -d /home/yolo/test/results/yuyv.XXXXXXXX)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/camera_run \
  --io dmabuf --max-frames 150 \
  --output "$run_dir/results.jsonl" --snapshot "$run_dir/snapshot.png"
```

默认设备为 `/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0`；当前指向 `/dev/video1`。YUYV 请求约 15 FPS；默认模型输入 800×608，检测坐标恢复至 800×600 原图。初始化失败退出。

### MJPEG 实时检测、跟踪和防抖

```bash
run_dir=$(mktemp -d /home/yolo/test/results/live.XXXXXXXX)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/camera_live \
  --preview off --npu-mode split --max-frames 300 \
  --output "$run_dir/results.jsonl"
```

无图形会话保持 `--preview off`。输出帧与检测异步，30 FPS 输出不表示每帧都执行检测。COCO 模型在上述命令中增加 `--config /home/yolo/yolo_run/config/camera_coco640.yaml`；恢复原始完整视野增加 `--stabilize off`。`--model` 优先于显式配置中的模型，CLI 路径相对当前工作目录。

### 本地图片或视频

```bash
run_dir=$(mktemp -d /home/yolo/test/results/file.XXXXXXXX)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/yolo_run \
  --input /opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg \
  --output "$run_dir/results.jsonl" --annotated "$run_dir/marked.jpg"
```

示例图片来自板上已有的 RKNN Model Zoo，也可替换为自己的图片或视频。标注视频输出用 `.avi`。配置中的模型/标签相对配置文件解析。

### 四路本地视频

```bash
clip=/path/to/your-video.mp4  # 替换为自己的本地视频
run_dir=$(mktemp -d /home/yolo/test/results/multi.XXXXXXXX)
./yolo_run/build-rk3588/yolo_multi --input "$clip" --input "$clip" --input "$clip" --input "$clip" \
  --duration-seconds 10 --preview off --record off --output-dir "$run_dir/channels"
```

逐路参数、四宫格、录像和基准命令见 [四路视频说明](yolo_run/video/MULTI_VIDEO.md)。只有一路时，每路一个在途任务的限制仍然生效，不能据三 Worker 推导单路三帧并发。

远程观看按 [推流部署](yolo_run/deploy/STREAMING.md) 分别启动 MediaMTX、发布器和 `camera_live --stream on`；COCO 完整步骤见 [COCO640 部署](yolo_run/deploy/COCO640.md)。板端地址以 `hostname -I` 为准，文档示例地址须按实际网络替换。

## 架构与内存边界

```text
USB YUYV → V4L2 DMA-BUF → RGA → RGB DMA-BUF → RKNN → 检测
USB MJPEG → V4L2 DMA-BUF → MPP NV12 → 防抖估计并固定本帧 ROI
                                      ├→ RGA ROI / RGB → RKNN → 跟踪校正
                                      └→ RGA ROI / BGR → 预测、画框与输出
                                                            └→ 可选 MPP H.264 推流
```

RGA 读取源分配并写入独立 RGB 分配，RKNN 导入同一 RGB FD。各 Worker 独占 Context 和模型输入；分配存活、当前帧租约、CPU Cache Sync 与设备完成分别管理。MJPEG 采集使用 4 个 V4L2 队列槽、8 块输入 DMA-BUF，换入空闲缓冲后归还队列槽，应用层压缩数据复制为 0；回放包池为 4 槽。

图像转换、CPU 画框、LK、防抖估计和码流读取仍有 CPU/设备访问，**不宣称端到端零拷贝**。详见 [架构、并发与所有权](docs/architecture.md) 和 [设计图](docs/design-diagrams.md)。

## 验证与后续工作

单路追踪提交前，板端完整回归通过 30/30 项，实摄短测的 91 个采集帧全部核对到终态。[板端验证记录](profiling/validation.md)

2026-11-18 的防抖验收使用 COCO FP16、split，关闭预览和推流。3,000 个输出帧约为 29.98 FPS，实际检测约为 14.31 FPS。实机场景静态背景不足，多数帧进入失锁回中，抑抖幅度由可重复的硬件回放测得。[防抖验收](test/reports/stabilization_validation.md)

[历史性能与精度](docs/performance.md) 保留三模型、单路推流和多路实验的日期及条件。仍需补 Hybrid 同帧 mAP、评测时间索引修正、当前版本长测、RTSP 输入与恢复、真实观看端延迟和独立工业数据集；见 [后续路线](docs/development-roadmap.md)。

源码与测试位置：[正式工程](yolo_run/README.md) · [测试指南](test/README.md) · [报告索引](test/reports/README.md) · [学习路线](学习路线.md) · [面试复习](docs/interview.md)。

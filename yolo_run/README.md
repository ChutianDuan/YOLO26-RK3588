# RK3588 YOLO26 正式运行工程

当前入口、四个程序的默认值和可运行示例见 [项目 README](../README.md)。核对日期：2026-11-18（Asia/Shanghai），包含 COCO640、DMA-BUF 单路径与默认开启的实时平移防抖。

```bash
bash /home/yolo/scripts/dev.sh build
cd /home/yolo/yolo_run
./build-rk3588/yolo_run --help
./build-rk3588/camera_run --help
./build-rk3588/camera_live --help
./build-rk3588/yolo_multi --help
```

本目录只构建正式应用和库；CPU 参考、故障注入和回归夹具由开发板本地维护。C++17、系统 GCC/OpenCV/libstdc++、Ninja 并发 4，统一链接现有 `Rockchip::RKNN` / `Rockchip::RGA` / `Rockchip::MPP`；环境与依赖锁见 [开发环境](../环境配置.md)。

## 运行与配置

| 入口 | 默认配置与行为 | 专题 |
|---|---|---|
| `yolo_run` | `config.yaml`，800×608、10 类 FP16；文件视频异步检测、LK/ByteTrack | [源码学习路线](../学习路线.md) |
| `camera_run` | 不传 `--config` 时使用 800×608 INT8；YUYV 800×600/15 FPS、DMA-BUF、逐帧检测 | [采集实现](camera/v4l2_camera.cpp) |
| `camera_live` | 不传 `--config` 时使用 INT8；`config/camera_live.json` 选择 split；MJPEG/MPP、防抖/预览默认开、推流默认关 | [实时流水线](camera/live_pipeline.cpp) |
| `yolo_multi` | `config/video_multi.yaml`，INT8；1～4 路文件、三个单核 Worker、每路一个在途 | [多路运行与验证](video/MULTI_VIDEO.md) |

COCO80 / 640×640 FP16 须传 `--config /home/yolo/yolo_run/config/camera_coco640.yaml`；切换模型须匹配尺寸、标签和双输出契约。`--model` 优先覆盖配置中的模型路径；配置路径相对于配置文件，CLI 模型路径相对于工作目录。单输出 `[1,300,6]` e2e 模型尚未接入此双输出引擎。

生产链路统一为 DMA-BUF/RGA，摄像头不提供 `copy/auto`、实时 OpenCV 解码或 CPU RKNN 输入分支。MJPEG 为 4 个 V4L2 槽/8 块输入 DMA-BUF，回放包池为 4 槽，应用层压缩数据复制为 0。文件 BGR 上传、CPU 画框/LK/防抖和编码输出读取仍保留。

`camera_live` 默认防抖裁剪 720×540 并缩放为 800×600；检测、跟踪和视频输出使用防抖画布坐标，JSONL schema v2 保存每帧 ROI。关闭用 `--stabilize off`，平滑参数为 `--stabilize-alpha 0.15`。接口、坐标映射和边界见 [防抖说明](deploy/STABILIZATION.md)。

远程观看使用 [推流部署](deploy/STREAMING.md) 或 [COCO640 浏览器监控](deploy/COCO640.md)：C++ 生成 H.264/MPEG-TS，独立 FFmpeg 发布，独立 MediaMTX 分发。地址以实机为准；手动启动各进程。

## DMA 代码阅读入口

| 职责 | 实现与所有权 |
|---|---|
| 分配与 CPU 访问 | [DmaBuffer / CpuAccess](image/dma_buffer.h)、[实现](image/dma_buffer.cpp)：DMA Heap、mmap、RGA handle、CPU START/END；不代替设备完成等待 |
| 当前帧占用 | [FrameLease](image/frame_lease.h)：`owner` 保护当前帧，最后租约释放后才可覆盖；`allocation` 保持底层内存存在 |
| MPP FD 注册 | [MppDmaBuffer](image/mpp_dma_buffer.h)：移动转移注册，先 put 注册、再释放分配引用；仅在媒体 Target 编译 |
| RGA → RKNN | [预处理](image/image_processing.cpp) 写独立 RGB 并发布输入；[引擎](model/yolo_engine.cpp) 分别持有注册缓存和在途输入 |
| 防抖 ROI | [FrameStabilizer](video/frame_stabilizer.cpp)：两槽小图灰度、CPU 背景估计；[流水线](camera/live_pipeline.cpp) 分流前固定本帧 ROI |

`Renderer` 的 Mat 是借用视图，CPU 使用必须在 `finish()` 前完成。共享引用、Cache Sync、RGA `IM_SYNC`、MPP/RKNN 完成确认分别承担不同职责。源 NV12 的最后消费者归还后可复用；NPU 此时可继续读取独立 RGB。分流与线程退出顺序可在 [实时流水线](camera/live_pipeline.cpp) 中核对。

## 验证边界

交通 INT8 的 900 帧定速回放中，default、all 和 split 的校正年龄 P95 分别为 105、136 和 73 ms。单路实摄、文件回放与四路调度的测量条件和数据见 [项目 README](../README.md)。编译成功、运行成功、结果正确与性能达标分别报告，不宣称端到端零拷贝、完整工业精度或无限时长稳定性。

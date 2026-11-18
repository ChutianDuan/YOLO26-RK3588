# DMA 分配与生命周期可读性整理验收

> **历史记录**。下文的接口、命令和结果对应当时版本。当前命令见[测试指南](../README.md)。 短测工具现名 `tools/validate_capture.py`。

日期：2026-10-20（上海时区）。验证设备为当前 RK3588 ARM64 板，内核 6.6.89-kdev。

## 改动和所有权

本次保持 CLI、配置、JSONL、模型、池容量、调度和错误处理策略，整理完整 DMA 链路：

| 对象 | 职责和释放边界 |
|---|---|
| DmaBuffer | 资源实现移至独立 cpp；持有 DMA-BUF FD、mmap 和 RGA handle，最后依次 release handle、munmap、close |
| CpuAccess | START/END 配对，正常显式 finish，异常由析构收尾；禁止复制，不承担设备完成等待 |
| FrameLease | 替换检测 RGB 和编码池的嵌套 shared_ptr；消费者共享同一租约，最后副本释放后允许复用 |
| MppDmaBuffer | 统一解码／编码 EXT_DMA 导入，持有注册与分配引用；不可复制、可以移动，先 put 注册再归还分配引用 |
| RKNN 注册与在途输入 | 注册可长期缓存；active_input 单独持有当前帧，失败后仍按原流程等待／清理 |

预处理提取了 BGR 行上传、RGB 目标准备、补边初始化及 TensorInput 发布。YUYV/NV12 各自保留布局、颜色和容量检查。V4L2 槽租约直接构造，Renderer 借用视图、MPP 多消费者归还及编码 EOI 边界均保留。PacketStorage／Surface 继续按槽包装的引用数判断空闲，RGB／编码池继续按独立 weak lease 判断。

代码阅读入口见 [正式工程 README](../../yolo_run/README.md#dma-代码阅读入口)。

## 编译和回归

系统 GCC、C++17、Ninja 并发 4。生产与测试工程均通过原生构建；最终构建日志没有编译警告或错误。ldd 确认三个正式入口继续使用 /opt 下的 RKNN/RGA/MPP 及系统 OpenCV、libstdc++，没有 Conda 库或缺失依赖。

| 验证组 | 整理前 | 整理后 |
|---|---|---|
| core、hardware、camera_preprocess、live_core、live_media、live_npu、stream_media、stream_timeout | 8/8 通过，25.10 秒 | 8/8 通过，24.93 秒 |
| RKNN 注册配平 | imports=11、releases=11、peak=1 | imports=11、releases=11、peak=1 |
| 同 RGB 的 CPU/FD 输入等价 | FP16/INT8 各 32 次通过 | FP16/INT8 各 32 次通过 |
| hardware 测试稳态 FD | FP16=19、INT8=18 | FP16=19、INT8=18 |

新增断言已通过：多个 TensorInput 副本持有时禁止覆盖，最后副本归还后即使 allocation 仍被引用也可复用；编码帧副本阻止选槽；MPP 注册移动构造、移动赋值替换及 reset 后的分配持有和 FD 关闭。原有 RKNN 错误等待、MPP 编码超时与 reset 失败隔离断言继续通过。

## 真摄像头短测

测试前检查设备空闲，没有停止其他进程。

- camera_test --capture 通过：旧采集帧阻止继续取帧，采集对象析构后被租用的源内存仍有效。
- camera_validation 8 帧通过：DMA/copy RGB 像素一致，letterbox 填充值为 114；checksum 变化 7 次。
- 现有 validate_reorganization.py 使用正式构建完成以下三项，所有检查通过：

| 输入模式 | 输出／推理帧数 | 应用上传字节 | FD 首／末 | 本次短测观察 |
|---|---|---|---|---|
| YUYV DMABUF | 150／150 | 0 | 16／16 | 约 14.42 推理 FPS，RSS 采样保持 69.918 MiB |
| YUYV copy | 30／30 | 28,800,000 | 13／13 | 上传记账为每帧 960,000 字节，RSS 采样保持 73.656 MiB |
| MJPEG MPP split | 300／300 | 由 source 单独统计压缩包复制 | 44／47 | 约 29.98 输出／推理 FPS，decoded_pool_dropped=0，结果／JSON 队列峰值均为 1 |

MJPEG 的 FD 从启动采样 44 增至 47；首次 RGB 分配／RKNN 注册按需建立，此后的采样均为 47。RSS 从 140.375 增至 141.25 MiB。YUYV 两次运行各记录 1 次驱动序号缺口；按现有最新帧策略处理，完成帧数及源时间／序号单调检查均通过。

测试前、回归后、摄像头短测后的内核 DMA-BUF 表均为 0 对象、0 字节。回归及摄像头短测日志中没有新增匹配 RKNPU/RGA/UVC/MPP 的 error、fault、failed、timeout 或 reset 条目。

## 当时的命令与证据

```bash
cd /home/yolo/yolo_run
cmake --preset rk3588-release
cmake --build --preset rk3588-release

cd /home/yolo/test
cmake --preset rk3588-release
cmake --build --preset rk3588-release
env -u LD_LIBRARY_PATH -u LD_PRELOAD ctest --preset rk3588-release \
  -R '^(core|hardware|camera_preprocess|live_core|live_media|live_npu|stream_media|stream_timeout)$'
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./build-rk3588/bin/camera_test --capture
/usr/bin/python3 tools/validate_reorganization.py \
  --build-dir /home/yolo/yolo_run/build-rk3588 --output-dir results/dma-short-NEW
```

短测输出目录必须不存在。完整的逐条命令、二进制哈希、帧输出和资源采样见 [摄像头报告](../results/dma_readability_u_lmygip/camera-short/report.json)。[整理前日志](../results/dma_readability_u_lmygip/baseline-4.log)、[整理后日志](../results/dma_readability_u_lmygip/after-ctest.log)、[测试详细输出](../results/dma_readability_u_lmygip/after-details.log)、[动态链接](../results/dma_readability_u_lmygip/linked-libraries.json)均已保留。

当前目录没有 Git 元数据，因此实施前保存了 [源码快照](../results/dma_readability_u_lmygip/source-before.tar.gz)，整理后的 [源码差异](../results/dma_readability_u_lmygip/source-changes.diff)单独提供。

## 验证边界和保留事项

本次验证了构建、短时运行、指定像素／所有权／输入等价断言，不构成长期无泄漏或性能达标证明；未重做长测、工业检测精度评测、真实驱动卡死恢复或 RTSP／浏览器观看验收。

审查还记录了原有行为差异：文件 BGR／LK 路径在 RGA 执行返回错误时抛异常，而摄像头 YUYV／NV12 和 Renderer 路径采用终止策略。该错误返回是否足以保证设备结束访问需要单独验证，本次没有改动或注入真实 RGA 故障。SDK、内核、驱动、设备权限及系统配置均未修改。

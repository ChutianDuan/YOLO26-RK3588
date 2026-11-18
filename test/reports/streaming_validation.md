# camera_live 编码与独立 RTSP 服务验收

本报告区分编译、运行、结果正确与性能测量；不作端到端零拷贝或工业性能达标声明。

## 环境与构建

- RK3588 / ARM64，Linux 6.6.89-kdev，系统 GCC/G++、C++17、Ninja、并发四。
- 复用 RKNN Runtime 2.5.0、RGA 1.10.6、固定版本 MPP 及系统 OpenCV 4.10.0。
- FFmpeg 7.1.1；编码由 C++ 调用 MPP，FFmpeg 仅 stream copy 发布。
- MediaMTX v1.21.2 官方 Linux ARM64 包，通过官方清单与固定 SHA-256 双重核验。
- 正式工程与测试工程均原生编译成功；`ldd` 确认使用 `/opt` Rockchip SDK、系统 OpenCV/FFmpeg/libstdc++，没有 Conda C++ 库。
- 二进制摘要、链接映射、内核和 FFmpeg 版本保存在 `test/results/stream_implementation/`。

## 已完成的正确性与故障验证

原有十二项 CTest 均已通过。新增四项通过的测试包括：

| 测试 | 结果 |
|---|---|
| `stream_media` | 真实 MPP 编码后由系统 FFmpeg 解码；800×600、Constrained Baseline、Level 3.1、无 B 帧、BT.601 limited；64 帧 PTS 保留源时间及人工间隔，色块各通道均值误差≤8，已知框/ID 位置正确；UDP 包长≤1316 且为 188 的倍数 |
| `stream_timeout` | 注入取包超时后在 reset/destroy 前保持输入租约；确认清理后归还；reset 失败则保留固定资源至进程退出 |
| `stream_publisher` | stream-copy 参数、十秒无进度看门狗机制（测试使用缩短阈值）、首次一秒退避与子进程清理 |
| `stream_rtsp` | 真摄像头 RTSP 拉流并解码，WebRTC 页面及 WHEP SDP 协商；慢客户端、MediaMTX 重启、FFmpeg 子进程被杀、发布器停止/重启；推理继续且发布恢复；慢编码有界丢帧、编码失败禁用通道而不停止推理；非法 CLI 参数拒绝 |

早期集成测试使用了本内核未提供的 `/proc/PID/task/PID/children`，测试脚本因此失败。已改为读取 `/proc/PID/stat` 的父 PID，并完成重测；未修改内核或系统配置。

集成测试使用独立临时端口，结果、JSONL、进程日志和 WHEP answer 位于 `test/results/stream_implementation/integration_*/`。

## 三十分钟真机长测

已完成发布就绪后的 **1800.000 秒**真摄像头长测；`camera_live` 自身运行 1802.254 秒，结束时由脚本发送 SIGINT 正常收尾。采用默认 INT8、split 三 Context、800×600、2 Mbps 目标码率，独立 FFmpeg、MediaMTX 和持续 RTSP 拉流端同时运行。

| 项目 | 实测 |
|---|---|
| 输出 / 推理 / 编码帧数 | 均为 53,988 |
| 输出 / 编码速率 | 29.956 / 29.956 FPS |
| 编码池丢帧 / 等待槽替换 / 编码关闭舍弃 | 0 / 0 / 0 |
| UDP 发送错误 / 发送量 | 0 / 487,374,020 字节 |
| 编码耗时，平均 / P95 / P99 | 1.746 / 3 / 3 ms |
| BGR→NV12 转换，平均 / P95 | 0.572 / 1 ms |
| 采集到编码完成，平均 / P95 / P99 | 35.073 / 39 / 39 ms |
| 摄像头进程 RSS，预热三十秒后 | 始终为 142.734 MiB |
| 摄像头进程 FD，预热三十秒后 | 60～61，无持续增长 |
| NPU Core0 / Core1 / Core2 平均报告负载 | 40.34% / 41.39% / 41.48% |
| 最高温度样本，各 thermal zone 最大值 | 49.923°C |

CPU 为预热三十秒后的约两秒采样平均，**100% 表示一个 CPU 核**，不表示整板利用率：

| 进程 | 平均 CPU | RSS 范围 MiB | FD 范围 |
|---|---:|---:|---:|
| camera_live | 22.49% | 142.734～142.734 | 60～61 |
| 发布 Python 监督进程 | 0.08% | 12.125～12.125 | 5 |
| FFmpeg 发布子进程 | 1.57% | 44.301～44.426 | 6 |
| MediaMTX | 3.56% | 41.941～50.656 | 12～14 |
| 本机 RTSP 拉流端 | 1.37% | 43.082～43.082 | 4 |

持续拉流端使用 stream copy 验证持续传输；独立集成测试另外执行了 RTSP 解码。用户在长测期间确认 Mac RTSP 和 Chrome/Edge 浏览器可正常观看，MediaMTX 日志记录到局域网 WebRTC peer connection established 和 H.264 reading。

源统计记录 **1 次驱动序号缺口**；正常退出时丢弃 **1 个尚未解码的压缩包**。解码池、推理等待槽、输出等待槽均未记录运行期丢帧，已经输出的 53,988 帧全部完成编码。UDP 发送成功不证明接收端绝无丢包。

内核 DMA-BUF 调试信息显示整个 `camera_live` 持有 39 个对象、95,043,584 字节（约 90.64 MiB，包含原有 RKNN/解码缓冲）。中后段对象标识与容量完全一致；退出后这些标识全部消失，内核表为 **0 对象、0 字节**。这同时验证了本次正常退出的分配释放，不等于覆盖所有真实驱动故障。

测试创建的摄像头、发布器、FFmpeg、MediaMTX 和拉流进程已全部退出，相关监听端口已关闭。正式二进制和源文件 SHA-256 与长测前记录一致。

原始记录与机器报告：

- [长测 report.json](../results/stream_implementation/soak_30min/report.json)
- [客户端反馈](../results/stream_implementation/soak_30min/client_feedback.json)
- [新增 CTest 日志](../results/stream_implementation/stream_ctest.log)
- [DMA-BUF 运行期快照](../results/stream_implementation/soak_30min/dma_buf_during.txt) / [后半程快照](../results/stream_implementation/soak_30min/dma_buf_late.txt) / [退出后快照](../results/stream_implementation/soak_30min/dma_buf_after.txt)

未设置性能硬门槛；上述为本次单路、当前场景和默认参数下的实测，不作为多路或工业场景性能承诺。

## 验证边界

- WHEP 协商成功不等于浏览器媒体播放成功；本次用户已确认 Mac RTSP 与浏览器均可正常观看，浏览器为用户选定的 Chrome/Edge（Chromium 系），具体版本和 RTSP 播放器型号未提供。长测日志也记录到局域网 WebRTC peer connection established 与 H.264 reading 会话。
- 采集到编码完成耗时不包含发布、网络与客户端缓冲；玻璃到玻璃延迟尚未测量。
- reset 失败隔离路径使用测试注入；未对真实驱动制造故障。同步 RGA 无法确认完成时沿用进程终止策略。
- 第一版为单路、局域网、无音频；未测试四路、跨网、认证部署和开机服务。
- SDK、设备权限、内核/驱动/引导配置均未修改；测试启动的进程已清理，未配置开机服务。

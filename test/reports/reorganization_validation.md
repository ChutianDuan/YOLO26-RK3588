# 目录整理与板端验收

> **历史记录**。下文的接口、命令和结果对应当时版本。当前命令见[测试指南](../README.md)。 旧构建目录已于 2026-11-18 删除，63 个文件可从 Git 提交 `6ad99f2` 找回。 短测工具现名 `tools/validate_capture.py`。

日期：2026-10-18（Asia/Shanghai）。本报告区分当前整理后回归与历史实验，测试使用当前 RK3588 实机。

## 交付与构建

- 正式源码与配置保留在 `yolo_run/`；测试、实验工具、验证记录和原 Camera 材料已统一迁入 `test/`。
- 原生 C++17 Release、系统 GCC/Ninja，Preset 构建并发度 4；生产与测试工程分别构建通过。
- 正式构建仅包含三个应用；测试与夹具程序仅在测试构建中生成。正式二进制不包含测试参数、模拟写入失败字符串或 checksum 字段。
- 动态链接检查：RKNN/RGA/MPP 来自既有 SDK，OpenCV/libstdc++ 来自系统；无 Conda 链接或缺失库。模型和 SDK 哈希与历史环境一致。
- 无新增依赖，未修改 SDK、内核、驱动、设备树、boot、设备权限或系统服务。

## 回归结果

- CTest **11/11 通过，90.54 秒**：原有 9 项、独立摄像头像素验证、历史路径回归。硬件相关测试串行执行。
- 独立像素验证：真实 DMA-BUF 与复制路径 8 帧 RGB 完全一致，114 补边正确；checksum 留在测试程序中。合成颜色/stride/租约与真实采集所有权回归通过。
- CLI 回归涵盖真实 DMA/复制、不支持 DMA-BUF 时 auto 回退和强制失败、输出保护、Ctrl+C；实时流程覆盖慢推理/输出、乱序、写出失败与队列背压。
- 真实夹具录制 8 帧成功；录制后的 JPEG 经 MPP/RGA 与 CPU 对照，最大帧平均绝对误差 0.489、最大帧 P95 通道误差 1，检查通过。
- README 中本地图片运行命令使用正式二进制成功生成 JSONL 和标注图。

## 当前正式二进制短测

以下均使用 `yolo_run/build-rk3588` 的正式应用，未用故障或测试入口代替。采集 800×600，模型 INT8 RGB 800×608；YUYV 单 Context，MJPEG 为 MPP/split、无屏画框。

| 测试 | 帧数 / 推理次数 | 输出/检测 FPS | CPU % | 采样 RSS 峰值 MiB | 上传字节 |
|---|---:|---:|---:|---:|---:|
| yuyv-dmabuf-150 | 150 / 150 | 14.426/14.426 | 7.52 | 69.69 | 0 |
| yuyv-copy-30 | 30 / 30 | 12.469/12.469 | 10.34 | 73.51 | 28800000 |
| mjpeg-split-300 | 300 / 300 | 29.786/29.786 | 17.01 | 132.41 | 8529938 |

YUYV 的上传为原始图像上传，MJPEG 的上传值为压缩包复制字节，两者不能直接视作同类像素复制。

| 测试 | 采集到结果 P95 ms | 输出 P95 ms | 校正数据年龄 P95 ms |
|---|---:|---:|---:|
| yuyv-dmabuf-150 | 110.117 | — | — |
| yuyv-copy-30 | 131.247 | — | — |
| mjpeg-split-300 | 82 | 36 | 103 |

短测包含流启动、首帧截图；此次移除了旧版启动时先消耗 8 帧的像素校验，因此 YUYV 150 帧与 30 帧墙钟 FPS 尤其受启动影响，不能直接与历史 3000 帧已完成启动校验后的值比较。
YUYV 两组各观察到 1 个驱动序号缺口、主动丢弃 1 个积压帧，损坏帧为 0；未将启动现象隐藏为“全程零丢帧”。
YUYV 直接路径 FD 为 16/16、copy 为 13/13，采样 RSS 首尾分别保持 69.69、73.51 MiB。MJPEG 冷启动 FD 44→47、RSS 131.41→132.41 MiB；第 150～300 帧的采样 FD 均为 47、RSS 均为 132.41 MiB，该有限区间没有继续增长。
MJPEG 结果/JSON 队列峰值未超过 3/64，没有解码池耗尽，输出采用的快照未晚于图像时间。三组短测期间内核增量日志没有新增硬件错误；这不代表真实卡死恢复已经验证。

## 迁移与历史指标重算

- 迁移清单共 2741 个文件；2724 个历史证据/旧构建文件逐文件大小和 SHA-256 均一致，17 个测试工具源码允许按新目录更新；整理前源码另有完整快照。
- 数据包 SHA256SUMS 全部通过；原评测器 4 项回归通过，材料包文件保持不变。
- 从迁移后的历史预测重新计算 FP32 与四种 RKNN/预处理组合，以及历史跟踪流程；FP32 参考、检测指标和汇总值与原机器报告完全一致。
- LK 22 次历史运行离线合并通过，固定检测帧/原始检测输出逐字段一致，所有 aggregate 与原报告完全一致。没有重新运行这些 NPU 精度实验。
- 历史原始 JSON/JSONL/日志/媒体/报告不改写；工具在读取时重定位 output/config，保留历史 command。

## 兼容变化与验证边界

正式 `camera_run` 移除了启动像素比对与 checksum；auto 只在 DMA-BUF 采集初始化失败时回退。正式 JSONL 已移除对应诊断字段，像素报告由测试程序独立输出。完整清单见 [迁移说明](migration.md)。
本次确认编译、设备运行、给定夹具/输入正确性、回归行为和有限短测资源表现。未重新执行三种模式 3000 帧长测、完整新模型精度实验、物理拔插、自动重连、真实驱动卡死恢复、多路、无限时长、工业独立精度或曝光到显示延迟。
历史 INT8 精度与 IDF1/MOTA 仍有明显不足，未因本次目录整理而改善。无屏画框通过不等于实际图形桌面完整验收；仍不宣称端到端零拷贝。

## 当时的命令与证据

```bash
cd /home/yolo/test
cmake --preset rk3588-release
cmake --build --preset rk3588-release
ctest --preset rk3588-release
/usr/bin/python3 tools/validate_reorganization.py \
  --build-dir /home/yolo/yolo_run/build-rk3588 --output-dir results/short-repeat-NEW
/usr/bin/python3 tools/summarize_metrics.py
/usr/bin/python3 tools/verify_migration.py
```

`short-repeat-NEW` 必须替换为不存在的新目录。完整重算命令见 [测试指南](../README.md)。

证据目录：[reorganization_20261018_193700](../results/reorganization_20261018_193700)。其中包含：

- [CTest](../results/reorganization_20261018_193700/ctest.log)、[环境与源码哈希](../results/reorganization_20261018_193700/environment.json)、各应用的 `.ldd.log` 与最终构建检查。
- [三组短测机器报告](../results/reorganization_20261018_193700/short-runs/report.json)、同目录 JSONL/PNG/stderr、内核增量日志。
- [指标逐组一致性](../results/reorganization_20261018_193700/metric-parity.json)、[迁移哈希检查](../results/reorganization_20261018_193700/migration-verification.json)。
- [BDD 重算报告](../results/reorganization_20261018_193700/bdd-regenerated/report.md)、[LK 合并报告](../results/reorganization_20261018_193700/lk-merged/report.md)、真实夹具及像素对照日志。

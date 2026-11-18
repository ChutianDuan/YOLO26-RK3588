# 板端验证记录

> **历史记录**。下文的接口、命令和结果对应当时版本。当前命令见[测试指南](../README.md)。

最新 RGA → RKNN 外部 DMA-BUF 输入实现、硬件回归及资源检查见 [FD 输入验证报告](../validation/rknn_dma_input/report.md)。检测目标使用已有 `system-uncached` heap，FP32 输出和 YOLO26 后处理保持原契约；下方历史记录对应改动前的输入提交方式。

最新 LK RGA 实现及验收见 [LK 的 RGA 实测报告](lk_rga_report.md)。已完成 22 次 1210 帧对照、CTest 3/3、3000 帧内存检查；默认 RGA 原分辨率，缩图带来的速度与异步 F1 变化单独报告。原始数据及检查位于 `validation/lk_rga_20261018`。

本文件记录首次精简版本的验收。后续带标注实验对齐了双阶段 Top-K，并拆分 LK/ByteTrack 计时；新测量由 `tools/compare_rknn.py` 生成，汇总见 [对比报告](comparison_report.md)，不应将本文件中的旧测量视作新版本精度。

验证日期：2026-10-18（Asia/Shanghai）。平台为 RK3588 aarch64、Linux 6.6.89-kdev，系统 GCC 14.2、OpenCV 4.10、RKNN Runtime 2.5.0 / driver 0.9.8、librga 1.10.6。

## 编译与自动测试

- 原生 Release 构建通过，并发 4，无编译警告。
- CTest **3/3 通过**，耗时 18.29 秒：核心算法/并发、硬件推理、CLI 集成。
- 核心测试验证布局拒绝、量化参数检查、双输出解码、阈值、Top-K、坐标还原、LK 位移、ByteTrack ID、动态步长、64 帧历史、任务上限、异常传递与 EOF 校正。
- CLI 覆盖 FP16/INT8 × RGA/OpenCV 图片、同步/异步视频、标注媒体重读、JSONL 帧数和非法参数；正常运行 stdout 为空，SDK 与应用诊断均路由至 stderr。
- [测试摘要](../validation/ctest.log)、[测试详情](../validation/test-details.log)、[环境及模型/SDK/二进制 SHA256](../validation/environment.json)。
- [动态库检查](../validation/ldd.log)确认 OpenCV/libstdc++ 来自系统，RKNN/RGA 来自现有 SDK；无 Conda、ONNX Runtime、OpenVINO、Drogon 或缺失库。

## 模型与预处理

现有两个模型都通过真实 NPU 推理及重复运行检查。输入为 RGB UINT8/NHWC，模型内含 /255 归一化；输出按 YOLO26 双分支解释，由 Runtime 转为 FP32。

| 模型 | OpenCV 框数 | RGA 框数 | RGA 中找到同类且 IoU > 0.7 的 OpenCV 框 |
| --- | ---: | ---: | ---: |
| FP16 | 8 | 8 | 8/8 |
| INT8 | 3 | 4 | 3/3 |

RGA 还通过非连续 Mat 输入、stride、RGB 通道和 114 填充的像素检查，以及输入租约禁止覆盖检查。
FP16 和 INT8 检测差异明显；该样例 INT8 未输出 FP16 中的 bus 框，因此默认继续使用 FP16。以上匹配仅为部署对照，不是带标注数据的精度验收。

- [FP16 标注图片](../validation/images/fp16-rga.jpg) / [原始结果](../validation/images/fp16-rga.jsonl)
- [INT8 标注图片](../validation/images/int8-rga.jpg) / [原始结果](../validation/images/int8-rga.jsonl)

## 3000 帧运行

使用 320×240、30 FPS 的重复 bus 合成视频，默认 FP16/RGA、异步、动态抽帧；无标注视频编码开销。最终版本复用同一输入完成复测。

| 项目 | 实测 |
| --- | ---: |
| 输出帧数 | 3000 |
| 推理次数 | 273 |
| 异步提交 / 应用 / 丢弃 | 272 / 272 / 0 |
| 未消费任务峰值 / 结束剩余 | 1 / 0 |
| 运动历史峰值 | 64 帧 |
| 动态步长范围 / 最终步长 | 2–10 / 8 |
| 总处理时间 | 50.89 秒 |
| 处理 FPS（包含 LK 帧） | 58.95 |
| 平均输入设置＋NPU＋输出获取 | 161.31 ms/次 |
| 进程平均 CPU 使用率 | 187.27% |
| 进程峰值 RSS | 155.13 MiB |
| 初始化后首 / 尾观测区间平均 RSS | 142.58 / 141.40 MiB |
| RSS 增长 | -1.19 MiB |
| NPU 温度峰值 | 40.69 °C |

未出现随帧数持续增长的 RSS；队列、历史、帧号、EOF 检查通过。处理 FPS 包含大量 LK 帧，**不能解释为模型每秒推理次数**。
记录了 256 次 NPU load 采样；[完整报告](../validation/soak/report.json)、[JSONL](../validation/soak/results.jsonl)、[运行日志](../validation/soak/stderr.log)保留在工程内。

复现命令见 [README](README.md)，可使用 `tools/soak_test.py --input validation/soak/input.avi` 复用已生成视频。

## 验证边界

- 已验证：编译、实际 RGA/NPU 运行、合成场景的算法和异步契约、媒体输出、单次 3000 帧内存表现。
- 未验证：工业数据 mAP/IDF1、真实遮挡与独立运动目标的长延迟补偿、指定吞吐量/延迟目标、长期稳定性、摄像头/RTSP/MPP 链路。
- OpenCV→DMA 存在复制，RKNN 使用标准输入接口；未实现或宣称端到端零拷贝。
- 未修改模型、SDK、内核、驱动、引导链、设备权限或系统服务；原工程备份为 `../yolo_run.before-rknn.tar.gz`。


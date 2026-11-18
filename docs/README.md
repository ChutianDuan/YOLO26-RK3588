# 项目文档索引

核对日期：2026-11-18（Asia/Shanghai）。当前说明以工作区源码、配置和构建入口为准；未提交的新功能也计入本次核对。

| 文档 | 用途 | 对照来源 |
|---|---|---|
| [项目入口](../README.md) | 功能、构建、四个程序与能力边界 | CMake、各程序 `--help`、默认配置 |
| [环境管理](environment.md) | system / vision、现有 SDK、依赖锁和 doctor | `scripts/dev.sh`、`scripts/doctor.py`、`env/` |
| [学习路线](../学习路线.md) | 按调用关系逐站读源码 | 当前 C++ 接口与测试 |
| [架构与所有权](architecture.md) | 采集、并发、DMA-BUF、RKNN、坐标与指标口径 | `yolo_run/` |
| [设计图](design-diagrams.md) | 线程、帧租约、退出与四路对照 | SVG / PNG / Mermaid / 离线图册 |
| [性能与精度记录](performance.md) | 按日期和条件解释已有测量 | 原始报告和 JSON/JSONL |
| [面试复习](interview.md) | 项目介绍、工程问题与证据边界 | 架构与历史测量 |
| [后续路线](development-roadmap.md) | 尚未完成的评测、输入与稳定性工作 | 当前实现与缺失证据 |
| [测试指南](../test/README.md) | CPU / 硬件回归、夹具、基准与长测 | `test/CMakeLists.txt`、`test/tools/` |
| [报告索引](../test/reports/README.md) | 各版本的验收记录及原始证据 | 报告日期与运行条件 |

部署专题：[COCO 640×640](../yolo_run/deploy/COCO640.md) · [实时防抖](../yolo_run/deploy/STABILIZATION.md) · [RTSP/WebRTC 输出](../yolo_run/deploy/STREAMING.md) · [四路本地视频](../yolo_run/video/MULTI_VIDEO.md)。

更新约定：功能默认值从配置和代码核对；测试名称和分组从 CMake 核对。历史报告中的数字、哈希和原始命令对应当时版本，历史命令可能不适用于现有 CLI；重跑使用当前测试指南。`test/results/` 和 `test/validation/` 不是测试发现目录，日期较早也不代表其中基线可删除。

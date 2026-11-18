# 验证报告索引

当前源码说明见 [项目入口](../../README.md)，测试方法见 [测试指南](../README.md)。这里的报告均为**对应日期和版本的记录**：测试项数、CLI、资源数量和性能不能直接当作最新版结论。清理不会重新测量或改写历史数值。

## 最近功能验收

| 报告 | 条件与边界 |
|---|---|
| [2026-11-18 平移防抖](stabilization_validation.md) | 28 项回归、MPP/RGA 抖动回放与 COCO FP16 实机 3000 帧；现场大面积遮挡，主要验证失锁降级 |
| [2026-11-18 DMA-BUF 单路径](dma_only_validation.md) | 26 项历史回归；生产删除 CPU/copy/auto 和压缩包复制，尚未加入后续防抖 |
| [COCO640 FP16](coco640_validation.md) | 双输出 80 类模型、RGA/NPU/回放与推流；不是完整 COCO mAP |
| [Runtime profiling](yolo_perf_validation.md) | 单图采集开关等价与逐层耗时；不是视频端到端吞吐 |

## 仍有用的历史证据

| 报告 | 保留理由 |
|---|---|
| [四路原生预测](multi_prediction_validation.md) | Kalman 时序、调度与 120 秒预测版验证 |
| [四路旧版基线](multi_video_validation.md) | 同文件回放 600 秒对照，不能替代当前预测版长测 |
| [单路推流](streaming_validation.md) | INT8 / 防抖接入前的 30 分钟真实摄像头与客户端反馈 |
| [DMA 生命周期整理](dma_readability_validation.md) | 分配/帧租约/注册和硬件像素；含已删除的 copy 模式历史记录 |
| [LK/RGA 对照](lk_rga_report.md) | 固定/动态帧调度与灰度缩放的条件对照 |
| [初期 FP16/INT8 对照](comparison_report.md) | 单条 BDD100K training 序列的历史精度，非工业独立测试集 |
| [历史指标汇总](metrics.md)、[JSON](metrics.json) | `summarize_metrics.py` 从保留机器记录重算，日期为 2026-10-18 |

三模型性能与同帧精度的其他原始证据入口见 [性能记录](../../docs/performance.md)。历史 tracking 评测的非单位 stride 索引问题见 [后续路线](../../docs/development-roadmap.md)，重跑时应修正适配口径，保留旧报告作对照。

一次性目录迁移/初始搭建报告和旧构建备份不再维护。历史命令中的 `validate_reorganization.py` 现对应 `tools/validate_capture.py`；`verify_migration.py` 已退役。原始报告内部的旧绝对路径通过 `tools/paths.py` 读取时映射，不改写逐帧证据与哈希。

初次环境验收、迁移时的目录映射和迁移当日的短测作为历史记录保留。[初次验收](initial_validation.md) · [目录迁移](migration.md) · [迁移短测](reorganization_validation.md)。

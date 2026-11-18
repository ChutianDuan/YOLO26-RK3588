# 2026-11-18 文档与测试整理记录

板端环境为 RK3588 / aarch64、Linux 6.6.89-kdev。核对基线为 Git 提交 `6ad99f2` 加工作区已有的平移防抖改动。运行入口、默认配置、坐标域和内存交接说明已按该工作区更新。[文档索引](README.md) · [测试指南](../test/README.md)

| 项目 | 本次板端结果 |
|---|---|
| 环境与构建 | `bash scripts/dev.sh doctor` 报告 0 项失败，生产与测试预设均构建通过，4 个正式程序的 `--help` 均可用 |
| 日常回归 | `bash scripts/dev.sh test` 按 `cpu` 标签选择 9 项，9/9 通过 |
| 完整回归 | 9 项纯软件和 19 项硬件测试串行执行，28/28 通过，256.07 秒。[CTest 日志](../test/results/docs-tests-20261118.Ml2uAp6r/ctest-full.log) |
| 摄像头短测 | 正式程序完成 150 帧 YUYV 与 300 帧 MJPEG，像素上传或压缩包复制计数均为 0，序号和时间戳单调，内核增量日志检查通过。[检查结果](../test/results/docs-tests-20261118.Ml2uAp6r/capture/report.json) |
| 离线材料 | 固定评测包通过 SHA256SUMS 校验，评测器 4/4 单元测试通过。历史指标重算后 `metrics.json` 与整理前逐字节一致 |
| LK 对比 | 现行基准默认运行 RGA 的 1.0 和 0.5 尺度。显式请求已删除的 OpenCV 预处理分支时，在建立输出目录前报错。历史结果仍可离线读取 |

已清除 `test/archive/build-before-cleanup-20261018_193700/` 中 63 个受 Git 跟踪的旧构建文件，约 12 MiB。当年的迁移清单、源码快照仍在 `test/archive/`。一次性迁移校验脚本 `verify_migration.py` 已退役。150/300 帧采集回归继续维护，入口由 `validate_reorganization.py` 改为 `validate_capture.py`。

历史对照实验显式关闭防抖，以完整的 800×600 原图对齐标注。已有模型、视频夹具、标注、旧版测量与最新防抖证据均保留，查看性能数字时应使用对应报告的模型、输入与测量窗口。本轮验证覆盖功能回归和短时采集，完整 mAP、长时稳定性与吞吐基准沿用对应日期的报告。[报告索引](../test/reports/README.md)

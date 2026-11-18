# 目录整理与兼容说明

> **历史记录**。下文的接口、命令和结果对应当时版本。当前命令见[测试指南](../README.md)。 旧构建目录已于 2026-11-18 删除，63 个文件可从 Git 提交 `6ad99f2` 找回。

整理日期：2026-10-18（Asia/Shanghai）。根目录主 README 为当前说明；这里记录迁移规则和接口变化。

## 旧新位置

| 整理前 | 整理后 |
|---|---|
| `yolo_run/test_cpp/*.cpp` | `test/cpp/`；ioctl 故障支持进入 `test/support/` |
| `yolo_run/test/*.py` | `test/python/` |
| `yolo_run/tools/` | `test/tools/` |
| `yolo_run/test/*.md`、`VALIDATION.md` | `test/reports/`，首次验收更名 `initial_validation.md` |
| `yolo_run/validation/` | `test/validation/`，内部结构不变 |
| `Camera/` | `test/camera/`，报告、压缩包及样张全部保留 |
| 原 `yolo_run/build-rk3588/` | `test/archive/build-before-cleanup-20261018_193700/` |
| 原生产/测试混合构建 | 正式程序仍在 `yolo_run/build-rk3588/`；测试工程统一到 `test/build-rk3588/bin/` |

迁移前源码保存在 [source-before-cleanup-20261018_193700.tar.gz](../archive/source-before-cleanup-20261018_193700.tar.gz)。旧构建包含当时的绝对路径，现可从本页开头列出的 Git 提交查阅。完整逐文件旧路径、新路径、字节数、SHA-256 见 [migration.json](../archive/migration.json)。

当时的 JSON、JSONL、日志和媒体保留原值。它们对应各自历史版本，不代表最新能力清单。当前入口使用新 README 与指标报告；历史文件内部断开的跨目录旧链接可通过迁移清单查找，工具通过 `paths.resolve_artifact` 在读取时重定位，**不重写历史文件**。

## 运行变化

`camera_run` 不再消耗 8 个启动对比帧，也不进行 RGB checksum 抽样读取。像素对比、114 补边验证和变化观察在独立 `camera_validation` 中执行。正常运行仍保留资源统计、实际格式/stride/容量检查、Cache Sync、租约和设备完成保护。

`auto` 只在 DMA-BUF 采集会话构造/初始化失败时，结束原会话并重建 MMAP。强制 `dmabuf` 仍失败退出；RGA/NPU 运行失败不会用自动切换掩盖。

摄像头正式 JSONL 不再产生：`validation_frames`、`validation_upload_bytes`、`validation_rgb_read_bytes`、`diagnostic_read_bytes`、`checksum_samples`、`checksum_changes`，以及 resource 的 `rgb_checksum`。新的诊断结果在测试输出中独立保存，旧记录的这些字段仍可读取。FPS、延迟、资源、上传字节及截图字段保留。

`LiveOptions` 移除故障配置字段；流水线通过三个可选内部回调连接测试支持。正式 `camera_live` 不接受 `--test-*`，延迟、乱序模拟和写出失败仅在 `test/support/live_fault_main.cpp` 实现。回放仍为正式输入功能，夹具生成和录制为测试工具。

## 构建与工具变化

在 `yolo_run` 运行生产 Preset；在 `test` 运行构建及 CTest Preset。正式工程不再注册 CTest。测试工程直接复用正式源码，不复制实现。

Python 实验工具统一用 `--build-dir BIN_DIRECTORY` 选择二进制；默认 `test/build-rk3588/bin`。`soak_test` 保留旧 `--build` 别名。实验输出目录必须新建；离线报告重算使用 `--from-dir` 指定历史输入并将 `--output-dir` 指向新目录，禁止覆盖证据。LK 的离线合并保留来源路径；不会把当前二进制哈希冒充历史测量哈希。

`camera/` 中与历史报告绑定的 `collect_modes.py`、`generate_report.py` 一并原样归档，只作为当时方法的证据，不从原归档目录直接重跑。它们会写入相邻报告文件，需另行复制到新实验目录、核对当时配置后使用。

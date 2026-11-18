# YOLO 环境管理

核对日期：2026-11-18（Asia/Shanghai）。

本项目保留当前可用的系统工具链、Rockchip SDK 和 `rk3588-infer`。新终端只初始化 Conda 命令，项目环境通过下列入口显式选择；root 和 admin 使用相同的入口。命令从项目根目录执行，或使用 `/home/yolo/scripts/dev.sh` 的绝对路径。

root 和 admin 均可执行环境检查、进入项目 shell 和运行对应 Python。构建仍使用已有目录权限；当前仓库及构建目录由 root 所有，构建命令应由有写权限的账号执行。

```bash
bash /home/yolo/scripts/dev.sh doctor
bash /home/yolo/scripts/dev.sh shell
bash /home/yolo/scripts/dev.sh python -c 'import sys; print(sys.executable)'
bash /home/yolo/scripts/dev.sh --role system python test/tools/summarize_metrics.py --help
bash /home/yolo/scripts/dev.sh build
bash /home/yolo/scripts/dev.sh test
```

## 两种角色

| 角色 | 解释器与用途 | 进入方式 |
|---|---|---|
| `vision`（默认） | `/opt/miniforge3/envs/rk3588-infer/bin/python`，Python 3.12.15；保留 Python 视觉/板端 RKNN Lite 开发能力 | `bash scripts/dev.sh shell` 或 `bash scripts/dev.sh python ...` |
| `system` | `/usr/bin/python3`，Python 3.13.3；发布器、现有板端评测和 CTest Python 脚本使用标准库 | `bash scripts/dev.sh --role system shell` 或 `bash scripts/dev.sh --role system python ...` |

`--role` 只用于 `shell` / `python`，必须放在子命令前。`doctor` 始终检查两种角色，`build` / `test` 始终采用系统 C++ 工具链；对这三个子命令传 `--role` 会报错。

`shell` 创建独立的子 Bash，将工作目录切换到项目根目录，并显示 `(yolo:vision)` / `(yolo:system)` 提示符。输入 `exit` 返回原终端；父终端环境不变。子 Bash 使用项目专用启动文件，不读取用户的 `.bashrc`。`vision` 会执行现有 Conda 初始化并激活指定 prefix，`system` 使用系统 PATH 且移除继承的 Conda 激活元数据。

`python` 透传全部 Python 参数及退出码，工作目录为项目根目录。例如，相对脚本路径始终以 `/home/yolo` 为基准；执行 `-c`、`-m` 与直接脚本都遵守普通 Python 行为。入口清理 `PYTHONHOME` / `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1`，保留项目目录导入能力。`vision` 的解释器、`sys.prefix`、`CONDA_PREFIX` 和 PATH 来自同一激活环境。

切换到 Agent_LLM 项目时，先退出此项目子 shell，再使用 Agent_LLM 自己的管理入口。两个项目的 Python 依赖继续各自维护。

## C++ 与 SDK

| 组件 | 当前来源 |
|---|---|
| G++ / CMake / Ninja | `/usr/bin/g++` 14.2.0、`/usr/bin/cmake` 3.31.6、`/usr/bin/ninja` 1.12.1；C++17，构建并发 4 |
| RKNN Runtime | `/opt/rknn/runtime-2.5.0`，2.5.0 |
| RGA / MPP | `/opt/rockchip/librga` 1.10.6、`/opt/rockchip/mpp` 现有固定提交构建 |
| SDK CMake targets | `/opt/rockchip/cmake/RockchipConfig.cmake` 的 `Rockchip::RKNN` / `Rockchip::RGA` / `Rockchip::MPP` |
| C++ OpenCV / FFmpeg | 系统 APT OpenCV 4.10.0、FFmpeg 7.1.1 及对应开发库 |
| MediaMTX | `tools/mediamtx/v1.21.2/mediamtx`，项目内独立 ARM64 程序 |

`build` 调用 `yolo_run` 现有 `rk3588-release` 配置和构建预设，产物位置仍是 `yolo_run/build-rk3588/`。入口在子进程清理 Conda、编译器、头文件和动态库路径覆盖，明确使用 `/usr/bin/cmake`；现有 CMake Presets 再固定系统编译器和依赖搜索路径。C++ 继续使用系统 OpenCV/libstdc++，Conda 的 headless OpenCV 只服务 Python。

`test` 先配置、构建 `test` 的同名预设，再通过 CTest `cpu` 标签串行运行九项回归：`historical_paths`、`split_benchmark_accounting`、`live_core`、`coco_core`、`stabilization_core`、`multi_scheduler`、`multi_recorder`、`multi_prediction`、`stream_publisher`。这些测试不调用 NPU、摄像头、RGA/MPP 硬件，不启动推流服务；发布器回归使用本地 Python 模拟子进程。测试产物仍在 `test/build-rk3588/`。完整硬件回归与长测继续按 [测试指南](../test/README.md) 手动执行，需要独占相应设备。

## 当前依赖规格与精确锁

后续环境管理以项目内三件套为准：

- [env/environment.yml](../env/environment.yml)：人工维护的 Conda 层意图；环境名 `rk3588-infer`，不包含机器专用 `prefix`。
- [env/conda-linux-aarch64.explicit.lock](../env/conda-linux-aarch64.explicit.lock)：当前 26 个 Conda 包的准确下载地址与 SHA256，固定 Linux aarch64 和现有 Python 3.12.15 构建。
- [env/requirements-linux-aarch64.lock](../env/requirements-linux-aarch64.lock)：当前 5 个 pip 包及 wheel SHA256；`numpy` 2.5.4、`opencv-python-headless` 5.0.0.93、`psutil` 7.2.2、`rknn-toolkit-lite2` 2.5.0、`ruamel.yaml` 0.19.1。

`packaging`、`setuptools` 和 `wheel` 由 Conda 管理，只出现在 Conda 锁中。旧 pip freeze 中的 `packaging @ file:///home/conda/...` 是上游打包时遗留的本地构建路径，当前文件不存在；不能作为 pip 重建来源。

本机现有离线包保留在 `/opt/miniforge3/pkgs`（26 个 Conda 归档）与 `/home/admin/rk3588-workspace/downloads/wheels`（5 个 pip wheels）。整理时均已按 SHA256 核对；大文件继续留在原缓存目录。未来需要独立重建环境时，先使用 Conda 精确锁，再用该 wheelhouse 与 pip 锁，配合 `--no-index --find-links <wheelhouse> --require-hashes --only-binary=:all:` 安装 pip 层。单独的 `environment.yml` 不代表完整精确重建。

以下是日后重建到新名字的示例，日常构建/测试不会执行。命令使用本机离线缓存，原 `rk3588-infer` 继续保留；管理入口仍指向原环境。

```bash
cd /home/yolo
/opt/miniforge3/bin/conda create --offline --name yolo-vision-rebuild \
  --file env/conda-linux-aarch64.explicit.lock
/opt/miniforge3/bin/conda run --no-capture-output --name yolo-vision-rebuild \
  python -m pip install --no-index \
  --find-links /home/admin/rk3588-workspace/downloads/wheels \
  --require-hashes --only-binary=:all: -r env/requirements-linux-aarch64.lock
```

日常继续管理现有环境。将来有意修改依赖时，应同时更新人工规格和对应精确锁，再运行 `doctor` 核对；不要用整个 Conda 环境的 `pip freeze` 覆盖 pip 层锁。

## doctor、历史记录与回退

`doctor` 核对两种 Python 的路径/版本/导入、vision 的 `pip check`、当前安装元数据与锁、可用的本地 wheel 哈希、系统工具和 SDK 文件、MediaMTX 版本、CMake 预设及已有生产程序的 `ldd`。检查不加载模型、不执行推理、不打开摄像头、不启动服务。系统 Python 当前没有 pip，相关检查会明确显示 `SKIP`，这是板端标准库脚本角色的预期状态。未生成生产二进制时，`ldd` 显示 `SKIP`，其余检查仍执行。

[环境配置.md](../环境配置.md) 已收敛为本指南入口。初始搭建日志留在 `/home/admin/rk3588-workspace/artifacts` 和 `/opt/rockchip/artifacts`；历史目录中的环境导出不作为当前安装命令。有效测试证据、模型和评测数据路径保留；过时构建归档与一次性迁移校验的清理见 [整理记录](maintenance-20261118.md)。

项目脚本只影响自身子进程，退出子 shell 即可返回原终端。全局终端策略的修改前备份为 `/etc/profile.d/rk3588-conda.sh.before-project-env-20261023`；需要恢复旧自动激活策略时，需按项目系统变更授权规则由管理员处理。Conda 环境与 SDK 本身未随入口调整发生变化。

# YOLO C++ Runtime profiling 板端验证

2026-11-06 11:09（北京时间），RK3588 aarch64 / Linux 6.6.89-kdev，系统 GCC/CMake/Ninja，
RKNN Runtime 2.5.0（74ea630ce8）、driver 0.9.8、系统 OpenCV 4.10.0。
新增 [yolo_perf_test](../cpp/yolo_perf_test.cpp)，通过 [YoloEngine](../../yolo_run/model/yolo_engine.h)
显式开启详细采集；生产调用默认 flags=0，且不查询性能。

原始证据位于 [results/yolo_perf_20261106.VJkFyPDV/final](../results/yolo_perf_20261106.VJkFyPDV/final/)。
其中 `manifest.json` 保存命令、二进制/三模型/图片 SHA256、开始时间及前后设备快照；
每模型保存 `report.json`、`perf_detail.txt`、从原表提取的 `layers.json` 和 stdout/stderr。
`results` 按现有规则不加入 Git；本报告保留关键数值。

测试使用已安装的官方 YOLO26 `bus.jpg`，RGA 生成固定 RGB UINT8/NHWC 输入，
`pass_through=0`，保留当前 letterbox 和双输出 FP32 后处理契约。
每模型先关闭采集、再启用采集，Context 串行创建和销毁；每阶段预热 3 次、测量 20 次，
另加 1 次不计时的推理获取该帧 PERF_RUN，启用采集时同时复制 PERF_DETAIL。
`core=default`，未显式绑定核心；本次逐层工作量表显示 NPU 算子只在 Core0。

| 模型 | 关闭采集完整 infer 均值 ms | P95 ms | 固定输入串行 FPS | 开启采集 infer 均值 ms | 采集单帧 PERF_RUN ms |
|---|---:|---:|---:|---:|---:|
| FP16 | 183.934 | 184.205 | 5.437 | 201.654 | 195.936 |
| INT8 | 44.577 | 45.590 | 22.433 | 58.293 | 59.039 |
| Hybrid | 49.702 | 50.033 | 20.120 | 63.721 | 60.293 |

FPS 包含完整 infer、后处理、输出释放和结果保留，排除图片读取、预处理、初始化、
预热及性能查询；这是单 Context 固定输入基线，不是视频端到端吞吐。
本次开启采集后均值分别增加约 9.6%、30.8%、28.2%，是当前短测观察值。
额外诊断帧与计时循环不是同一组帧，不能用单帧 PERF_RUN 直接减均值来算开销。

逐层诊断发现：

- FP16 的 CPU `InputOperator:images` 为 **99.477 ms**，占该帧算子汇总约 **52.35%**，
  是最慢的单个算子；`ConvExSwish` 合计 54.710 ms。RGA 已在计时前完成，
  所以该输入算子耗时属于 Runtime 图内处理。结合当前 UINT8 输入及 `pass_through=0`，
  可进一步检查输入转换路径，但尚未分离类型转换、归一化、布局和搬运各自的耗时。
- INT8 的 `ConvExSwish` 合计 **31.890 ms（59.61%）**；两个 `exSDPAttention`
  合计 **6.070 ms（11.35%）**。最慢两个 Attention 分别为 3.059 / 3.011 ms。
- Hybrid 的 `ConvExSwish` 合计 **25.642 ms（47.67%）**，两个 Attention 合计
  **5.699 ms（10.59%）**；26 次 `exDataConvert` 合计 **2.732 ms（5.08%）**。
  转换成本是后续分析混合精度图时可关注的项目，本次不修改模型或转换策略。

上述百分比来自各自开启采集的单帧算子表，不直接推算关闭采集后的节省比例。

| 采集单帧算子表 | CPU 汇总 ms | NPU 汇总 ms | 表内总和 ms |
|---|---:|---:|---:|
| FP16 | 100.993 | 89.026 | 190.019 |
| INT8 | 0.862 | 52.638 | 53.500 |
| Hybrid | 0.997 | 52.798 | 53.795 |

`RKNN_QUERY_PERF_RUN` 是 Runtime 运行时间，可能包含图内 CPU 算子，不能标成纯 NPU 时间；
算子表汇总和 PERF_RUN 的统计范围也有差别。程序因此使用 `diagnostic_runtime_run_ms`
和 `Runtime run` 标识该指标。

验证结论：

- **编译成功**：原生 Release C++17，构建并发度 4；链接现有 `/opt/rknn`、`/opt/rockchip`
  SDK 及系统 OpenCV/libstdc++，未链接 Conda C++ 库。
- **运行成功**：三个模型返回有效 PERF_RUN 与非空 PERF_DETAIL（分别 50800 / 46858 / 56361 字节）。
  相关 CTest **10/10** 通过：`stream_publisher`、`historical_paths`、`split_benchmark_accounting`、
  `live_core`、`live_npu`、`core`、`hardware`、`yolo_perf`、`multi_scheduler`、`multi_prediction`。
  指标名称澄清后的最终 `yolo_perf` 再次通过；另检查 9 组非法 CLI / 已存在输出目录均被拒绝。
- **结果一致性通过**：同模型关闭/开启采集及重复运行的数量、类别、次序、分数、坐标完全一致，
  最大坐标/分数差均为 0。FP16 / INT8 / Hybrid 各为 8 / 4 / 10 个检测，未断言跨精度模型等价，
  未运行标注集精度评测。
- **性能达标未判定**：未设性能阈值；未测视频链路、多 Worker、多路或长时稳定性。
  仅采样前后 NPU 温度 37.000 / 38.846°C、频率均为 1 GHz，不能当作全程峰值或锁频证明。

查询位于 `outputs_get` 后、`outputs_release` 前且保持 Context 锁；立即复制 Runtime 所有的
`perf_data`，不释放 SDK 指针。输入帧租约、DMA-BUF 分配/注册、输出缓冲沿用现有 RAII。
未修改 SDK、权限、驱动、频率策略或服务，无新增系统配置风险；测试期间使用 RGA/NPU。

复测命令及参数边界见 [测试说明](../README.md#runtime-profiling)。

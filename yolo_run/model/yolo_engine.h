#pragma once
#include <memory>
#include <string>
#include "model/inference_types.h"
#include "config/app_config.h"

namespace yolo {
struct NpuPerformance {
    int64_t run_duration_us = 0; // RKNN_QUERY_PERF_RUN：Runtime 运行时间，微秒，可含图内 CPU 算子。
    std::string detail;         // 仅启用 collect_perf 时提供；用于诊断，不用于正式吞吐测量。
};
// 一个模型 context 串行执行；infer() 内部加锁，不与其他 engine 共享 context 或注册对象。
class YoloEngine {
public:
    // -1 preserves Runtime defaults; otherwise an explicit rknn_core_mask.
    // collect_perf 会降低运行性能；默认不启用详细性能采集。
    explicit YoloEngine(const AppConfig& config, int core_mask = -1, bool collect_perf = false);
    ~YoloEngine();
    YoloEngine(const YoloEngine&) = delete;
    YoloEngine& operator=(const YoloEngine&) = delete;
    // 输入及其帧租约需覆盖整个调用；返回框坐标已经恢复到原图尺寸。
    // performance 非空才查询；其内容仅在 infer() 成功返回后有效，未启用采集时 detail 为空。
    InferResult infer(const TensorInput& input, int64_t frame_index = 0, NpuPerformance* performance = nullptr);
    // 构造完成后布局只读；将它传给 ImagePreprocessor，避免只按可见 RGB 大小分配。
    InputMemoryLayout inputMemoryLayout() const;
    size_t maxConcurrency() const { return 1; }
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace yolo

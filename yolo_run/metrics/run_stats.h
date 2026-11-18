#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <sys/resource.h>
#include "model/inference_types.h"

namespace yolo {
using Clock = std::chrono::steady_clock;
inline double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct TimingAggregate {
    uint64_t count = 0;
    double total_ms = 0, max_ms = 0;
    void add(double value) { ++count; total_ms += value; max_ms = std::max(max_ms, value); }
    double mean() const { return count ? total_ms / count : 0; }
};
struct RunStats {
    uint64_t frames = 0, inference_count = 0;
    uint64_t async_submissions = 0, async_corrections = 0, discarded_results = 0;
    uint64_t skipped_detections = 0, forced_detections = 0;
    size_t max_pending = 0, history_peak = 0, pending_at_end = 0;
    int min_stride = 1, max_stride = 1, final_stride = 1;
    double source_fps = 0, elapsed_ms = 0, cpu_percent = 0, peak_rss_mb = 0;
    TimingAggregate decode, preprocess, infer, postprocess, flow, tracker, queue_wait;
    int flow_width = 0, flow_height = 0;
    TimingAggregate flow_prepare, flow_motion, flow_compensation;
    std::array<TimingAggregate, 6> flow_object;
    void add(const InferResult& r) {
        ++inference_count;
        preprocess.add(r.preprocess_ms); infer.add(r.infer_ms);
        postprocess.add(r.postprocess_ms); queue_wait.add(r.queue_wait_ms);
    }
};
struct UsageTimer {
    Clock::time_point start = Clock::now();
    rusage initial{};
    UsageTimer() { getrusage(RUSAGE_SELF, &initial); }
    void finish(RunStats& stats) const {
        rusage end{};
        getrusage(RUSAGE_SELF, &end);
        auto seconds = [](const rusage& r) {
            return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
        };
        stats.elapsed_ms = elapsedMs(start);
        stats.cpu_percent = stats.elapsed_ms > 0 ? (seconds(end) - seconds(initial)) * 100000 / stats.elapsed_ms : 0;
        stats.peak_rss_mb = end.ru_maxrss / 1024.0;
    }
};
}  // namespace yolo

#pragma once
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include "model/inference_types.h"
#include "video/optical_flow_tracker.h"

namespace yolo {
using InferFunction = std::function<InferResult(const cv::Mat&, int64_t)>;
struct AsyncResult {
    int64_t frame_index = -1;
    InferResult result;
    std::exception_ptr error;
};
// Exactly one request, in-flight inference OR unconsumed result. No extra thread per frame.
class AsyncInferWorker {
public:
    explicit AsyncInferWorker(InferFunction infer);
    ~AsyncInferWorker();
    bool submit(const cv::Mat& frame, int64_t index);
    std::optional<AsyncResult> pop(bool wait = false);
    bool pending() const;
private:
    void run();
    InferFunction infer_;
    mutable std::mutex mutex_;
    std::condition_variable request_cv_, result_cv_;
    bool stopping_ = false, pending_ = false, request_ = false;
    cv::Mat frame_;
    int64_t index_ = -1;
    std::chrono::steady_clock::time_point submitted_;
    std::optional<AsyncResult> result_;
    std::thread thread_;
};

class MotionHistory {
public:
    static constexpr size_t capacity = 64;
    void push(FrameMotion motion);
    size_t size() const { return frames_.size(); }
    std::optional<std::vector<Detection>> compensate(const std::vector<Detection>& detections,
        int64_t source, int64_t current, int width, int height) const;
private:
    std::deque<FrameMotion> frames_;
};

class DetectionPolicy {
public:
    DetectionPolicy(double source_fps, float detect_fps, bool dynamic);
    void observe(const WeakTrackQuality& quality, const std::vector<TrackedDetection>& before,
                 const std::vector<TrackedDetection>& after);
    bool due(int64_t frame) const { return last_submission_ < 0 || frame - last_submission_ >= current; }
    void submitted(int64_t frame) { last_submission_ = frame; }
    int base = 1, current = 1, minimum = 1, maximum = 1;
private:
    bool dynamic_;
    int stable_count_ = 0;
    int64_t last_submission_ = -1;
};
}  // namespace yolo

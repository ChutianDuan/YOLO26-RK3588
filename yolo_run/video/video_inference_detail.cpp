#include "video/video_inference_detail.h"
#include "metrics/run_stats.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace yolo {
AsyncInferWorker::AsyncInferWorker(InferFunction infer) : infer_(std::move(infer)) {
    // 每条视频只创建一个固定线程，避免按帧创建线程或让多个推理同时占用同一缓冲。
    thread_ = std::thread(&AsyncInferWorker::run, this);
}
AsyncInferWorker::~AsyncInferWorker() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    request_cv_.notify_one();
    if (thread_.joinable()) thread_.join(); // 等执行中的回调结束，再允许外层销毁 engine/预处理器。
}
bool AsyncInferWorker::submit(const cv::Mat& frame, int64_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    // pending 覆盖“待执行、执行中、结果未消费”整个周期，容量始终最多一个请求。
    if (stopping_ || pending_) return false;
    frame.copyTo(frame_); // 复用 Worker 自有 Mat；复制后解码器可安全覆盖原 frame。
    index_ = index;
    submitted_ = Clock::now();
    pending_ = request_ = true;
    request_cv_.notify_one();
    return true;
}
bool AsyncInferWorker::pending() const { std::lock_guard<std::mutex> lock(mutex_); return pending_; }
std::optional<AsyncResult> AsyncInferWorker::pop(bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait) result_cv_.wait(lock, [this] { return result_.has_value() || !pending_; });
    if (!result_) return std::nullopt;
    auto result = std::move(result_);
    result_.reset();
    // 结果取走后才开放下一次提交，防止旧结果被新任务覆盖。
    pending_ = false;
    return result;
}
void AsyncInferWorker::run() {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_);
        request_cv_.wait(lock, [this] { return stopping_ || request_; });
        if (!request_) return;
        request_ = false;
        const auto index = index_;
        const double queue_ms = elapsedMs(submitted_);
        // 在锁外执行耗时检测；pending 仍为 true，submit 无法覆盖 Worker 正在读取的 frame_。
        lock.unlock();
        AsyncResult value;
        value.frame_index = index;
        try {
            value.result = infer_(frame_, index);
            if (value.result.frame_index != index) throw std::runtime_error("inference returned a mismatched source frame");
            value.result.queue_wait_ms = queue_ms;
        } catch (...) { value.error = std::current_exception(); } // 将异常交回视频主线程统一终止任务。
        lock.lock();
        result_ = std::move(value);
        lock.unlock();
        result_cv_.notify_one();
    }
}

void MotionHistory::push(FrameMotion motion) {
    // 只保留有限连续历史；旧检测无法覆盖这段历史时宁可丢弃，也不无界缓存整段视频。
    if (!frames_.empty() && motion.frame_index != frames_.back().frame_index + 1) frames_.clear();
    frames_.push_back(motion);
    if (frames_.size() > capacity) frames_.pop_front();
}
std::optional<std::vector<Detection>> MotionHistory::compensate(const std::vector<Detection>& detections,
    int64_t source, int64_t current, int width, int height) const {
    if (source < 0 || current < source || current - source > int64_t(capacity)) return std::nullopt;
    int64_t expected = source + 1;
    float dx = 0, dy = 0;
    // 逐帧累积平移补偿；它不是重新检测，也不保证独立运动目标在长延迟下完全准确。
    for (const auto& m : frames_) {
        if (m.frame_index <= source || m.frame_index > current) continue;
        if (m.frame_index != expected || !m.valid || !std::isfinite(m.dx) || !std::isfinite(m.dy)) return std::nullopt;
        ++expected;
        dx += m.dx; dy += m.dy;
    }
    if (expected != current + 1) return std::nullopt;
    std::vector<Detection> output;
    output.reserve(detections.size());
    for (auto d : detections) {
        d.x1 = std::clamp(d.x1 + dx, 0.0F, float(width));
        d.x2 = std::clamp(d.x2 + dx, 0.0F, float(width));
        d.y1 = std::clamp(d.y1 + dy, 0.0F, float(height));
        d.y2 = std::clamp(d.y2 + dy, 0.0F, float(height));
        if (d.x2 > d.x1 && d.y2 > d.y1) output.push_back(d);
    }
    return output;
}

DetectionPolicy::DetectionPolicy(double fps, float target, bool dynamic) : dynamic_(dynamic) {
    if (std::isfinite(fps) && fps > 0 && target > 0)
        base = std::max(1, int(std::ceil(std::min(100000.0, fps / target))));
    current = minimum = maximum = base;
}
void DetectionPolicy::observe(const WeakTrackQuality& quality, const std::vector<TrackedDetection>& before,
                             const std::vector<TrackedDetection>& after) {
    if (!dynamic_ || base <= 1) return;
    std::unordered_map<int, Detection> previous;
    for (const auto& t : before) previous.emplace(t.track_id, t.detection);
    size_t changes = 0, matches = 0;
    bool abrupt = false;
    for (const auto& t : after) {
        const auto it = previous.find(t.track_id);
        if (it == previous.end()) { ++changes; continue; }
        ++matches;
        const auto& a = it->second;
        const auto& b = t.detection;
        const float diagonal = std::max(10.0F, std::hypot(a.x2 - a.x1, a.y2 - a.y1));
        const float displacement = std::hypot((b.x1 + b.x2 - a.x1 - a.x2) / 2, (b.y1 + b.y2 - a.y1 - a.y2) / 2);
        const float area = (a.x2 - a.x1) * (a.y2 - a.y1);
        const float ratio = area > 1 ? (b.x2 - b.x1) * (b.y2 - b.y1) / area : 1;
        abrupt |= displacement > std::max(12.0F, diagonal * 0.35F) || ratio < .60F || ratio > 1.60F;
    }
    changes += before.size() - matches;
    if (isSevereWeakQualityDrop(quality) || abrupt || changes >= std::max<size_t>(2, before.size() / 3 + 1)) {
        current = std::min(base, 2); stable_count_ = 0;
    } else if (isComplexWeakQuality(quality)) {
        current = std::min(base, 3); stable_count_ = 0;
    } else if (isStableWeakQuality(quality)) {
        ++stable_count_;
        current = stable_count_ >= 3 ? std::min(std::max(base, 10), base + (base >= 6 ? 4 : 2)) : base;
    } else { current = base; stable_count_ = 0; }
    minimum = std::min(minimum, current);
    maximum = std::max(maximum, current);
}
}  // namespace yolo

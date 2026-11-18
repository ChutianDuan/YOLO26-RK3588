#include "video/video_inference.h"
#include "image/image_processing.h"
#include "tracking/byte_tracker.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace yolo {
RunStats inferVideoFile(const std::filesystem::path& path, const AppConfig& config,
                       const InferFunction& infer, const FrameSink& sink) {
    UsageTimer timer;
    RunStats stats;
    cv::VideoCapture capture(path.string());
    if (!capture.isOpened()) throw std::runtime_error("cannot open video: " + path.string());
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("video has no valid source FPS");
    stats.source_fps = fps;
    DetectionPolicy policy(fps, config.video_detect_fps, config.video_stride_mode == "dynamic");
    std::unique_ptr<AsyncInferWorker> worker;
    ByteTracker tracker;
    MotionHistory history;
    std::vector<TrackedDetection> tracks;
    cv::Mat frame, next;
    TrackingPreprocessor flow_preprocessor(config);
    TrackingImage previous_gray, gray;
    std::vector<cv::Mat> previous_pyramid, current_pyramid;
    // OpenCV 解码得到 CPU BGR Mat；检测和 LK 各用自己的预处理器/DMA 缓冲。
    // 灰度固定双槽保存前后帧；金字塔可能引用灰度内存，必须先释放视图再释放读租约。
    auto read = [&](cv::Mat& target) {
        const auto t = Clock::now();
        const bool ok = capture.read(target);
        if (ok && !target.empty()) stats.decode.add(elapsedMs(t));
        return ok && !target.empty();
    };
    if (!read(frame)) throw std::runtime_error("video has no readable frames");
    const auto size = frame.size();
    bool refresh = false;
    int64_t index = 0;
    while (true) {
        if (frame.size() != size || frame.type() != CV_8UC3)
            throw std::runtime_error("video frame format or dimensions changed");
        FrameResult output;
        output.frame_index = index;
        output.timestamp_ms = index * 1000.0 / fps;
        output.width = frame.cols; output.height = frame.rows;
        const auto tracking_start = Clock::now();
        double flow_ms = 0, tracker_ms = 0;
        // 每帧准备跟踪灰度图：RGA 输出 NV12，CPU 直接通过带 stride 的 Mat 读取 Y 平面。
        gray = flow_preprocessor.prepare(frame);
        stats.flow_prepare.add(elapsedMs(tracking_start));
        stats.flow_width = gray.gray.cols; stats.flow_height = gray.gray.rows;
        double compensation_ms = 0;
        WeakTrackResult weak;
        FrameMotion motion;
        motion.frame_index = index;
        if (index > 0) {
            // 检测间隔内由 CPU LK 估计目标/背景运动，再维持 ByteTrack 的轨迹和 ID。
            const auto before = tracks;
            auto tracked = trackFlowFrame(previous_gray.gray, gray.gray, tracks, frame.size(), index,
                                          previous_pyramid, current_pyramid);
            weak = std::move(tracked.weak);
            motion = tracked.motion;
            stats.flow_motion.add(tracked.motion_ms);
            flow_ms += elapsedMs(tracking_start);
            const auto byte_start = Clock::now();
            tracks = tracker.updateTracked(weak.tracks, index);
            tracker_ms += elapsedMs(byte_start);
            policy.observe(weak.quality, before, tracks);
        } else { flow_ms += elapsedMs(tracking_start); stats.flow_motion.add(0); }
        for (size_t phase = 0; phase < weak.stage_ms.size(); ++phase)
            stats.flow_object[phase].add(weak.stage_ms[phase].value_or(0));
        history.push(motion);
        stats.history_peak = std::max(stats.history_peak, history.size());

        auto apply = [&](InferResult result, bool asynchronous) {
            stats.add(result);
            if (result.frame_index > index || result.frame_index < 0)
                throw std::runtime_error("invalid inference source frame");
            const auto compensation_start = Clock::now();
            // 异步检测框属于旧帧，先用有限运动历史补偿到当前帧，不能直接当作当前坐标使用。
            // 历史过期或运动无效时丢弃结果并请求刷新，避免错误地校正当前轨迹。
            auto corrected = asynchronous ? history.compensate(result.detections, result.frame_index, index, frame.cols, frame.rows)
                                          : std::optional<std::vector<Detection>>(result.detections);
            const auto ms = elapsedMs(compensation_start);
            flow_ms += ms;
            compensation_ms += ms;
            if (!corrected) { ++stats.discarded_results; refresh = true; return; }
            const auto before = tracks;
            const auto t = Clock::now();
            tracks = tracker.update(*corrected, index);
            tracker_ms += elapsedMs(t);
            policy.observe(weak.quality, before, tracks);
            output.is_detection_frame = true;
            output.tracks_source = asynchronous ? "async_corrected" : "detected";
            output.corrected_from_frame_index = asynchronous ? result.frame_index : -1;
            output.detections = std::move(*corrected);
            if (asynchronous) ++stats.async_corrections;
            refresh = false;
        };
        auto poll = [&](bool wait) {
            if (!worker) return;
            auto ready = worker->pop(wait);
            if (!ready) return;
            if (ready->error) std::rethrow_exception(ready->error);
            if (ready->frame_index != ready->result.frame_index) throw std::runtime_error("async result source mismatch");
            apply(std::move(ready->result), true);
        };
        poll(false); // 非阻塞收取已完成检测，主线程继续处理逐帧光流和输出。
        const bool forced = refresh || (config.video_stride_mode == "dynamic" && isSevereWeakQualityDrop(weak.quality));
        if (index == 0 || (!output.is_detection_frame && (forced || policy.due(index)))) {
            if (!worker) {
                auto result = infer(frame, index);
                if (result.frame_index != index) throw std::runtime_error("sync result source mismatch");
                apply(std::move(result), false);
                policy.submitted(index);
                if (forced) ++stats.forced_detections;
                // 首帧同步建立轨迹；此后仅固定 Worker 调用检测链路，不并发使用同一预处理器。
                if (index == 0 && config.video_model_async && policy.base > 1)
                    worker = std::make_unique<AsyncInferWorker>(infer);
            } else if (worker->submit(frame, index)) {
                policy.submitted(index);
                ++stats.async_submissions;
                stats.max_pending = 1;
                if (forced) ++stats.forced_detections;
            } else ++stats.skipped_detections; // Worker 忙时跳过本次检测提交，不丢弃待输出视频帧。
        }
        const bool has_next = read(next);
        // EOF 等待剩余推理；只修正尚未写出的最后一帧，不回填已写出的历史结果。
        if (!has_next) poll(true);
        stats.flow.add(flow_ms);
        stats.flow_compensation.add(compensation_ms);
        stats.tracker.add(tracker_ms);
        output.tracks = tracks;
        if (!output.is_detection_frame) output.tracks_source = tracks.empty() ? "empty" : "lk";
        sink(frame, output);
        ++stats.frames;
        if (!has_next) break;
        // 先销毁旧金字塔视图，再释放对应灰度 DMA READ 租约，才能安全复用双槽缓冲。
        previous_pyramid = std::move(current_pyramid);
        previous_gray = std::move(gray);
        std::swap(frame, next);
        ++index;
    }
    stats.pending_at_end = worker && worker->pending() ? 1 : 0;
    stats.min_stride = policy.minimum; stats.max_stride = policy.maximum; stats.final_stride = policy.current;
    timer.finish(stats);
    return stats;
}
}  // namespace yolo

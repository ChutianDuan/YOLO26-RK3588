#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include "config/app_config.h"
#include "metrics/run_stats.h"
#include "video/video_inference_detail.h"

namespace yolo {
struct FrameResult {
    int64_t frame_index = 0;
    double timestamp_ms = 0;
    int width = 0, height = 0;
    bool is_detection_frame = false;
    std::string tracks_source = "empty";
    int64_t corrected_from_frame_index = -1;
    std::vector<Detection> detections;
    std::vector<TrackedDetection> tracks;
};
// Sink borrows this frame only during the call. Results are never accumulated.
using FrameSink = std::function<void(const cv::Mat&, const FrameResult&)>;
RunStats inferVideoFile(const std::filesystem::path& path, const AppConfig& config,
                       const InferFunction& infer, const FrameSink& sink);
}  // namespace yolo

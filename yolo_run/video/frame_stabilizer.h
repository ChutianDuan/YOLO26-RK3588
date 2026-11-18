#pragma once
#include <array>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include "image/image_processing.h"

namespace yolo {
struct StabilizationInfo {
    cv::Rect roi{0, 0, 800, 600}; // Raw NV12 -> the virtual 800x600 detection/output canvas.
    cv::Point2d delta{}, compensation{}; // Raw-image pixels; positive motion moves ROI right/down.
    bool enabled = false, valid = false, clamped = false, reference_reset = false;
    int sampled_points = 0, valid_points = 0, inliers = 0, background_cells = 0;
    double inlier_ratio = 0, gray_ms = 0, motion_ms = 0, total_ms = 0;
    double mask_source_timestamp_ms = -1, mask_target_timestamp_ms = -1;
};

// CPU estimation only. DMA-backed gray pixels must stay alive through the next update().
// Owned cv::Mat inputs retain their allocation normally; production uses a fixed DMA pair.
class GrayStabilizer {
public:
    explicit GrayStabilizer(double alpha = 0.15);
    StabilizationInfo update(const cv::Mat& gray, double timestamp_ms,
                             const std::vector<TrackedDetection>& previous_tracks = {});
    double previousTimestamp() const { return timestamp_; }
    cv::Rect previousRoi() const { return roi_; }
private:
    double alpha_, timestamp_ = -1;
    cv::Point2d compensation_{};
    cv::Rect roi_{40, 30, 720, 540};
    cv::Mat previous_, mask_;
    std::array<std::vector<cv::Mat>, 2> pyramids_;
    std::vector<cv::Point2f> points_, candidates_, forward_, backward_;
    std::vector<unsigned char> status_, backward_status_;
    std::vector<float> error_, backward_error_, dx_, dy_, scratch_;
    std::vector<size_t> accepted_;
};

// Only the decode thread calls this adapter. RGA downsamples NV12, CPU reads its Y plane.
class FrameStabilizer {
public:
    explicit FrameStabilizer(double alpha = 0.15);
    ~FrameStabilizer();
    StabilizationInfo update(const Nv12FrameView& frame, double timestamp_ms,
                             const std::vector<TrackedDetection>& previous_tracks = {});
    double previousTimestamp() const;
    void finish(); // End the last gray READ after the producer loop has stopped.
    FrameStabilizer(const FrameStabilizer&) = delete;
    FrameStabilizer& operator=(const FrameStabilizer&) = delete;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yolo

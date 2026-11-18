#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>

namespace yolo::multi {
// Output-thread owned. OpenCV/libjpeg encodes reusable JPEG storage; system
// libavformat muxes it into AVI, without a second JPEG encoder or child process.
class MjpegRecorder {
public:
    MjpegRecorder(const std::filesystem::path& path, double fps, cv::Size size);
    ~MjpegRecorder();
    void write(const cv::Mat& bgr, int64_t index);
    void finish();
    MjpegRecorder(const MjpegRecorder&) = delete;
    MjpegRecorder& operator=(const MjpegRecorder&) = delete;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yolo::multi

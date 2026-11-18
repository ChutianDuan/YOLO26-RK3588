#pragma once
#include <filesystem>
#include <fstream>
#include <opencv2/videoio.hpp>
#include "video/video_inference.h"

namespace yolo {
class ResultWriter {
public:
    ResultWriter(const AppConfig& config, const std::filesystem::path& jsonl,
                 const std::filesystem::path& annotated, bool video, double fps = 0);
    void frame(const cv::Mat& image, const FrameResult& result);
    void finish(const RunStats& stats);
private:
    AppConfig config_;
    std::ofstream jsonl_;
    std::filesystem::path annotated_;
    bool video_;
    double fps_;
    cv::VideoWriter video_writer_;
    cv::Mat drawing_;
};
}  // namespace yolo

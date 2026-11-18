#pragma once

#include <string>
#include <vector>

namespace yolo {

struct AppConfig {
    std::string model_path;
    std::string labels_path;
    int input_width = 800;
    int input_height = 608;
    int num_classes = 10;
    std::vector<std::string> class_names;
    float conf_threshold = 0.25F;
    std::vector<float> class_conf_thresholds;
    std::string preprocess = "rga";
    bool use_letterbox = true;
    std::string lk_preprocess = "rga";
    float lk_scale = 1.0F;
    float video_detect_fps = 4.0F;
    bool video_model_async = true;
    std::string video_stride_mode = "dynamic";
    int opencv_threads = 2;
    int inference_timeout_ms = 1000;
};

AppConfig loadAppConfig(const std::string& path);
void validateConfig(const AppConfig& config);

}  // namespace yolo

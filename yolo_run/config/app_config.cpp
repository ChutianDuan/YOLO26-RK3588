#include "config/app_config.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace yolo {
namespace {
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string unquote(std::string s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return nlohmann::json::parse(s).get<std::string>();
    }
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') return s.substr(1, s.size() - 2);
    return s;
}
std::string withoutComment(const std::string& s) {
    char quote = 0;
    bool escape = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char ch = s[i];
        if (escape) { escape = false; continue; }
        if (quote == '"' && ch == '\\') { escape = true; continue; }
        if (quote && ch == quote) quote = 0;
        else if (!quote && (ch == '"' || ch == '\'')) quote = ch;
        else if (!quote && ch == '#') return s.substr(0, i);
    }
    return s;
}
int integer(const std::string& s) {
    size_t end = 0;
    const int value = std::stoi(s, &end);
    if (end != s.size()) throw std::runtime_error("expected integer");
    return value;
}
float number(const std::string& s) {
    size_t end = 0;
    const float value = std::stof(s, &end);
    if (end != s.size() || !std::isfinite(value)) throw std::runtime_error("expected finite number");
    return value;
}
bool boolean(const std::string& s) {
    if (s == "true") return true;
    if (s == "false") return false;
    throw std::runtime_error("expected true or false");
}
void validateModelDimensions(const AppConfig& c) {
    if (c.input_width < 1 || c.input_width > 16384 || c.input_height < 1 || c.input_height > 16384)
        throw std::runtime_error("model input dimensions must be in [1,16384]");
    if (c.num_classes < 1 || c.num_classes > 1000)
        throw std::runtime_error("num_classes must be in [1,1000]");
}
}

void validateConfig(const AppConfig& c) {
    if (std::filesystem::path(c.model_path).extension() != ".rknn")
        throw std::runtime_error("model_path must be a .rknn model");
    validateModelDimensions(c);
    if (!std::isfinite(c.conf_threshold) || c.conf_threshold < 0 || c.conf_threshold > 1)
        throw std::runtime_error("conf_threshold must be in [0,1]");
    if (!c.class_conf_thresholds.empty() && c.class_conf_thresholds.size() != size_t(c.num_classes))
        throw std::runtime_error("class_conf_thresholds must be [] or have num_classes entries");
    for (const float v : c.class_conf_thresholds)
        if (!std::isfinite(v) || (v != -1 && (v < 0 || v > 1)))
            throw std::runtime_error("class thresholds must be -1 or in [0,1]");
    if (c.preprocess != "rga" || c.lk_preprocess != "rga")
        throw std::runtime_error("preprocess and lk_preprocess must be rga; CPU preprocessing was removed");
    if (!std::isfinite(c.lk_scale) || c.lk_scale < 0.25F || c.lk_scale > 1.0F)
        throw std::runtime_error("lk_scale must be in [0.25,1]");
    if (c.video_stride_mode != "dynamic" && c.video_stride_mode != "fixed")
        throw std::runtime_error("video_stride_mode must be dynamic or fixed");
    if (!std::isfinite(c.video_detect_fps) || c.video_detect_fps < 0 || c.video_detect_fps > 10000)
        throw std::runtime_error("video_detect_fps must be in [0,10000]; 0 detects every frame");
    if (c.opencv_threads < 1 || c.opencv_threads > 16 || c.inference_timeout_ms < 1 || c.inference_timeout_ms > 60000)
        throw std::runtime_error("invalid opencv_threads or inference_timeout_ms");
    if (c.class_names.size() != size_t(c.num_classes))
        throw std::runtime_error("labels must contain exactly num_classes entries");
}

AppConfig loadAppConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open config: " + path);
    AppConfig c;
    std::set<std::string> seen;
    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim(withoutComment(line));
        if (line.empty()) continue;
        try {
            const auto colon = line.find(':');
            if (colon == std::string::npos) throw std::runtime_error("expected key: value");
            const std::string key = trim(line.substr(0, colon));
            const std::string value = unquote(trim(line.substr(colon + 1)));
            if (!seen.insert(key).second) throw std::runtime_error("duplicate key: " + key);
            if (key == "model_path") c.model_path = value;
            else if (key == "labels_path") c.labels_path = value;
            else if (key == "input_width") c.input_width = integer(value);
            else if (key == "input_height") c.input_height = integer(value);
            else if (key == "num_classes") c.num_classes = integer(value);
            else if (key == "conf_threshold") c.conf_threshold = number(value);
            else if (key == "class_conf_thresholds") c.class_conf_thresholds = nlohmann::json::parse(value).get<std::vector<float>>();
            else if (key == "preprocess") c.preprocess = value;
            else if (key == "use_letterbox") c.use_letterbox = boolean(value);
            else if (key == "lk_preprocess") c.lk_preprocess = value;
            else if (key == "lk_scale") c.lk_scale = number(value);
            else if (key == "video_detect_fps") c.video_detect_fps = number(value);
            else if (key == "video_model_async") c.video_model_async = boolean(value);
            else if (key == "video_stride_mode") c.video_stride_mode = value;
            else if (key == "opencv_threads") c.opencv_threads = integer(value);
            else if (key == "inference_timeout_ms") c.inference_timeout_ms = integer(value);
            else throw std::runtime_error("unknown or removed setting: " + key);
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) + ": " + e.what());
        }
    }
    const auto base = std::filesystem::absolute(path).parent_path();
    auto resolve = [&base](const std::string& p) {
        if (p.empty()) throw std::runtime_error("model_path and labels_path are required");
        return (base / p).lexically_normal().string();
    };
    c.model_path = resolve(c.model_path);
    c.labels_path = resolve(c.labels_path);
    validateModelDimensions(c);
    std::ifstream labels(c.labels_path);
    if (!labels) throw std::runtime_error("cannot open labels: " + c.labels_path);
    const auto json = nlohmann::json::parse(labels);
    if ((!json.is_array() && !json.is_object()) || json.size() != size_t(c.num_classes))
        throw std::runtime_error("expected " + std::to_string(c.num_classes) + " labels");
    for (int i = 0; i < c.num_classes; ++i) {
        const auto& value = json.is_array() ? json.at(i) : json.at(std::to_string(i));
        c.class_names.push_back(value.get<std::string>());
        if (c.class_names.back().empty()) throw std::runtime_error("empty class name");
    }
    validateConfig(c);
    return c;
}
}  // namespace yolo

#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include "config/app_config.h"
#include "image/image_processing.h"
#include "io/result_writer.h"
#include "model/yolo_engine.h"

#ifndef YOLO_DEFAULT_CONFIG_PATH
#define YOLO_DEFAULT_CONFIG_PATH "config.yaml"
#endif

namespace {
bool samePath(const std::filesystem::path& a, const std::filesystem::path& b) {
    if (a.empty() || b.empty()) return false;
    std::error_code ec;
    if (std::filesystem::equivalent(a, b, ec)) return true;
    return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b);
}
}
int main(int argc, char** argv) {
    try {
        const std::set<std::string> keys{"--config", "--input", "--model", "--preprocess", "--output", "--annotated"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout << "Usage: yolo_run --input FILE [--config FILE] [--model FILE.rknn]\n"
                             "                [--preprocess rga] [--output results.jsonl]\n"
                             "                [--annotated image.jpg|video.avi]\n";
                return 0;
            }
            if (!keys.count(key) || i + 1 == argc || args.count(key)) throw std::runtime_error("invalid, missing or repeated argument: " + key);
            args[key] = argv[++i];
        }
        if (!args.count("--input")) throw std::runtime_error("--input is required; see --help");
        // The vendor libraries also print diagnostics via C stdout. Keep every
        // runtime diagnostic on stderr; --help above retains normal stdout.
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot route SDK diagnostics to stderr");
        auto config = yolo::loadAppConfig(args.count("--config") ? args.at("--config") : YOLO_DEFAULT_CONFIG_PATH);
        if (args.count("--model")) config.model_path = std::filesystem::absolute(args.at("--model")).lexically_normal().string();
        if (args.count("--preprocess")) config.preprocess = args.at("--preprocess");
        yolo::validateConfig(config);
        const std::filesystem::path input = args.at("--input");
        if (!std::filesystem::is_regular_file(input)) throw std::runtime_error("input must be a local regular file: " + input.string());
        const std::filesystem::path output = args.count("--output") ? args.at("--output") : "results.jsonl";
        const std::filesystem::path annotated = args.count("--annotated") ? args.at("--annotated") : "";
        for (const auto& destination : {output, annotated}) {
            for (const auto& source : {input, std::filesystem::path(config.model_path), std::filesystem::path(config.labels_path),
                    std::filesystem::path(args.count("--config") ? args.at("--config") : YOLO_DEFAULT_CONFIG_PATH)})
                if (samePath(destination, source)) throw std::runtime_error("output would overwrite an input/config/model file");
        }
        if (samePath(output, annotated)) throw std::runtime_error("JSONL and annotated output paths must differ");
        cv::setNumThreads(config.opencv_threads);
        cv::Mat image = cv::imread(input.string(), cv::IMREAD_COLOR);
        const bool video = image.empty();
        double fps = 0;
        if (video) {
            cv::VideoCapture probe(input.string());
            if (!probe.isOpened()) throw std::runtime_error("cannot decode input as image or video");
            fps = probe.get(cv::CAP_PROP_FPS);
        }
        // 初始化一次模型及输出缓冲，再把查询到的 stride/最低容量交给预处理器。
        yolo::YoloEngine engine(config);
        yolo::ImagePreprocessor preprocessor(config, engine.inputMemoryLayout());
        // 单帧检测入口：BGR Mat → RGA/DMA-BUF → RKNN → 原图坐标检测框。
        // 首帧由主线程同步调用；启用视频异步后由固定 Worker 串行调用同一套预处理/推理对象。
        const yolo::InferFunction infer = [&](const cv::Mat& frame, int64_t index) {
            const auto start = yolo::Clock::now();
            auto tensor = preprocessor.prepare(frame); // 持有本帧租约，直到 infer 返回并离开此作用域。
            const double preprocess_ms = yolo::elapsedMs(start);
            auto result = engine.infer(tensor, index);
            result.preprocess_ms = preprocess_ms;
            return result;
        };
        // 图片直接输出检测；视频层额外负责 LK/ByteTrack 和异步时序，writer 统一写 JSONL/标注。
        yolo::ResultWriter writer(config, output, annotated, video, fps);
        yolo::RunStats stats;
        if (video) {
            stats = yolo::inferVideoFile(input, config, infer, [&](const cv::Mat& frame, const yolo::FrameResult& r) { writer.frame(frame, r); });
        } else {
            yolo::UsageTimer timer;
            auto result = infer(image, 0);
            yolo::FrameResult frame;
            frame.width = image.cols; frame.height = image.rows;
            frame.is_detection_frame = true;
            frame.tracks_source = "detected";
            frame.detections = result.detections;
            writer.frame(image, frame);
            stats.frames = 1;
            stats.add(result);
            timer.finish(stats);
        }
        writer.finish(stats);
        std::cerr << "Completed " << stats.frames << " frames, " << stats.inference_count << " inferences; results: " << output << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

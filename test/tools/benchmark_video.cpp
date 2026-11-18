#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>
#include "config/app_config.h"
#include "image/image_processing.h"
#include "io/result_writer.h"
#include "model/yolo_engine.h"

using namespace yolo;
int main(int argc, char** argv) {
    try {
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("diagnostic routing failed");
        std::map<std::string, std::string> args;
        const std::set<std::string> options{"--config", "--input", "--output", "--mode", "--frame-map", "--ready-file"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (!options.count(key) || i + 1 >= argc || args.count(key)) throw std::runtime_error("invalid argument: " + key);
            args[key] = argv[++i];
        }
        const auto config = loadAppConfig(args.at("--config"));
        const auto mode = args.at("--mode");
        if (mode != "detection" && mode != "pipeline") throw std::runtime_error("mode must be detection or pipeline");
        std::set<int64_t> selected;
        if (mode == "detection") {
            std::ifstream map_file(args.at("--frame-map"));
            const auto map = nlohmann::json::parse(map_file);
            for (const auto& row : map) {
                const auto index = row.at("frame_index").get<int64_t>();
                if (index < 0 || !selected.insert(index).second) throw std::runtime_error("invalid frame map");
            }
            if (selected.empty()) throw std::runtime_error("empty frame map");
        }
        cv::setNumThreads(config.opencv_threads);
        YoloEngine engine(config);
        ImagePreprocessor preprocessor(config, engine.inputMemoryLayout());
        const InferFunction infer = [&](const cv::Mat& frame, int64_t index) {
            const auto start = Clock::now();
            auto input = preprocessor.prepare(frame);
            const double preprocessing_ms = elapsedMs(start);
            auto result = engine.infer(input, index);
            result.preprocess_ms = preprocessing_ms;
            return result;
        };
        double fps;
        {
            cv::VideoCapture warmup(args.at("--input"));
            cv::Mat first;
            if (!warmup.read(first) || first.empty()) throw std::runtime_error("warmup video decode failed");
            if (first.cols != 1280 || first.rows != 720) throw std::runtime_error("expected display-oriented video 1280x720; verify rotation metadata");
            fps = warmup.get(cv::CAP_PROP_FPS);
            for (int i = 0; i < 5; ++i) infer(first, 0);
        }
        ResultWriter writer(config, args.at("--output"), {}, true, fps);
        if (args.count("--ready-file")) {
            std::ofstream ready(args.at("--ready-file"));
            ready << "warmup_complete\n";
            if (!ready) throw std::runtime_error("cannot write measurement marker");
        }
        RunStats stats;
        if (mode == "pipeline") {
            stats = inferVideoFile(args.at("--input"), config, infer,
                [&](const cv::Mat& frame, const FrameResult& result) { writer.frame(frame, result); });
        } else {
            UsageTimer timer;
            cv::VideoCapture capture(args.at("--input"));
            cv::Mat frame;
            int64_t index = 0;
            while (true) {
                const auto start = Clock::now();
                const bool ok = capture.read(frame);
                const auto decode_ms = elapsedMs(start);
                if (!ok || frame.empty()) break;
                stats.decode.add(decode_ms);
                if (frame.cols != 1280 || frame.rows != 720) throw std::runtime_error("decoded orientation/dimensions changed");
                if (selected.count(index)) {
                    auto result = infer(frame, index);
                    stats.add(result);
                    FrameResult output;
                    output.frame_index = index;
                    output.timestamp_ms = index * 1000.0 / fps;
                    output.width = frame.cols; output.height = frame.rows;
                    output.is_detection_frame = true;
                    output.tracks_source = "detected";
                    output.detections = std::move(result.detections);
                    writer.frame(frame, output);
                    ++stats.frames;
                    if (stats.frames % 50 == 0) std::cerr << "Annotated inferences: " << stats.frames << '/' << selected.size() << '\n';
                }
                ++index;
            }
            if (stats.frames != selected.size()) throw std::runtime_error("missing annotated source frames");
            stats.source_fps = fps;
            timer.finish(stats);
        }
        writer.finish(stats);
        std::cerr << "Finished " << mode << ": decoded=" << stats.decode.count << " inferred=" << stats.inference_count << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << '\n';
        return 1;
    }
}

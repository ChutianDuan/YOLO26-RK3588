#include "camera/live_media.h"
#include "model/yolo_engine.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
int main(int argc, char** argv) {
    const auto directory = std::filesystem::temp_directory_path() / ("live_npu_" + std::to_string(getpid()));
    try {
        using namespace yolo;
        using namespace yolo::live;
        std::string model = std::string(YOLO_SOURCE_DIR) + "/../yolo_model_rknn/yolo26s_800x608_int8.rknn";
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: live_npu_test [--model MODEL.rknn]\n";
            return 0;
        }
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--model")
                throw std::runtime_error("Usage: live_npu_test [--model MODEL.rknn]");
            model = std::filesystem::absolute(argv[2]).lexically_normal().string();
        }
        std::filesystem::create_directory(directory);
        cv::Mat bus = cv::imread("/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg"), resized;
        if (bus.empty()) throw std::runtime_error("missing installed bus fixture");
        cv::resize(bus, resized, {800,600});
        if (!cv::imwrite((directory / "0.jpg").string(), resized)) throw std::runtime_error("cannot write JPEG");
        {
            std::ofstream manifest(directory / "manifest.jsonl");
            manifest << "{\"type\":\"mjpeg_fixture\",\"width\":800,\"height\":600}\n"
                     << "{\"file\":\"0.jpg\",\"sequence\":0,\"timestamp_ms\":0}\n";
        }
        auto source = replaySource(directory / "manifest.jsonl", false);
        std::atomic<bool> stop{false}; auto packet = source->next(stop);
        MjpegDecoder decoder; auto frame = decoder.decode(*packet);
        auto config = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
        config.model_path = model;
        config.conf_threshold = .001F; config.class_conf_thresholds.clear();
        std::vector<Detection> baseline;
        nlohmann::json report = nlohmann::json::array();
        for (int mask : {-1,7,1,2,4}) {
            YoloEngine engine(config, mask);
            ImagePreprocessor processor(config, engine.inputMemoryLayout());
            auto tensor = processor.prepareNv12(frame->image);
            float max_box_error = 0, max_score_error = 0;
            for (int repeat = 0; repeat < 3; ++repeat) {
                const auto result = engine.infer(tensor);
                if (result.detections.empty()) throw std::runtime_error("fixed NPU fixture produced no candidates");
                for (const auto& d : result.detections) {
                    if (!std::isfinite(d.score) || !std::isfinite(d.x1) || !std::isfinite(d.y1) ||
                        !std::isfinite(d.x2) || !std::isfinite(d.y2) || d.score < 0 || d.score > 1 ||
                        d.class_id < 0 || d.class_id >= config.num_classes || d.x2 < d.x1 || d.y2 < d.y1)
                        throw std::runtime_error("invalid fixed-input detection");
                }
                if (baseline.empty()) baseline = result.detections;
                if (baseline.size() != result.detections.size()) throw std::runtime_error("core modes changed detection count");
                for (size_t i = 0; i < baseline.size(); ++i) {
                    const auto& a = baseline[i]; const auto& b = result.detections[i];
                    if (a.class_id != b.class_id) throw std::runtime_error("core modes changed candidate class/order");
                    max_box_error = std::max({max_box_error, std::abs(a.x1-b.x1),std::abs(a.y1-b.y1),std::abs(a.x2-b.x2),std::abs(a.y2-b.y2)});
                    max_score_error = std::max(max_score_error, std::abs(a.score-b.score));
                }
            }
            if (max_box_error > 1 || max_score_error > .01F) throw std::runtime_error("NPU equivalence tolerance exceeded (1px/0.01 score)");
            report.push_back({{"core_mask",mask},{"repeats",3},{"candidates",baseline.size()},
                {"max_box_error_px",max_box_error},{"max_score_error",max_score_error}});
        }
        std::cout << report.dump(2) << '\n';
        std::filesystem::remove_all(directory); return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; std::filesystem::remove_all(directory); return 1; }
}

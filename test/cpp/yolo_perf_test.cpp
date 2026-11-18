#include "config/app_config.h"
#include "image/image_processing.h"
#include "model/yolo_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rknn_api.h>

namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using namespace yolo;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int positiveCount(const std::string& text, int limit) {
    require(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
            "invalid positive count: " + text);
    const auto count = std::stoul(text);
    require(count > 0 && count <= static_cast<unsigned>(limit), "count out of range: " + text);
    return static_cast<int>(count);
}
Json statistics(std::vector<double> values) {
    require(!values.empty(), "no timing samples");
    for (double value : values) require(std::isfinite(value) && value >= 0, "invalid timing sample");
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double q) { return values[static_cast<size_t>(std::ceil(q * values.size())) - 1]; };
    return {{"mean", mean}, {"min", values.front()}, {"p50", percentile(.5)},
            {"p95", percentile(.95)}, {"max", values.back()}};
}
struct Difference {
    float box = 0, score = 0;
    void compare(const std::vector<Detection>& expected, const std::vector<Detection>& actual,
                 const AppConfig& config) {
        require(expected.size() == actual.size(), "profiling changed detection count");
        for (size_t i = 0; i < actual.size(); ++i) {
            const auto& a = expected[i];
            const auto& b = actual[i];
            require(b.class_id >= 0 && b.class_id < config.num_classes && b.class_id == a.class_id,
                    "profiling changed detection class/order");
            require(std::isfinite(b.score) && b.score >= 0 && b.score <= 1 &&
                    std::isfinite(b.x1) && std::isfinite(b.y1) && std::isfinite(b.x2) && std::isfinite(b.y2) &&
                    b.x1 >= 0 && b.y1 >= 0 && b.x2 > b.x1 && b.y2 > b.y1,
                    "invalid detection");
            score = std::max(score, std::abs(a.score - b.score));
            box = std::max({box, std::abs(a.x1 - b.x1), std::abs(a.y1 - b.y1),
                            std::abs(a.x2 - b.x2), std::abs(a.y2 - b.y2)});
        }
        require(box <= 1e-3F && score <= 1e-6F,
                "profiling changed detection coordinates/scores (tolerance 0.001px / 0.000001)");
    }
};
struct Phase {
    Json report;
    std::vector<Detection> reference;
    NpuPerformance performance;
};

Phase runPhase(const AppConfig& config, const cv::Mat& image, int mask, bool collect,
               int warmup, int iterations, const std::vector<Detection>& baseline, Difference& difference) {
    // Separate, serial contexts: baseline really uses flags=0. Each context owns its registration.
    YoloEngine engine(config, mask, collect);
    ImagePreprocessor processor(config, engine.inputMemoryLayout());
    // Fixed image: RGB/letterbox preparation and DMA allocation happen outside the measured loop.
    // This lease stays alive until every synchronous inference and its output retrieval finishes.
    const auto input = processor.prepare(image);
    for (int i = 0; i < warmup; ++i) engine.infer(input, i);
    std::vector<InferResult> results;
    std::vector<double> wall_ms, infer_ms, post_ms;
    results.reserve(iterations);
    wall_ms.reserve(iterations);
    infer_ms.reserve(iterations);
    post_ms.reserve(iterations);
    const auto loop_start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        const auto start = Clock::now();
        auto result = engine.infer(input, i); // No query, string copy or printing in the timing loop.
        wall_ms.push_back(milliseconds(start));
        results.push_back(std::move(result));
    }
    const double loop_ms = milliseconds(loop_start);
    auto reference = baseline.empty() ? results.front().detections : baseline;
    require(!reference.empty(), "fixture produced no detections; use an image containing known objects");
    for (const auto& result : results) {
        difference.compare(reference, result.detections, config);
        infer_ms.push_back(result.infer_ms);
        post_ms.push_back(result.postprocess_ms);
    }
    // One additional, unmeasured inference queries the performance of that exact frame, inside the lock.
    NpuPerformance perf;
    const auto diagnostic = engine.infer(input, iterations, &perf);
    difference.compare(reference, diagnostic.detections, config);
    require(perf.run_duration_us > 0, "RKNN_QUERY_PERF_RUN returned no duration");
    require(collect ? !perf.detail.empty() : perf.detail.empty(), "unexpected detail collection state");
    Json report{{"collect_perf", collect}, {"init_flags", collect ? RKNN_FLAG_COLLECT_PERF_MASK : 0},
                {"warmup", warmup}, {"iterations", iterations}, {"detection_count", reference.size()},
                {"infer_wall_ms", statistics(wall_ms)}, {"bind_run_get_ms", statistics(infer_ms)},
                {"postprocess_ms", statistics(post_ms)}, {"loop_wall_ms", loop_ms},
                {"diagnostic_run_duration_us", perf.run_duration_us},
                {"diagnostic_runtime_run_ms", perf.run_duration_us / 1000.0},
                {"detail_bytes", perf.detail.size()}};
    if (!collect) report["fixed_input_serial_infer_fps"] = iterations * 1000.0 / loop_ms;
    return {std::move(report), std::move(reference), std::move(perf)};
}

void writeFile(const std::filesystem::path& path, const std::string& data) {
    std::ofstream file(path, std::ios::binary);
    require(bool(file), "cannot create: " + path.string());
    file.write(data.data(), static_cast<std::streamsize>(data.size()));
    file.close();
    require(bool(file), "cannot write: " + path.string());
}
void usage() {
    std::cout << "Usage: yolo_perf_test [--config CONFIG] [--model MODEL.rknn] [--image IMAGE]\n"
                 "                      [--preprocess rga] [--core default|0|1|2|all]\n"
                 "                      [--warmup 1..100] [--iterations 1..1000] [--output-dir NEW_DIR]\n"
                 "Defaults: project config/model, installed bus.jpg, config preprocess, default core,\n"
                 "          3 warmups, 20 measured inferences per phase. Runs baseline then detail.\n"
                 "Fixed-input inference only; detailed collection timings are diagnostic.\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        const std::set<std::string> allowed{"--config", "--model", "--image", "--preprocess", "--core",
                                            "--warmup", "--iterations", "--output-dir"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            require(allowed.count(key) && i + 1 < argc && !args.count(key), "invalid argument: " + key);
            args[key] = argv[++i];
            require(!args[key].empty() && args[key].rfind("--", 0) != 0, "missing value: " + key);
        }
        const int warmup = args.count("--warmup") ? positiveCount(args.at("--warmup"), 100) : 3;
        const int iterations = args.count("--iterations") ? positiveCount(args.at("--iterations"), 1000) : 20;
        const std::map<std::string, int> masks{{"default", -1}, {"0", RKNN_NPU_CORE_0},
            {"1", RKNN_NPU_CORE_1}, {"2", RKNN_NPU_CORE_2}, {"all", RKNN_NPU_CORE_0_1_2}};
        const std::string core = args.count("--core") ? args.at("--core") : "default";
        require(masks.count(core), "invalid --core: " + core);
        const std::string config_path = args.count("--config") ? args.at("--config") :
            std::string(YOLO_SOURCE_DIR) + "/config.yaml";
        auto config = loadAppConfig(config_path);
        if (args.count("--model")) config.model_path = std::filesystem::absolute(args.at("--model")).lexically_normal().string();
        if (args.count("--preprocess")) config.preprocess = args.at("--preprocess");
        validateConfig(config);
        cv::setNumThreads(config.opencv_threads);
        const auto image_path = std::filesystem::absolute(args.count("--image") ? args.at("--image") :
            "/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg").lexically_normal();
        const auto image = cv::imread(image_path.string());
        require(!image.empty(), "cannot read image: " + image_path.string());
        std::filesystem::path output;
        if (args.count("--output-dir")) {
            output = std::filesystem::absolute(args.at("--output-dir")).lexically_normal();
            require(std::filesystem::create_directory(output), "output directory must be new: " + output.string());
        }
        Difference difference;
        std::cerr << "Baseline: COLLECT_PERF disabled; fixed input, preprocessing excluded\n";
        const auto baseline = runPhase(config, image, masks.at(core), false, warmup, iterations, {}, difference);
        std::cerr << "Profile: COLLECT_PERF enabled; timings for diagnosis only\n";
        const auto profile = runPhase(config, image, masks.at(core), true, warmup, iterations, baseline.reference, difference);
        Json report{{"model", config.model_path}, {"model_bytes", std::filesystem::file_size(config.model_path)},
            {"config", std::filesystem::absolute(config_path).lexically_normal().string()},
            {"image", image_path.string()}, {"image_width", image.cols}, {"image_height", image.rows},
            {"input_width", config.input_width}, {"input_height", config.input_height},
            {"input_format", "RGB UINT8 NHWC, pass_through=0; output want_float=1"},
            {"use_letterbox", config.use_letterbox}, {"conf_threshold", config.conf_threshold},
            {"class_conf_thresholds", config.class_conf_thresholds}, {"preprocess", config.preprocess},
            {"core", core}, {"core_mask", masks.at(core)}, {"opencv_threads", config.opencv_threads},
            {"baseline", baseline.report}, {"profile", profile.report},
            {"checks", {{"passed", true}, {"max_box_error_px", difference.box}, {"max_score_error", difference.score}}},
            {"scope", "single context, fixed input; excludes image decode/preprocessing/warmup and perf queries; "
                      "baseline FPS includes infer/output release/postprocess and result retention, not video throughput; "
                      "diagnostic Runtime duration is one extra frame, not a timing-loop average, and may include CPU operators; "
                      "no accuracy target assessed"}};
        if (!output.empty()) {
            writeFile(output / "perf_detail.txt", profile.performance.detail);
            writeFile(output / "report.json", report.dump(2) + "\n");
        }
        std::cout << report.dump(2) << "\n\nRKNN_QUERY_PERF_DETAIL (diagnostic frame):\n";
        std::cout.write(profile.performance.detail.data(), static_cast<std::streamsize>(profile.performance.detail.size()));
        std::cout << "\nRuntime run (RKNN_QUERY_PERF_RUN): " << profile.performance.run_duration_us / 1000.0 << " ms\n";
        if (!output.empty()) std::cerr << "Reports: " << output << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL yolo_perf_test: " << e.what() << '\n';
        return 1;
    }
}

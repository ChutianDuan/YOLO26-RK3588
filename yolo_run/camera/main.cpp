#include "camera/v4l2_camera.h"
#include "image/dma_buffer.h"
#include "metrics/run_stats.h"
#include "model/yolo_engine.h"
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
using Json = nlohmann::json;
using namespace yolo;
// camera_run 是 YUYV 单 Context 的逐帧检测入口；三 Worker/轨迹 ID 在 camera_live。
volatile sig_atomic_t stopping = 0;
void signalStop(int) { stopping = 1; }

class JsonLines {
public:
    explicit JsonLines(const std::string& path) {
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error("cannot create new output " + path + ": " + std::strerror(errno));
        file = fdopen(fd, "w");
        if (!file) { close(fd); throw std::runtime_error("fdopen output failed"); }
    }
    ~JsonLines() { if (file) fclose(file); }
    void write(const Json& value) {
        const auto text = value.dump();
        if (fwrite(text.data(), 1, text.size(), file) != text.size() || fputc('\n', file) == EOF || fflush(file))
            throw std::runtime_error("cannot write camera JSONL");
    }
private:
    FILE* file = nullptr;
};

std::shared_ptr<DmaBuffer> allocation(const TensorInput& tensor) {
    return std::static_pointer_cast<DmaBuffer>(tensor.dma.allocation);
}
cv::Mat rgbView(const TensorInput& t) {
    const auto& d = t.dma;
    // 只构造带真实字节步长的外部视图；调用方另行持有 tensor owner 和 CPU READ 区间。
    return cv::Mat(d.height, d.width, CV_8UC3, d.data, size_t(d.width_stride) * 3);
}
Json timing(const TimingAggregate& t) {
    return {{"count", t.count}, {"mean_ms", t.mean()}, {"max_ms", t.max_ms}};
}
Json detections(const std::vector<Detection>& values, const AppConfig& c) {
    auto array = Json::array();
    for (const auto& d : values)
        array.push_back({{"class_id", d.class_id}, {"class_name", c.class_names.at(d.class_id)},
                         {"score", d.score}, {"bbox", {d.x1, d.y1, d.x2, d.y2}}});
    return array;
}
void snapshot(const std::filesystem::path& path, const TensorInput& tensor, const InferResult& result,
              const AppConfig& config) {
    // 截图是显式 CPU 读取用途；RGB→BGR 产生独立 Mat 后才结束 DMA-BUF READ 区间。
    CpuAccess read(allocation(tensor), DMA_BUF_SYNC_READ);
    cv::Mat bgr;
    cv::cvtColor(rgbView(tensor), bgr, cv::COLOR_RGB2BGR);
    read.finish();
    for (const auto& d : result.detections) {
        // 检测框已是原图坐标；截图画布为模型输入，需要重新应用缩放/补边来画框。
        const auto& g = tensor.letterbox;
        const cv::Point p1(cvRound(d.x1 * g.scale_x + g.pad_w), cvRound(d.y1 * g.scale_y + g.pad_h));
        const cv::Point p2(cvRound(d.x2 * g.scale_x + g.pad_w), cvRound(d.y2 * g.scale_y + g.pad_h));
        cv::rectangle(bgr, p1, p2, cv::Scalar(30, 220, 60), 2);
        cv::putText(bgr, config.class_names.at(d.class_id) + " " + std::to_string(int(d.score * 100)) + "%",
                    {p1.x, std::max(18, p1.y - 4)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {30, 220, 60}, 1);
    }
    if (!cv::imwrite(path.string(), bgr)) throw std::runtime_error("cannot save snapshot");
}

struct Resources {
    uint64_t samples = 0;
    size_t first_fd = 0, last_fd = 0, min_fd = 0, max_fd = 0;
    double first_rss = 0, last_rss = 0, max_rss = 0, max_temp = 0;
    Json sample() {
        size_t fds = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) { (void)entry; ++fds; }
        uint64_t virtual_pages = 0, resident_pages = 0;
        std::ifstream("/proc/self/statm") >> virtual_pages >> resident_pages;
        const double rss = resident_pages * double(sysconf(_SC_PAGESIZE)) / 1048576.0;
        double temp = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/sys/class/thermal")) {
            if (entry.path().filename().string().rfind("thermal_zone", 0) != 0) continue;
            double value = 0; std::ifstream(entry.path() / "temp") >> value;
            temp = std::max(temp, value / 1000.0);
        }
        std::string load;
        std::getline(std::ifstream("/sys/kernel/debug/rknpu/load"), load);
        if (!samples++) { first_fd = min_fd = fds; first_rss = rss; }
        last_fd = fds; min_fd = std::min(min_fd, fds); max_fd = std::max(max_fd, fds);
        last_rss = rss; max_rss = std::max(max_rss, rss); max_temp = std::max(max_temp, temp);
        return {{"fd_count", fds}, {"rss_mb", rss}, {"temperature_c", temp}, {"npu_load", load}};
    }
    Json summary() const {
        return {{"samples", samples}, {"first_fd", first_fd}, {"last_fd", last_fd}, {"min_fd", min_fd}, {"max_fd", max_fd},
                {"first_rss_mb", first_rss}, {"last_rss_mb", last_rss}, {"max_rss_mb", max_rss}, {"max_temperature_c", max_temp}};
    }
};
uint64_t unsignedValue(const std::string& text) {
    uint64_t value = 0;
    const auto rc = std::from_chars(text.data(), text.data() + text.size(), value);
    if (rc.ec != std::errc() || rc.ptr != text.data() + text.size() || text.empty())
        throw std::runtime_error("--max-frames must be a nonnegative integer");
    return value;
}
void requireNewFile(const std::filesystem::path& path) {
    if (path.empty() || std::filesystem::exists(std::filesystem::symlink_status(path)) ||
        (!path.parent_path().empty() && !std::filesystem::is_directory(path.parent_path())))
        throw std::runtime_error("output must be a new file in an existing directory: " + path.string());
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::set<std::string> keys{"--device", "--config", "--model", "--io", "--max-frames", "--output", "--snapshot"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout << "Usage: camera_run [--device /dev/v4l/by-id/...] [--io dmabuf]\n"
                             "                  [--model FILE.rknn] [--config FILE] [--max-frames N]\n"
                             "                  [--output NEW.jsonl] [--snapshot NEW.jpg|png]\n"
                             "Capture: 800x600 YUYV, request 15 FPS; RGB model size/labels come from --config.\n"
                             "Without --config, use legacy 800x608 INT8. --model overrides config.\n"
                             "N=0 runs until SIGINT/SIGTERM.\n"
                             "Outputs must not exist. No tracking, preview or recording.\n";
                return 0;
            }
            if (!keys.count(key) || i + 1 == argc || args.count(key)) throw std::runtime_error("invalid/missing/repeated argument: " + key);
            args[key] = argv[++i];
        }
        auto get = [&](const char* key, const std::string& fallback) { return args.count(key) ? args.at(key) : fallback; };
        const auto mode = get("--io", "dmabuf");
        if (mode != "dmabuf") throw std::runtime_error("--io must be dmabuf; CPU copy/fallback was removed");
        const auto limit = unsignedValue(get("--max-frames", "0"));
        const auto device = get("--device", "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0");
        const auto config_path = get("--config", std::string(YOLO_SOURCE_DIR) + "/config.yaml");
        auto config = loadAppConfig(config_path);
        if (args.count("--model"))
            config.model_path = std::filesystem::absolute(args.at("--model")).lexically_normal().string();
        else if (!args.count("--config"))
            config.model_path = std::string(YOLO_SOURCE_DIR) + "/../yolo_model_rknn/yolo26s_800x608_int8.rknn";
        validateConfig(config);
        if (!config.use_letterbox)
            throw std::runtime_error("camera requires letterbox");
        const auto output = get("--output", "camera_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(getpid()) + ".jsonl");
        const std::filesystem::path picture = get("--snapshot", "");
        requireNewFile(output);
        if (!picture.empty()) {
            requireNewFile(picture);
            const auto ext = picture.extension().string();
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") throw std::runtime_error("snapshot must be .jpg, .jpeg or .png");
            if (std::filesystem::weakly_canonical(output) == std::filesystem::weakly_canonical(picture))
                throw std::runtime_error("JSONL and snapshot must differ");
        }
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot route SDK stdout to stderr");
        struct sigaction action{};
        action.sa_handler = signalStop; sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) || sigaction(SIGTERM, &action, nullptr)) throw std::runtime_error("cannot install signal handlers");
        cv::setNumThreads(config.opencv_threads);
        // 先查询模型输入布局，再构造预处理器，避免按紧密 RGB 字节数误分配 FP16 容量。
        YoloEngine engine(config);
        ImagePreprocessor processor(config, engine.inputMemoryLayout());
        auto camera = std::make_unique<V4l2Camera>(device);
        const auto format = camera->format();
        const auto geometry = resizeGeometry(800, 600, config);
        JsonLines writer(output);
        writer.write({{"type", "camera"}, {"device", device}, {"requested_io", mode}, {"io_mode", "dmabuf"},
                      {"model_path", config.model_path}, {"source_width", 800}, {"source_height", 600},
                      {"model_width", config.input_width}, {"model_height", config.input_height}, {"num_classes", config.num_classes},
                      {"labels_path", config.labels_path}, {"pixel_format", "YUYV"}, {"color_space", "BT.601 limited"},
                      {"bytes_per_line", format.bytes_per_line}, {"size_image", format.size_image}, {"buffers", format.buffers},
                      {"requested_fps", 15}, {"negotiated_fps", format.fps}, {"bbox_coordinates", "source_800x600"},
                      {"padding", {{"top", geometry.top}, {"bottom", config.input_height - geometry.height - geometry.top},
                          {"left", geometry.left}, {"right", config.input_width - geometry.width - geometry.left}, {"rgb", {114, 114, 114}}}},
                      {"scale_x", geometry.transform.scale_x}, {"scale_y", geometry.transform.scale_y}});
        std::cerr << "Camera dmabuf: 800x600 YUYV -> " << config.input_width << 'x' << config.input_height
                  << " RGB, negotiated " << format.fps << " FPS\n";
        RunStats stats;
        UsageTimer timer;
        Resources resources;
        TimingAggregate capture_wait, end_to_end, snapshot_timing;
        uint64_t snapshot_bytes = 0;
        const auto before = camera->counters();
        std::string error;
        try {
            while (!stopping && (!limit || stats.frames < limit)) {
                const auto start = Clock::now();
                auto frame = camera->next(&stopping);
                capture_wait.add(elapsedMs(start));
                if (!frame) break;
                const auto prepare_start = Clock::now();
                // RGA 同步读取 YUYV 源并写入独立 RGB；tensor 保留 RGB 当前帧的占用。
                auto tensor = processor.prepareYuyv(frame->image);
                const double preprocess_ms = elapsedMs(prepare_start);
                const auto sequence = frame->sequence;
                const double timestamp = frame->timestamp_ms;
                const bool driver_timestamp = frame->timestamp_from_driver;
                frame.reset(); // RGA IM_SYNC 已完成；立即归还源帧，NPU 不再使用它。
                // 同步推理期间 tensor 仍在；即使 engine 清空内部 active_input，
                // 调用方也继续保留它直到本轮截图/统计结束，随后才允许下一帧覆盖 RGB。
                auto result = engine.infer(tensor, sequence);
                result.preprocess_ms = preprocess_ms;
                stats.add(result); ++stats.frames;
                const double latency = std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count() - timestamp;
                end_to_end.add(latency);
                writer.write({{"type", "frame"}, {"frame_index", stats.frames - 1}, {"sequence", sequence},
                              {"timestamp_ms", timestamp}, {"timestamp_clock", driver_timestamp ? "v4l2_monotonic" : "host_monotonic"},
                              {"width", 800}, {"height", 600}, {"model_width", config.input_width}, {"model_height", config.input_height},
                              {"detections", detections(result.detections, config)}, {"preprocess_ms", preprocess_ms},
                              {"infer_ms", result.infer_ms}, {"postprocess_ms", result.postprocess_ms}, {"latency_ms", latency}});
                if (stats.frames == 1 && !picture.empty()) {
                    const auto snap_start = Clock::now();
                    snapshot(picture, tensor, result, config);
                    snapshot_timing.add(elapsedMs(snap_start));
                    snapshot_bytes = uint64_t(config.input_width) * config.input_height * 3;
                }
                if (stats.frames == 1 || stats.frames % 30 == 0 || (limit && stats.frames == limit)) {
                    auto sample = resources.sample();
                    sample["type"] = "resource"; sample["frame_count"] = stats.frames;
                    writer.write(sample);
                }
                if (stats.frames % 150 == 0) std::cerr << "Processed " << stats.frames << " frames, " << stats.frames * 1000.0 / elapsedMs(timer.start) << " FPS\n";
            }
        } catch (const std::exception& e) { error = e.what(); }
        const auto after = camera->counters();
        // 结束循环后先 STREAMOFF；现有缓冲与 RKNN 注册随后按各自 RAII 顺序释放。
        camera->stop();
        timer.finish(stats);
        const double seconds = stats.elapsed_ms / 1000;
        const uint64_t captured = after.dequeued - before.dequeued;
        const double ts_elapsed = after.last_timestamp_ms - before.last_timestamp_ms;
        const double observed_fps = before.dequeued && ts_elapsed > 0 ?
            uint32_t(after.last_sequence - before.last_sequence) * 1000.0 / ts_elapsed :
            (after.dequeued > 1 && after.last_timestamp_ms > after.first_timestamp_ms ?
             uint32_t(after.last_sequence - after.first_sequence) * 1000.0 / (after.last_timestamp_ms - after.first_timestamp_ms) : 0);
        writer.write({{"type", "summary"}, {"status", error.empty() ? (stopping ? "interrupted" : "completed") : "error"}, {"error", error},
                      {"io_mode", "dmabuf"}, {"frame_count", stats.frames}, {"inference_count", stats.inference_count},
                      {"dequeued_count", captured}, {"stale_dropped", after.stale_dropped - before.stale_dropped},
                      {"corrupt_dropped", after.corrupt_dropped - before.corrupt_dropped}, {"driver_sequence_gaps", after.sequence_gaps - before.sequence_gaps},
                      {"negotiated_fps", format.fps}, {"observed_source_fps", observed_fps}, {"dequeued_fps", seconds > 0 ? captured / seconds : 0},
                      {"processing_fps", seconds > 0 ? stats.frames / seconds : 0}, {"inference_fps", seconds > 0 ? stats.inference_count / seconds : 0},
                      {"elapsed_ms", stats.elapsed_ms}, {"cpu_percent", stats.cpu_percent}, {"peak_rss_mb", stats.peak_rss_mb},
                      {"application_upload_bytes", 0},
                      {"snapshot_conversion_count", snapshot_timing.count}, {"snapshot_conversion_output_bytes", snapshot_bytes},
                      {"resources", resources.summary()},
                      {"timing", {{"capture_wait", timing(capture_wait)}, {"preprocess", timing(stats.preprocess)}, {"infer", timing(stats.infer)},
                                  {"postprocess", timing(stats.postprocess)}, {"capture_to_result", timing(end_to_end)}, {"snapshot", timing(snapshot_timing)}}}});
        std::cerr << (error.empty() ? "Completed " : "Stopped with error: " + error + "; ") << stats.frames << " frames; "
                  << (seconds > 0 ? stats.frames / seconds : 0) << " FPS; upload=" << 0 << " bytes; " << output << '\n';
        return error.empty() ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}

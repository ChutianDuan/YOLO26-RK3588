#include "camera/live_pipeline.h"
#include <charconv>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <unistd.h>
#include <opencv2/core.hpp>

namespace {
volatile sig_atomic_t stopped = 0;
// 信号处理器只设置标志；线程停止、硬件等待和资源释放由正常控制流完成。
void signalStop(int) { stopped = 1; }
uint64_t integer(const std::string& text) {
    uint64_t value = 0;
    const auto rc = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || rc.ec != std::errc() || rc.ptr != text.data() + text.size()) throw std::runtime_error("invalid nonnegative integer: " + text);
    return value;
}
double number(const std::string& text) {
    size_t used = 0; const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) throw std::runtime_error("invalid finite number: " + text);
    return value;
}
bool toggle(const std::string& text) {
    if (text == "on") return true;
    if (text == "off") return false;
    throw std::runtime_error("expected on or off");
}
}
int yolo::live::runLiveCli(int argc, char** argv, const LiveCallbacks& callbacks) {
    try {
        using namespace yolo;
        using namespace yolo::live;
        std::set<std::string> keys{"--device", "--config", "--model", "--output", "--max-frames", "--preview", "--decoder",
            "--npu-mode", "--detect-fps", "--max-predict-ms", "--replay", "--stream", "--stream-port", "--stream-bitrate",
            "--stabilize", "--stabilize-alpha"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            if (key == "--help" || key == "-h") {
                std::cout << "Usage: camera_live [--device DEVICE | --replay manifest.jsonl]\n"
                    "  [--preview on|off] [--decoder mpp] [--npu-mode default|all|split]\n"
                    "  [--detect-fps 0] [--max-predict-ms 500] [--max-frames 0]\n"
                    "  [--model MODEL.rknn] [--config config.yaml] [--output NEW.jsonl]\n"
                    "  [--stream on|off] [--stream-port 12340] [--stream-bitrate 2000000]\n"
                    "  [--stabilize on|off] [--stabilize-alpha 0.15]\n"
                    "Stabilization defaults on: 720x540 ROI zoomed to 800x600; boxes use stabilized coordinates.\n"
                    "800x600 MJPEG, request 30 FPS; RGB model size/labels come from --config.\n"
                    "Without --config, use the legacy 800x608 INT8 model. --model overrides config.\n"
                    "NPU mode defaults to split (3 independent single-core workers); camera_live.json may override it.\n"
                    "Preview defaults on; use off without a graphical desktop.\n";
                return 0;
            }
            if (!keys.count(key) || i + 1 == argc || args.count(key)) throw std::runtime_error("invalid/missing/repeated argument: " + key);
            args[key] = argv[++i];
        }
        auto get = [&](const char* key, const std::string& fallback) { return args.count(key) ? args.at(key) : fallback; };
        if (args.count("--device") && args.count("--replay")) throw std::runtime_error("--device and --replay are mutually exclusive");
        auto config = loadAppConfig(get("--config", std::string(YOLO_SOURCE_DIR) + "/config.yaml"));
        if (args.count("--model"))
            config.model_path = std::filesystem::absolute(args.at("--model")).lexically_normal().string();
        else if (!args.count("--config"))
            config.model_path = std::string(YOLO_SOURCE_DIR) + "/../yolo_model_rknn/yolo26s_800x608_int8.rknn";
        validateConfig(config);
        if (!config.use_letterbox)
            throw std::runtime_error("camera_live requires letterbox");
        // 模式优先级：命令行 > camera_live.json > LiveOptions 中的 split 兜底。
        // 参数值 "default" 特指 Runtime 默认核选择，不是省略 --npu-mode 的含义。
        LiveOptions options;
        std::ifstream defaults(std::string(YOLO_SOURCE_DIR) + "/config/camera_live.json");
        if (defaults) { nlohmann::json settings; defaults >> settings; options.npu_mode = settings.at("npu_mode"); }
        options.npu_mode = get("--npu-mode", options.npu_mode);
        options.preview = toggle(get("--preview", "on"));
        const auto decoder = get("--decoder", "mpp");
        if (decoder != "mpp") throw std::runtime_error("--decoder must be mpp; CPU decoding was removed");
        options.stream = toggle(get("--stream", "off"));
        options.stabilize = toggle(get("--stabilize", "on"));
        options.stabilize_alpha = number(get("--stabilize-alpha", "0.15"));
        if (options.stabilize_alpha <= 0 || options.stabilize_alpha > 1)
            throw std::runtime_error("stabilize-alpha must be in (0,1]");
        const auto port = integer(get("--stream-port", "12340"));
        const auto bitrate = integer(get("--stream-bitrate", "2000000"));
        if (port < 1 || port > 65535 || bitrate < 64000 || bitrate > 14000000)
            throw std::runtime_error("stream-port must be 1..65535; stream-bitrate must be 64000..14000000");
        options.stream_port = static_cast<uint16_t>(port); options.stream_bitrate = static_cast<int>(bitrate);
        options.detect_fps = number(get("--detect-fps", "0"));
        options.max_predict_ms = number(get("--max-predict-ms", "500"));
        options.max_frames = integer(get("--max-frames", "0"));
        options.output = get("--output", "camera_live_" + std::to_string(std::time(nullptr)) + "_" + std::to_string(getpid()) + ".jsonl");
        if (options.npu_mode != "default" && options.npu_mode != "all" && options.npu_mode != "split") throw std::runtime_error("invalid --npu-mode");
        if (options.detect_fps < 0 || options.detect_fps > 1000 || options.max_predict_ms < 0 || options.max_predict_ms > 1000)
            throw std::runtime_error("detect-fps must be 0..1000, max-predict-ms 0..1000");
        if (std::filesystem::exists(std::filesystem::symlink_status(options.output))) throw std::runtime_error("output must be a new file");
        if (options.preview && !std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
            throw std::runtime_error("preview needs a graphical session; use --preview off for headless validation");
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot route SDK output to stderr");
        struct sigaction action{}; action.sa_handler = signalStop; sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) || sigaction(SIGTERM, &action, nullptr)) throw std::runtime_error("sigaction failed");
        cv::setNumThreads(config.opencv_threads);
        // 源对象覆盖 runLive 的整个调用期；返回前已 join 所有使用它的线程。
        auto source = args.count("--replay") ? replaySource(args.at("--replay")) : cameraSource(get("--device", "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"));
        const auto summary = runLive(*source, config, options, &stopped, callbacks);
        std::cerr << summary.dump(2) << '\n';
        return summary.at("status") == "error" ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}

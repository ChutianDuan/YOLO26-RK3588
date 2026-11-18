#include "video/multi_video.h"
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <unistd.h>

#ifndef YOLO_MULTI_CONFIG
#define YOLO_MULTI_CONFIG "config/video_multi.yaml"
#endif
namespace yolo::multi {
namespace {
volatile sig_atomic_t interrupted = 0;
void stopSignal(int) { interrupted = 1; }
}
int runCli(int argc, char** argv, const Callbacks& callbacks) {
    try {
        Options options;
        std::map<std::string, std::string> args;
        const std::set<std::string> keys{"--config", "--model", "--output-dir", "--preview", "--record", "--duration-seconds",
            "--detect-fps", "--track-max-age-ms"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout << "Usage: yolo_multi --input FILE [--input FILE ... (1..4)] --output-dir NEW_DIR\n"
                    "  [--config FILE] [--model FILE.rknn] [--preview auto|on|off]\n"
                    "  [--record on|off] [--duration-seconds N]\n"
                    "  [--detect-fps N (0: unlimited)] [--track-max-age-ms 1..1000]\n"
                    "Defaults: INT8, three single-core workers, ByteTrack/Kalman prediction (500 ms), record on.\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("missing value: " + key);
            if (key == "--input") { options.inputs.emplace_back(argv[++i]); continue; }
            if (!keys.count(key) || args.count(key)) throw std::runtime_error("invalid or repeated argument: " + key);
            args[key] = argv[++i];
        }
        if (options.inputs.empty() || options.inputs.size() > 4 || !args.count("--output-dir"))
            throw std::runtime_error("1..4 --input files and --output-dir are required");
        options.output_dir = args.at("--output-dir");
        if (args.count("--preview")) options.preview = args.at("--preview");
        if (args.count("--record")) {
            if (args.at("--record") != "on" && args.at("--record") != "off") throw std::runtime_error("--record requires on|off");
            options.record = args.at("--record") == "on";
        }
        if (args.count("--duration-seconds")) {
            size_t end = 0; const auto& value = args.at("--duration-seconds");
            options.duration_seconds = std::stod(value, &end);
            if (end != value.size()) throw std::runtime_error("invalid duration");
        }
        auto number = [&](const char* key, double fallback) {
            if (!args.count(key)) return fallback;
            size_t end = 0; const auto& value = args.at(key);
            const double parsed = std::stod(value, &end);
            if (end != value.size()) throw std::runtime_error(std::string("invalid number: ") + key);
            return parsed;
        };
        options.detect_fps = number("--detect-fps", options.detect_fps);
        options.track_max_age_ms = number("--track-max-age-ms", options.track_max_age_ms);
        auto config = loadAppConfig(args.count("--config") ? args.at("--config") : YOLO_MULTI_CONFIG);
        if (args.count("--model")) config.model_path = std::filesystem::absolute(args.at("--model")).lexically_normal().string();
        validateConfig(config);
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot route SDK diagnostics to stderr");
        interrupted = 0;
        auto previous_int = std::signal(SIGINT, stopSignal), previous_term = std::signal(SIGTERM, stopSignal);
        nlohmann::json summary;
        try { summary = run(config, options, &interrupted, callbacks); }
        catch (...) { std::signal(SIGINT, previous_int); std::signal(SIGTERM, previous_term); throw; }
        std::signal(SIGINT, previous_int); std::signal(SIGTERM, previous_term);
        if (!summary.at("fatal_error").get<std::string>().empty()) std::cerr << "Error: " << summary.at("fatal_error") << '\n';
        for (const auto& channel : summary.at("channels")) for (const char* key : {"source_error", "output_error", "record_error"})
            if (!channel.at(key).get<std::string>().empty())
                std::cerr << "Channel " << channel.at("channel_id") << " " << key << ": " << channel.at(key) << '\n';
        std::cerr << "Four-video run: " << summary.at("status") << "; results: " << options.output_dir << '\n';
        return summary.at("status") == "failed" ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
} // namespace yolo::multi

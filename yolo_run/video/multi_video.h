#pragma once
#include <csignal>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include "config/app_config.h"

namespace yolo::multi {
struct Options {
    std::vector<std::filesystem::path> inputs;
    std::filesystem::path output_dir;
    std::string preview = "auto";
    bool record = true;
    double duration_seconds = 0;
    double detect_fps = 0; // 0: all published frames compete for NPU workers.
    double track_max_age_ms = 500;
};
// Test instrumentation runs on the owning thread; callbacks must be thread safe.
struct Callbacks {
    std::function<void(size_t, size_t, int64_t)> before_infer;
    std::function<void(size_t, int64_t)> before_decode;
    std::function<void(size_t)> before_write;
};
nlohmann::json run(const AppConfig& config, const Options& options,
                   const volatile sig_atomic_t* signal_stop = nullptr,
                   const Callbacks& callbacks = {});
int runCli(int argc, char** argv, const Callbacks& callbacks = {});
} // namespace yolo::multi

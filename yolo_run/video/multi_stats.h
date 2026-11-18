#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sys/resource.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace yolo::multi {
using Json = nlohmann::json;
using Steady = std::chrono::steady_clock;
inline double nowMs() {
    return std::chrono::duration<double, std::milli>(Steady::now().time_since_epoch()).count();
}
// Bounded 1-ms histogram. Samples >=4096 ms share an explicitly reported overflow bin.
class Distribution {
public:
    void add(double ms) {
        if (!std::isfinite(ms) || ms < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        ++bins_[std::min<size_t>(4096, size_t(std::ceil(ms)))]; ++count_;
        sum_ += ms; maximum_ = std::max(maximum_, ms);
    }
    Json json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto quantile = [&](double p) {
            uint64_t n = 0;
            for (size_t i = 0; i < bins_.size(); ++i) {
                n += bins_[i]; if (n >= std::ceil(count_ * p)) return double(i);
            }
            return 0.0;
        };
        return {{"count", count_}, {"mean_ms", count_ ? sum_ / count_ : 0}, {"max_ms", maximum_},
            {"p50_ms", count_ ? quantile(.5) : 0}, {"p95_ms", count_ ? quantile(.95) : 0},
            {"p99_ms", count_ ? quantile(.99) : 0}, {"quantile_resolution_ms", 1}, {"overflow_bin_ms", 4096}};
    }
private:
    mutable std::mutex mutex_;
    std::array<uint64_t, 4097> bins_{};
    uint64_t count_ = 0;
    double sum_ = 0, maximum_ = 0;
};
inline double cpuSeconds() {
    rusage r{}; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 + r.ru_stime.tv_sec + r.ru_stime.tv_usec / 1e6;
}
inline Json readNumber(const std::filesystem::path& path, double divisor = 1) {
    double v = 0; std::ifstream in(path);
    if (!(in >> v)) return nullptr;
    return v / divisor;
}
class ResourceSampler {
public:
    Json sample() {
        const double time = nowMs(), cpu = cpuSeconds();
        Json row{{"monotonic_ms", time}, {"cpu_percent", nullptr}, {"rss_mb", nullptr},
                 {"pss_mb", nullptr}, {"fd_count", nullptr}, {"npu_load_percent", nullptr},
                 {"temperature_c", Json::object()}, {"cpu_frequency_khz", Json::object()}};
        if (last_ms_ && time - last_ms_ > 1) row["cpu_percent"] = (cpu - last_cpu_) * 100000 / (time - last_ms_);
        last_ms_ = time; last_cpu_ = cpu;
        uint64_t size = 0, resident = 0;
        if (std::ifstream("/proc/self/statm") >> size >> resident)
            row["rss_mb"] = resident * double(sysconf(_SC_PAGESIZE)) / 1048576;
        std::string line;
        std::ifstream rollup("/proc/self/smaps_rollup");
        while (std::getline(rollup, line)) if (line.rfind("Pss:", 0) == 0)
            row["pss_mb"] = std::stod(line.substr(4)) / 1024;
        std::error_code ec;
        size_t fds = 0;
        for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd", ec)) { (void)e; ++fds; }
        if (!ec) row["fd_count"] = fds;
        std::ifstream load("/sys/kernel/debug/rknpu/load");
        if (std::getline(load, line)) {
            int a = 0, b = 0, c = 0;
            if (std::sscanf(line.c_str(), "NPU load: Core0: %d%%, Core1: %d%%, Core2: %d%%", &a, &b, &c) == 3)
                row["npu_load_percent"] = {a, b, c};
        }
        for (const auto& e : std::filesystem::directory_iterator("/sys/class/thermal", ec)) {
            if (e.path().filename().string().rfind("thermal_zone", 0)) continue;
            std::string name; std::ifstream(e.path() / "type") >> name;
            if (!name.empty()) row["temperature_c"][name] = readNumber(e.path() / "temp", 1000);
        }
        for (const auto& e : std::filesystem::directory_iterator("/sys/devices/system/cpu/cpufreq", ec))
            row["cpu_frequency_khz"][e.path().filename().string()] = readNumber(e.path() / "scaling_cur_freq");
        row["npu_frequency_hz"] = readNumber("/sys/class/devfreq/fdab0000.npu/cur_freq");
        return row;
    }
private:
    double last_ms_ = 0, last_cpu_ = 0;
};
} // namespace yolo::multi

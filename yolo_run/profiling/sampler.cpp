#include "profiling/sampler.h"
#include <ctime>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unistd.h>
namespace yolo::trace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
uint64_t timestamp() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}
std::string line(const fs::path &path) {
    std::string value;
    std::getline(std::ifstream(path), value);
    return value;
}
} // namespace
Sampler::Sampler(const fs::path &dir) : output_(dir / "samples.jsonl") {
    std::error_code ec;
    for (auto &e : fs::directory_iterator("/sys/class/thermal", ec)) {
        auto type = line(e.path() / "type");
        if (e.path().filename().string().find("thermal_zone") == 0 && !type.empty())
            gauges_[type + " mC"] = e.path() / "temp";
    }
    for (auto &e : fs::directory_iterator("/sys/devices/system/cpu/cpufreq", ec))
        if (e.path().filename().string().find("policy") == 0)
            gauges_[e.path().filename().string() + " kHz"] = e.path() / "scaling_cur_freq";
    gauges_["NPU Hz"] = "/sys/class/devfreq/fdab0000.npu/cur_freq";
    gauges_["DDR SCMI Hz"] = "/sys/kernel/debug/clk/scmi_clk_ddr/clk_rate";
}
void Sampler::sample() {
    Json row = {{"begin_ns", timestamp()}};
    Json values = Json::object();
    for (const auto &[name, path] : gauges_) {
        int64_t v;
        if (std::ifstream(path) >> v)
            values[name] = v;
        else
            values[name] = nullptr;
    }
    unsigned long size, resident;
    if (std::ifstream("/proc/self/statm") >> size >> resident)
        values["RSS MiB"] = resident * double(sysconf(_SC_PAGESIZE)) / 1048576;
    std::ifstream proc("/proc/stat");
    std::string s;
    while (std::getline(proc, s) && s.rfind("cpu", 0) == 0) {
        std::istringstream fields(s);
        std::string core;
        fields >> core;
        uint64_t total = 0, idle = 0, v;
        for (int i = 0; i < 8 && fields >> v; ++i) {
            total += v;
            if (i == 3 || i == 4)
                idle += v;
        }
        auto previous = cpu_[core];
        if (previous.first && total > previous.first)
            values[core + " %"] =
                100.0 * (1 - double(idle - previous.second) / (total - previous.first));
        cpu_[core] = {total, idle};
    }
    auto load = line("/sys/kernel/debug/rknpu/load");
    for (int i = 0; i < 3; ++i) {
        auto tag = "Core" + std::to_string(i) + ":";
        auto p = load.find(tag);
        int percent;
        if (p != std::string::npos && std::istringstream(load.substr(p + tag.size())) >> percent)
            values["NPU " + std::to_string(i) + " %"] = percent;
    }
    row["timestamp_ns"] = timestamp();
    row["values"] = std::move(values);
    output_ << row.dump() << '\n';
    output_.flush();
}
} // namespace yolo::trace

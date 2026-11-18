#include "camera/live_pipeline.h"
#include <algorithm>
#include <charconv>
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>

// This entry and every injected delay/failure are linked only by test/.
int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> controls;
        std::vector<char*> forwarded{argv[0]};
        const std::vector<std::string> allowed{"--test-infer-delay-ms", "--test-output-delay-ms",
            "--test-write-delay-ms", "--test-fail-write-after", "--test-reorder", "--test-encode-delay-ms", "--test-fail-encode-after"};
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            if (key.rfind("--test-", 0) != 0) { forwarded.push_back(argv[i]); continue; }
            if (std::find(allowed.begin(), allowed.end(), key) == allowed.end() ||
                controls.count(key) || i + 1 >= argc)
                throw std::runtime_error("invalid/repeated test argument: " + key);
            controls[key] = argv[++i];
        }
        auto integer = [&](const char* key, int fallback) {
            const auto found = controls.find(key);
            if (found == controls.end()) return fallback;
            unsigned value = 0;
            const auto& text = found->second;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (text.empty() || parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || value > INT_MAX)
                throw std::runtime_error("invalid test integer: " + text);
            return int(value);
        };
        const int infer = integer("--test-infer-delay-ms", 0), output = integer("--test-output-delay-ms", 0);
        const int write = integer("--test-write-delay-ms", 0), fail_after = integer("--test-fail-write-after", -1);
        const auto reorder = controls.count("--test-reorder") ? controls.at("--test-reorder") : "off";
        if (reorder != "on" && reorder != "off") throw std::runtime_error("invalid test toggle");
        yolo::live::LiveCallbacks callbacks;
        callbacks.before_infer = [=](int worker) {
            if (infer && (reorder != "on" || worker == 0))
                std::this_thread::sleep_for(std::chrono::milliseconds(infer));
        };
        callbacks.before_output = [=] {
            if (output) std::this_thread::sleep_for(std::chrono::milliseconds(output));
        };
        callbacks.before_write = [write, fail_after, writes = 0]() mutable {
            if (write) std::this_thread::sleep_for(std::chrono::milliseconds(write));
            if (fail_after >= 0 && writes++ >= fail_after) throw std::runtime_error("injected JSONL write failure");
        };
        callbacks.before_encode = [delay=integer("--test-encode-delay-ms",0), limit=integer("--test-fail-encode-after",-1), n=0]() mutable {
            if (delay) std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            if (limit >= 0 && n++ >= limit) throw std::runtime_error("injected encoder failure");
        };
        const int count = int(forwarded.size());
        forwarded.push_back(nullptr);
        return yolo::live::runLiveCli(count, forwarded.data(), callbacks);
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}

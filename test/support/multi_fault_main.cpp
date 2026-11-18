#include "video/multi_video.h"
#include <chrono>
#include <cstdlib>
#include <thread>
int main(int argc, char** argv) {
    yolo::multi::Callbacks callbacks;
    const std::string fault = std::getenv("YOLO_MULTI_TEST_FAULT") ? std::getenv("YOLO_MULTI_TEST_FAULT") : "";
    if (fault == "writer") callbacks.before_write = [](size_t c) { if (c == 0) throw std::runtime_error("injected writer failure"); };
    if (fault == "record-lag") callbacks.before_write = [](size_t c) { if (c == 0) std::this_thread::sleep_for(std::chrono::milliseconds(300)); };
    if (fault == "slow-infer") callbacks.before_infer = [](size_t, size_t c, int64_t) {
        if (c == 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    };
    if (fault == "infer-error") callbacks.before_infer = [](size_t, size_t c, int64_t) { if (c == 0) throw std::runtime_error("injected infer failure"); };
    if (fault == "decode-error") callbacks.before_decode = [](size_t c, int64_t index) {
        if (c == 0 && index == 3) throw std::runtime_error("injected decode failure");
    };
    return yolo::multi::runCli(argc, argv, callbacks);
}

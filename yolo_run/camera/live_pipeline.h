#pragma once
#include <csignal>
#include <functional>
#include "camera/live_media.h"
#include "config/app_config.h"

namespace yolo::live {
// 实时 MJPEG 入口的调度配置；不同于 camera_run 的同步 YUYV 逐帧检测。
struct LiveOptions {
    // split：3 个独立 Context/Worker 各绑定一个 NPU 核，处理不同帧；
    // all：单 Context/Worker 绑定三核，default：单 Context 保留 Runtime 核选择。
    std::string output, npu_mode = "split";
    bool preview = true;
    bool stream = false;
    bool stabilize = true;
    double stabilize_alpha = 0.15;
    uint16_t stream_port = 12340;
    int stream_bitrate = 2000000;
    uint64_t max_frames = 0;
    // detect_fps=0 不主动限检测提交频率；预测超出最后匹配时间上限后隐藏轨迹。
    double detect_fps = 0, max_predict_ms = 500;

};
// 回调在相应 Worker/输出/写入线程执行，捕获的共享状态必须自行保证线程安全。
struct LiveCallbacks {
    std::function<void(int)> before_infer;
    std::function<void()> before_output, before_write;
    std::function<void()> before_encode;
};
int runLiveCli(int argc, char** argv, const LiveCallbacks& callbacks = {});
// 阻塞到停止或源结束；内部固定线程执行采集、解码、推理、轨迹校正和 JSONL 写入。
// source 必须覆盖整个调用期；返回前 join 线程，随后各硬件对象按 RAII 清理。
nlohmann::json runLive(FrameSource& source, const AppConfig& config, const LiveOptions& options,
                       const volatile sig_atomic_t* signal_stop = nullptr,
                       const LiveCallbacks& callbacks = {});
} // namespace yolo::live

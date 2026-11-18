#pragma once
#include <csignal>
#include <memory>
#include <optional>
#include <string>
#include "image/image_processing.h"

namespace yolo {
// G_FMT/G_PARM 的实际协商结果；bytes_per_line 是字节步长，size_image 是槽容量。
// 不用请求的 800x600 去替代驱动返回的布局/容量检查。
struct CameraFormat {
    int width = 0, height = 0, bytes_per_line = 0;
    size_t size_image = 0;
    double fps = 0;
    unsigned buffers = 0;
};
struct CameraCounters {
    uint64_t dequeued = 0, stale_dropped = 0, corrupt_dropped = 0, sequence_gaps = 0;
    uint32_t first_sequence = 0, last_sequence = 0;
    double first_timestamp_ms = 0, last_timestamp_ms = 0;
};
struct CameraFrame {
    // image.owner 是归还采集槽的租约；image.dma 是分配所有者。
    // data 指向的源图在 owner 生命周期内有效，RGA 读完前不能主动释放该 owner。
    YuyvFrameView image;
    uint32_t sequence = 0;
    double timestamp_ms = 0;
    bool timestamp_from_driver = false;
};
// YUYV 同步入口使用，next/租约归还均按单线程约束调用。
// image.owner 的最后一个副本释放时，运行中才 QBUF；停止后不再给驱动入队。
// 租约还会保住共享 State，使外层对象销毁后仍被持有的采集帧不立即失效。
class V4l2Camera {
public:
    explicit V4l2Camera(const std::string& device);
    ~V4l2Camera();
    V4l2Camera(const V4l2Camera&) = delete;
    V4l2Camera& operator=(const V4l2Camera&) = delete;
    std::optional<CameraFrame> next(const volatile sig_atomic_t* stop = nullptr);
    CameraFormat format() const;
    CameraCounters counters() const;
    void stop();
private:
    class State;
    std::shared_ptr<State> state_;
};
} // namespace yolo

#pragma once
#include "image/image_processing.h"
#include "profiling/trace.h"
#include "video/frame_stabilizer.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

namespace yolo::live {
double nowMs();
// sequence 标识源帧；timestamp_ms 进入本机单调时钟域，用于异步关联、预测和延迟。
// raw_timestamp_ms 保留原驱动/夹具时间，不直接与本机完成时间相减。
struct Stamp {
    int64_t sequence = 0;
    double timestamp_ms = 0, raw_timestamp_ms = 0;
    uint32_t flags = 0;
    std::string clock = "host_monotonic";
    uint64_t frame_id = 0, dispatch = 0;
    double arrival_ms = 0;
    // 不限速回放会提前读到未来的媒体帧，此时端到端耗时从实际到达时间起算。
    double latencyOrigin(bool unpaced) const { return unpaced ? arrival_ms : timestamp_ms; }
};
struct PacketStorage;
struct EncodedFrame {
    // 帧持有压缩池槽的共享引用，解码完成前阻止生产者把该槽写成下一包。
    std::shared_ptr<PacketStorage> storage;
    size_t bytes = 0;
    Stamp stamp;
    const unsigned char* data() const;
    // 同步 CPU READ 视图：保持 allocation 和 Cache Sync，回调不能保存裸指针稍后读取。
    void withCpuRead(const std::function<void(const unsigned char*, size_t)>& reader) const;
};
// 单一生产者使用固定预分配池；V4L2 排队或消费者持有的槽不会被重用。
class PacketPool {
public:
    explicit PacketPool(size_t capacity, size_t slots = 4);
    std::shared_ptr<PacketStorage> acquire(uint64_t frame_id = 0);

private:
    std::vector<std::shared_ptr<PacketStorage>> slots_;
};
struct DecodedFrame {
    // MPP 路径：image.owner 持有 Surface 帧租约，image.dma 只负责底层分配存活。
    // 下游必须同时保留 owner，不能只留下 FD/Mat，否则输出池可能提前重写这一帧。
    Nv12FrameView image;
    Stamp stamp;
    double decode_ms = 0;
    StabilizationInfo stabilization;
};
// 有界队列支持竞争消费。满队列时 FIFO 阻塞生产者，替换策略释放最旧等待项。
// take() 把描述/租约移交给一个消费者；旧等待项被替换不影响已经取走的在途帧。
// close() 取消所有等待项并唤醒线程。正常 EOF 调用 finish()，允许消费者排空队列。
template <class T> class LatestSlot {
public:
    using Id = uint64_t (*)(const T&);
    using Dispatch = void (*)(T&, uint64_t);
    explicit LatestSlot(trace::Stage stage = trace::PacketQueue, Id id = nullptr, size_t capacity = 1,
                        bool fifo = false, Dispatch dispatch = nullptr)
        : slots_(capacity), stage_(stage), id_(id), dispatch_(dispatch), fifo_(fifo) {
        if (!capacity) throw std::invalid_argument("queue capacity must be positive");
    }
    bool put(T value) {
        auto lock = queueLock(trace::enabled() && id_ ? id_(value) : 0);
        if (fifo_) cv_.wait(lock, [&] { return closed_ || size_ < slots_.size(); });
        if (closed_) {
            ++cancelled_;
            event(trace::Evict, value, 2);
            return false;
        }
        if (size_ == slots_.size()) {
            event(trace::Evict, *slots_[head_], 1);
            pop();
            ++dropped_;
        }
        event(trace::Enter, value, 0);
        slots_[(head_ + size_) % slots_.size()] = std::move(value);
        ++size_;
        cv_.notify_all();
        return true;
    }
    std::optional<T> take(bool wait = true) {
        auto lock = queueLock();
        if (wait) cv_.wait(lock, [&] { return closed_ || size_; });
        if (!size_) return {};
        auto value = std::move(slots_[head_]);
        pop();
        if (dispatch_) dispatch_(*value, ++dispatch_count_);
        event(trace::Leave, *value, 0);
        cv_.notify_all();
        return value;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        while (size_) {
            event(trace::Evict, *slots_[head_], 2);
            pop();
            ++cancelled_;
        }
        cv_.notify_all();
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }
    uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }
    uint64_t cancelled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cancelled_;
    }
    bool finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_ && !size_;
    }

private:
    std::unique_lock<std::mutex> queueLock(uint64_t frame = 0) {
        std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
        if (trace::enabled() && !lock.try_lock()) {
            trace::Scope wait(trace::Lock, frame, stage_);
            lock.lock();
        } else if (!lock.owns_lock())
            lock.lock();
        return lock;
    }
    void event(trace::Kind kind, const T& v, uint64_t reason) {
        if (trace::enabled() && id_) trace::emit(kind, stage_, id_(v), 0, reason);
    }
    void pop() {
        slots_[head_].reset();
        head_ = (head_ + 1) % slots_.size();
        --size_;
    }
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::optional<T>> slots_;
    size_t head_ = 0, size_ = 0;
    trace::Stage stage_;
    Id id_;
    Dispatch dispatch_;
    bool fifo_, closed_ = false;
    uint64_t dropped_ = 0, cancelled_ = 0, dispatch_count_ = 0;
};
class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual std::optional<EncodedFrame> next(const std::atomic<bool>& stop) = 0;
    virtual nlohmann::json info() const = 0;
    virtual nlohmann::json counters() const = 0; // 采集生产者 join 后再读取，避免与计数写入竞争。
};
std::unique_ptr<FrameSource> cameraSource(const std::string& device);
std::unique_ptr<FrameSource> replaySource(const std::filesystem::path& manifest, bool paced = true);

// 同一实例只由解码线程调用；MPP 按配置预分配 Surface，同步确认解码后发布共享 NV12。
// 对象析构先停止/销毁 MPP Context；仍被消费者租用的 Surface 可独立延长存活期。
class MjpegDecoder {
public:
    explicit MjpegDecoder(int timeout_ms = 1000, size_t pool_size = 8);
    ~MjpegDecoder();
    std::optional<DecodedFrame> decode(const EncodedFrame& packet);
    MjpegDecoder(const MjpegDecoder&) = delete;
    MjpegDecoder& operator=(const MjpegDecoder&) = delete;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
class Renderer {
public:
    Renderer();
    // 输出线程独占；同步 RGA 转 BGR 后返回借用 Mat，CPU RW 区间持续到 finish()。
    // 双槽不提供外部异步租约，调用者不得跨后续 render 保存该视图供异步使用。
    cv::Mat convert(const DecodedFrame& frame);
    // 结束 CPU 访问，不是等待显示后端扫描完成，也不负责输出 NV12 源租约的归还。
    void finish();
    // finish() 后、下次 convert() 前：同步读取当前 BGR，写入有独立帧租约的
    // BT.601 limited NV12 目标；不将借用 Mat 交给异步编码线程。
    void convertForEncode(const Nv12FrameView& destination);
    ~Renderer();
private:
    std::array<std::shared_ptr<DmaBuffer>, 2> buffers_;
    std::unique_ptr<class RenderAccess> access_;
    size_t next_ = 0;
};
} // namespace yolo::live

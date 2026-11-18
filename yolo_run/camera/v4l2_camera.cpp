#include "camera/v4l2_camera.h"
#include "image/dma_buffer.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace yolo {
namespace {
int call(int fd, unsigned long request, void* arg) {
    int rc;
    do { rc = ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR);
    return rc;
}
[[noreturn]] void fail(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}
void checked(int fd, unsigned long request, void* arg, const char* operation) {
    if (call(fd, request, arg) < 0) fail(operation);
}
}

// State 持有采集 FD/分配，Lease 又共享 State；先停止采集，再等最后租约
// 退出才释放实际资源，避免外层 V4l2Camera 析构后消费者仍在访问旧源帧。
class V4l2Camera::State : public std::enable_shared_from_this<State> {
public:
    struct Slot {
        std::shared_ptr<DmaBuffer> dma;
        size_t size = 0;
        bool leased = false;
    };
    struct Lease {
        Lease(std::shared_ptr<State> capture, unsigned slot_index)
            : state(std::move(capture)), index(slot_index) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        std::shared_ptr<State> state;
        unsigned index;
        // 归还的是当前帧占用，不是销毁 allocation；运行中会 QBUF 供驱动写下一帧。
        ~Lease() { state->release(index); }
    };
    ~State() {
        shutdown();
        if (fd >= 0) close(fd);
    }
    void initialize(const std::string& device) {
        fd = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) fail("open camera " + device);
        v4l2_capability caps{};
        checked(fd, VIDIOC_QUERYCAP, &caps, "VIDIOC_QUERYCAP");
        const auto c = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
        if (!(c & V4L2_CAP_VIDEO_CAPTURE) || !(c & V4L2_CAP_STREAMING))
            throw std::runtime_error("camera needs single-plane video capture and streaming capabilities");
        v4l2_format request{};
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.fmt.pix.width = 800; request.fmt.pix.height = 600;
        request.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; request.fmt.pix.field = V4L2_FIELD_NONE;
        checked(fd, VIDIOC_S_FMT, &request, "VIDIOC_S_FMT YUYV 800x600");
        checked(fd, VIDIOC_G_FMT, &request, "VIDIOC_G_FMT");
        const auto& p = request.fmt.pix;
        if (p.width != 800 || p.height != 600 || p.pixelformat != V4L2_PIX_FMT_YUYV || p.field != V4L2_FIELD_NONE ||
            p.bytesperline < 1600 || p.bytesperline > 32768 || p.bytesperline % 2 || p.sizeimage < p.bytesperline * p.height)
            throw std::runtime_error("camera did not negotiate supported 800x600 progressive YUYV layout");
        const uint32_t encoding = p.ycbcr_enc ? p.ycbcr_enc : uint32_t(V4L2_MAP_YCBCR_ENC_DEFAULT(p.colorspace));
        const uint32_t range = p.quantization ? p.quantization : uint32_t(V4L2_MAP_QUANTIZATION_DEFAULT(false, p.colorspace, encoding));
        if (encoding != V4L2_YCBCR_ENC_601 || range != V4L2_QUANTIZATION_LIM_RANGE)
            throw std::runtime_error("camera YUYV must use BT.601 limited range");
        format.width = p.width; format.height = p.height;
        format.bytes_per_line = p.bytesperline; format.size_image = p.sizeimage;
        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1; parm.parm.capture.timeperframe.denominator = 15;
        checked(fd, VIDIOC_S_PARM, &parm, "VIDIOC_S_PARM 15 FPS");
        checked(fd, VIDIOC_G_PARM, &parm, "VIDIOC_G_PARM");
        const auto interval = parm.parm.capture.timeperframe;
        if (!interval.numerator || !interval.denominator) throw std::runtime_error("invalid camera frame interval");
        format.fps = double(interval.denominator) / interval.numerator;
        // 应用从 DMA Heap 分配，V4L2 通过 FD 导入，RGA 读取同一分配。
        v4l2_requestbuffers buffers{};
        buffers.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; buffers.memory = memory; buffers.count = 4;
        checked(fd, VIDIOC_REQBUFS, &buffers, "VIDIOC_REQBUFS DMABUF");
        if (buffers.count < 2 || buffers.count > slots.size()) throw std::runtime_error("camera needs 2 to 4 fixed buffers");
        format.buffers = buffers.count;
        for (unsigned i = 0; i < format.buffers; ++i) {
            auto& slot = slots[i];
            // UVC 可能由内核 CPU 写入，沿用已验证的 uncached heap。
            slot.size = format.size_image;
            slot.dma = std::make_shared<DmaBuffer>(slot.size, "/dev/dma_heap/system-uncached");
            auto b = buffer(i);
            checked(fd, VIDIOC_QBUF, &b, "VIDIOC_QBUF initialize");
        }
        auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        checked(fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
        streaming = true;
    }
    v4l2_buffer buffer(unsigned i) const {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = memory; b.index = i;
        b.m.fd = slots[i].dma->fd; b.length = slots[i].size;
        return b;
    }
    void shutdown() noexcept {
        // STREAMOFF 确认采集不再写槽；现存 Lease 仍保留 State 和底层内存。
        stopped = true;
        if (streaming) {
            auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (call(fd, VIDIOC_STREAMOFF, &type) < 0) {
                std::cerr << "fatal: camera STREAMOFF failed; terminating before frame memory release\n";
                std::terminate();
            }
            streaming = false;
        }
    }
    void release(unsigned index) noexcept {
        slots[index].leased = false;
        if (stopped) return;
        auto b = buffer(index);
        // 析构不能抛异常；记录 QBUF 错误，在下一次 next 的正常控制流中报告。
        if (call(fd, VIDIOC_QBUF, &b) < 0) queue_error = errno;
    }
    std::optional<CameraFrame> next(const volatile sig_atomic_t* stop) {
        int timeouts = 0;
        unsigned bad = 0;
        while (!stop || !*stop) {
            if (stopped) throw std::runtime_error("camera already stopped");
            if (queue_error) { errno = queue_error; fail("VIDIOC_QBUF release"); }
            // 当前同步采集 API 要求上一帧先归还；共享租约不意味着支持多线程任意取帧。
            for (const auto& slot : slots) if (slot.leased) throw std::runtime_error("previous capture frame still leased");
            pollfd event{fd, POLLIN, 0};
            const int rc = poll(&event, 1, 1000);
            if (rc < 0) { if (errno == EINTR) continue; fail("poll camera"); }
            if (rc == 0) { if (++timeouts == 3) throw std::runtime_error("camera timed out three times (1 second each)"); continue; }
            if (event.revents & (POLLERR | POLLHUP | POLLNVAL)) throw std::runtime_error("camera disconnected or poll error");
            timeouts = 0;
            // 单次最多取出池容量内的已完成帧，归还旧帧只保留最新者，控制采集积压延迟。
            std::optional<CameraFrame> newest;
            for (unsigned drained = 0; drained < format.buffers; ++drained) {
                v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = memory;
                if (call(fd, VIDIOC_DQBUF, &b) < 0) {
                    if (errno == EAGAIN) break;
                    fail("VIDIOC_DQBUF");
                }
                if (b.index >= format.buffers) throw std::runtime_error("camera returned invalid buffer index");
                auto& slot = slots[b.index];
                slot.leased = true;
                // 直接构造共享租约，避免临时租约析构提前 QBUF。
                // 只复制 CameraFrame/YuyvFrameView 元数据会共享这份租约，不复制像素。
                auto lease = std::make_shared<Lease>(shared_from_this(), b.index);
                const bool monotonic = (b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
                const double ts = monotonic ? b.timestamp.tv_sec * 1000.0 + b.timestamp.tv_usec / 1000.0 :
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
                if (!counts.dequeued) { counts.first_sequence = b.sequence; counts.first_timestamp_ms = ts; }
                else {
                    const uint32_t delta = b.sequence - counts.last_sequence;
                    if (delta > 1 && delta < 0x80000000U) counts.sequence_gaps += delta - 1;
                }
                ++counts.dequeued; counts.last_sequence = b.sequence; counts.last_timestamp_ms = ts;
                if (b.flags & V4L2_BUF_FLAG_ERROR || b.bytesused < format.size_image || b.bytesused > slot.size) {
                    ++counts.corrupt_dropped;
                    if (++bad >= 12) throw std::runtime_error("too many incomplete/corrupt camera frames");
                    continue;
                }
                // reset 触发旧帧租约归还，而非直接写入或释放旧 allocation。
                if (newest) { newest.reset(); ++counts.stale_dropped; }
                const auto* data = static_cast<const unsigned char*>(slot.dma->data);
                newest = CameraFrame{{lease, slot.dma, data, slot.size, format.width, format.height,
                                       format.bytes_per_line}, b.sequence, ts, monotonic};
            }
            if (newest) return newest;
        }
        return std::nullopt;
    }
    int fd = -1, queue_error = 0;
    bool streaming = false, stopped = false;
    static constexpr v4l2_memory memory = V4L2_MEMORY_DMABUF;
    CameraFormat format;
    CameraCounters counts;
    std::array<Slot, 4> slots;
};
V4l2Camera::V4l2Camera(const std::string& device) : state_(std::make_shared<State>()) { state_->initialize(device); }
V4l2Camera::~V4l2Camera() { state_->shutdown(); }
std::optional<CameraFrame> V4l2Camera::next(const volatile sig_atomic_t* stop) { return state_->next(stop); }
CameraFormat V4l2Camera::format() const { return state_->format; }
CameraCounters V4l2Camera::counters() const { return state_->counts; }
void V4l2Camera::stop() { state_->shutdown(); }
} // namespace yolo

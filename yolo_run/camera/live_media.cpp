#include "camera/live_media.h"
#include "image/dma_buffer.h"
#include "image/mpp_dma_buffer.h"
#include "profiling/trace.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <mpp_buffer.h>
#include <mpp_frame.h>
#include <mpp_meta.h>
#include <mpp_packet.h>
#include <opencv2/imgproc.hpp>
#include <poll.h>
#include <rk_mpi.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace yolo::live {
using Json = nlohmann::json;
double nowMs() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) throw std::runtime_error("CLOCK_MONOTONIC failed");
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
namespace {
void check(MPP_RET rc, const char* operation) {
    if (rc != MPP_OK) throw std::runtime_error(std::string(operation) + ": " + std::to_string(rc));
}
void rgaCheck(IM_STATUS rc, const char* operation) {
    if (rc != IM_STATUS_SUCCESS && rc != IM_STATUS_NOERROR)
        throw std::runtime_error(std::string(operation) + ": " + imStrError(rc));
}
int call(int fd, unsigned long request, void* arg) {
    int rc; do { rc = ioctl(fd, request, arg); } while (rc < 0 && errno == EINTR); return rc;
}
void io(int fd, unsigned long request, void* arg, const char* operation) {
    if (call(fd, request, arg) < 0) throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}
// 解码输出池的一个槽：allocation、MPP 导入和输出帧包装一起存活。
// 外部 Nv12FrameView::owner 持有整个 Surface，才能阻止下一次解码覆盖它。
struct Surface {
    std::shared_ptr<DmaBuffer> dma;
    MppDmaBuffer buffer;
    MppFrame frame = nullptr;
    // 按官方示例预留每对齐像素 4 字节；这是容量预留，实际格式仍是 NV12。
    // 真正的可见尺寸、水平/垂直 stride 在每次 MPP 返回后单独检查。
    Surface()
        : dma(std::make_shared<DmaBuffer>(800U * 608U * 4, "/dev/dma_heap/system-uncached")),
          buffer(dma) {
        check(mpp_frame_init(&frame), "mpp_frame_init");
        mpp_frame_set_buffer(frame, buffer.get());
    }
    // 先清理 MPP 包装，再由成员 shared_ptr 归还底层分配；不能手动提前 close FD。
    ~Surface() {
        if (frame) mpp_frame_deinit(&frame);
        // 成员逆序析构：buffer 撤销 MPP 注册，最后 dma 释放分配引用。
    }
};
Stamp parseStamp(const Json& j) {
    Stamp s;
    s.sequence = j.at("sequence"); s.timestamp_ms = j.at("timestamp_ms");
    s.raw_timestamp_ms = j.value("raw_timestamp_ms", s.timestamp_ms);
    s.flags = j.value("timestamp_flags", 0U); s.clock = j.value("timestamp_clock", "fixture"); return s;
}
}
struct PacketStorage {
    uint32_t index = 0;
    std::shared_ptr<DmaBuffer> dma;
    MppDmaBuffer buffer;
    // 末尾额外保留 256 字节并逐包清零；MppPacket 有效长度仍由 bytes 字段限定。
    explicit PacketStorage(size_t capacity)
        : dma(std::make_shared<DmaBuffer>(capacity + 256, "/dev/dma_heap/system-uncached")),
          buffer(dma) {}
    // buffer 先于 dma 析构；包池占用仍按 PacketStorage 的引用数判断。
};
const unsigned char* EncodedFrame::data() const {
    return storage ? static_cast<const unsigned char*>(storage->dma->data) : nullptr;
}
void EncodedFrame::withCpuRead(const std::function<void(const unsigned char*, size_t)>& reader) const {
    if (!storage || !storage->dma || bytes > storage->dma->size)
        throw std::runtime_error("invalid encoded frame CPU read");
    auto retained = storage;
    CpuAccess read(retained->dma, DMA_BUF_SYNC_READ);
    reader(data(), bytes);
    read.finish();
}
PacketPool::PacketPool(size_t capacity, size_t slots) : slots_(slots) {
    if (!capacity || capacity > 64 * 1024 * 1024) throw std::runtime_error("invalid MJPEG packet capacity");
    for (size_t i = 0; i < slots_.size(); ++i) {
        slots_[i] = std::make_shared<PacketStorage>(capacity);
        slots_[i]->index = i;
    }
}
std::shared_ptr<PacketStorage> PacketPool::acquire(uint64_t frame_id) {
    // 池本身占一份引用；只有没有 EncodedFrame 消费者时才能写下一包。
    // acquire 仅由单一生产者调用，引用数检查不替代通用的多生产者锁。
    for (size_t i = 0; i < slots_.size(); ++i) {
        const auto &slot = slots_[i];
        if (slot.use_count() == 1)
            return trace::lease(slot, trace::PacketPool, i, frame_id);
    }
    return {};
}

namespace {
class CameraSource final : public FrameSource {
public:
    ~CameraSource() override {
        // 必须先确认驱动停止写入，再释放排队中的 DMA-BUF。
        if (streaming_) {
            auto type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            if (call(fd_, VIDIOC_STREAMOFF, &type) < 0) {
                std::cerr << "fatal: MJPEG STREAMOFF failed before DMA release\n"; std::terminate();
            }
        }
        if (fd_ >= 0) close(fd_);
    }
    void initialize(const std::string& device) {
        fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("open camera " + device + ": " + std::strerror(errno));
        v4l2_capability caps{}; io(fd_, VIDIOC_QUERYCAP, &caps, "QUERYCAP");
        const auto cap = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
        if (!(cap & V4L2_CAP_VIDEO_CAPTURE) || !(cap & V4L2_CAP_STREAMING))
            throw std::runtime_error("MJPEG requires single-planar streaming capture");
        v4l2_format f{}; f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        f.fmt.pix.width = 800; f.fmt.pix.height = 600;
        f.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; f.fmt.pix.field = V4L2_FIELD_NONE;
        io(fd_, VIDIOC_S_FMT, &f, "S_FMT MJPEG"); io(fd_, VIDIOC_G_FMT, &f, "G_FMT MJPEG");
        const auto& p = f.fmt.pix;
        const auto enc = p.ycbcr_enc ? p.ycbcr_enc : uint32_t(V4L2_MAP_YCBCR_ENC_DEFAULT(p.colorspace));
        // UVC reports sRGB + DEFAULT for MJPG. JPEG is full range, unlike raw
        // YUYV; v4l2-ctl also interprets this compressed format as full range.
        const auto range = p.quantization ? p.quantization : uint32_t(V4L2_QUANTIZATION_FULL_RANGE);
        if (p.width != 800 || p.height != 600 || p.pixelformat != V4L2_PIX_FMT_MJPEG ||
            p.field != V4L2_FIELD_NONE || enc != V4L2_YCBCR_ENC_601 || range != V4L2_QUANTIZATION_FULL_RANGE ||
            !p.sizeimage || p.sizeimage > 64 * 1024 * 1024)
            throw std::runtime_error("camera requires progressive 800x600 MJPEG BT.601 full range");
        v4l2_streamparm parm{}; parm.type = f.type;
        parm.parm.capture.timeperframe.numerator = 1; parm.parm.capture.timeperframe.denominator = 30;
        io(fd_, VIDIOC_S_PARM, &parm, "S_PARM 30 FPS"); io(fd_, VIDIOC_G_PARM, &parm, "G_PARM");
        const auto t = parm.parm.capture.timeperframe;
        if (!t.numerator || !t.denominator) throw std::runtime_error("invalid camera FPS");
        const double fps = double(t.denominator) / t.numerator;
        v4l2_requestbuffers req{}; req.type = f.type; req.memory = V4L2_MEMORY_DMABUF; req.count = 4;
        io(fd_, VIDIOC_REQBUFS, &req, "REQBUFS DMABUF");
        if (req.count != 4) throw std::runtime_error("camera_live requires four DMA-BUF queue slots");
        // 四个槽交给 V4L2，其余槽供出队、等待解码和 MPP 在途包轮换；没有逐帧分配。
        trace::emit(trace::Counter, trace::PoolCapacity, 0, 0, 8, trace::PacketPool);
        pool_ = std::make_unique<PacketPool>(p.sizeimage, 8);
        size_image_ = p.sizeimage;
        for (unsigned i = 0; i < buffers_.size(); ++i) {
            buffers_[i] = pool_->acquire();
            auto b = buffer(i);
            io(fd_, VIDIOC_QBUF, &b, "initial QBUF DMABUF");
        }
        auto type = f.type; io(fd_, VIDIOC_STREAMON, &type, "STREAMON"); streaming_ = true;
        metadata_ = {{"source", "v4l2"}, {"device", device}, {"width", 800}, {"height", 600},
            {"pixel_format", "MJPG"}, {"negotiated_fps", fps}, {"color_range", "BT601_full"},
            {"buffers", 4}, {"size_image", p.sizeimage}, {"capture_io", "dmabuf"}, {"packet_pool_buffers", 8},
            {"v4l2_colorspace", p.colorspace}, {"v4l2_quantization", p.quantization}, {"v4l2_ycbcr_enc", p.ycbcr_enc}};
    }
    std::optional<EncodedFrame> next(const std::atomic<bool>& stop) override {
        unsigned timeouts = 0;
        while (!stop.load()) {
            const auto poll_attempt = trace::emit(trace::Begin, trace::Poll);
            pollfd p{fd_, POLLIN, 0}; const int rc = poll(&p, 1, 1000);
            const int poll_errno = errno;
            trace::emit(trace::End, trace::Poll, 0, poll_attempt, rc > 0 ? 1 : 0);
            errno = poll_errno;
            if (rc < 0) { if (errno == EINTR) continue; throw std::runtime_error("camera poll failed"); }
            if (!rc) { if (++timeouts == 3) throw std::runtime_error("camera three one-second timeouts"); continue; }
            if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) throw std::runtime_error("camera disconnected/poll error");
            timeouts = 0;
            v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_DMABUF;
            const auto attempt = trace::emit(trace::Begin, trace::Dqbuf);
            const int dq = call(fd_, VIDIOC_DQBUF, &b);
            const auto saved_errno = errno;
            const uint64_t id = dq == 0 ? dequeued_ + 1 : 0;
            trace::emit(trace::End, trace::Dqbuf, id, attempt, dq == 0 ? 0 : saved_errno);
            if (dq < 0) {
                errno = saved_errno;
                if (errno == EAGAIN) continue;
                throw std::runtime_error("camera DQBUF failed");
            }
            if (b.index >= 4) throw std::runtime_error("camera invalid buffer index");
            // QBUF 始终在采集线程执行。正常帧换入空闲 DMA，丢帧则归还原 DMA。
            struct Return {
                CameraSource& source;
                unsigned index;
                ~Return() {
                    auto next = source.buffer(index);
                    if (call(source.fd_, VIDIOC_QBUF, &next) < 0) {
                        std::cerr << "fatal: MJPEG QBUF failed\n"; std::terminate();
                    }
                }
            } release{*this, b.index};
            ++dequeued_;
            const double arrival = nowMs();
            trace::emit(trace::Instant, trace::Capture, id, 0, b.sequence);
            trace::emit(trace::Instant, trace::SourceRawSequence, id, 0, b.sequence);
            trace::emit(trace::Instant, trace::SourceTime, id, 0,
                        uint64_t(b.timestamp.tv_sec) * 1000000000 + b.timestamp.tv_usec * 1000);
            trace::emit(trace::Instant, trace::SourceFlags, id, 0, b.flags);
            trace::emit(trace::Instant, trace::CaptureSlot, id, 0, b.index);
            trace::emit(trace::Instant, trace::PacketPool, id, 0, 0, buffers_[b.index]->index);
            if (have_sequence_) {
                const uint32_t delta = b.sequence - last_sequence_;
                if (delta > 1 && delta < 0x80000000U) {
                    gaps_ += delta - 1;
                    trace::emit(trace::Instant, trace::SourceGap, id, 0, delta - 1);
                }
                extended_sequence_ += delta;
            } else { extended_sequence_ = b.sequence; have_sequence_ = true; }
            last_sequence_ = b.sequence;
            if ((b.flags & V4L2_BUF_FLAG_ERROR) || b.bytesused < 4 || b.bytesused > size_image_) {
                trace::emit(trace::Drop, trace::Rejected, id, 0, 1);
                ++corrupt_;
                if (++bad_run_ >= 12)
                    throw std::runtime_error("12 consecutive corrupt MJPEG frames");
                continue;
            }
            const bool mono = (b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
            const double raw = b.timestamp.tv_sec * 1000.0 + b.timestamp.tv_usec / 1000.0;
            const double timestamp = mono ? raw : nowMs();
            if (last_ts_ && timestamp <= last_ts_) {
                trace::emit(trace::Drop, trace::Rejected, id, 0, 2);
                ++timestamp_dropped_;
                continue;
            }
            if (!first_ts_) first_ts_ = timestamp;
            last_ts_ = timestamp;
            trace::emit(trace::Instant, trace::Valid, id, 0, extended_sequence_);
            trace::Scope search(trace::BufferSearch, id, trace::PacketPool);
            auto spare = pool_->acquire();
            search.end();
            if (!spare) {
                trace::emit(trace::Drop, trace::Decode, id, 0, 3);
                ++pool_dropped_;
                continue;
            }
            auto storage = std::exchange(buffers_[b.index], std::move(spare));
            // 压缩数据已由 V4L2 写进同一块 DMA-BUF；只保留已验证的包尾填零契约。
            CpuAccess write(storage->dma, DMA_BUF_SYNC_WRITE);
            std::memset(static_cast<unsigned char*>(storage->dma->data) + b.bytesused, 0, 256);
            write.finish();
            bad_run_ = 0;
            return EncodedFrame{storage,
                                b.bytesused,
                                {extended_sequence_, timestamp, raw, b.flags,
                                 mono ? "v4l2_monotonic" : "host_monotonic", id, 0, arrival}};
        }
        return {};
    }
    Json info() const override { return metadata_; }
    Json counters() const override {
        return {{"dequeued", dequeued_}, {"corrupt_dropped", corrupt_}, {"packet_pool_dropped", pool_dropped_},
            {"timestamp_dropped", timestamp_dropped_}, {"driver_sequence_gaps", gaps_},
            {"compressed_copy_bytes", 0}, {"observed_source_fps", last_ts_ > first_ts_ ?
                (dequeued_ - 1 + gaps_) * 1000.0 / (last_ts_ - first_ts_) : 0}};
    }
private:
    v4l2_buffer buffer(unsigned index) const {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_DMABUF; b.index = index;
        b.m.fd = buffers_[index]->dma->fd; b.length = buffers_[index]->dma->size;
        return b;
    }
    int fd_ = -1; bool streaming_ = false, have_sequence_ = false;
    std::array<std::shared_ptr<PacketStorage>, 4> buffers_;
    std::unique_ptr<PacketPool> pool_;
    Json metadata_;
    size_t size_image_ = 0;
    uint32_t last_sequence_ = 0;
    int64_t extended_sequence_ = 0;
    uint64_t dequeued_ = 0, corrupt_ = 0, pool_dropped_ = 0, gaps_ = 0, timestamp_dropped_ = 0;
    unsigned bad_run_ = 0;
    double first_ts_ = 0, last_ts_ = 0;
};
class ReplaySource final : public FrameSource {
public:
    ReplaySource(const std::filesystem::path& path, bool paced) : directory_(path.parent_path()), stream_(path), paced_(paced) {
        std::string line;
        if (!std::getline(stream_, line)) throw std::runtime_error("empty replay manifest");
        metadata_ = Json::parse(line);
        if (metadata_.value("type", "") != "mjpeg_fixture" || metadata_.at("width") != 800 || metadata_.at("height") != 600)
            throw std::runtime_error("unsupported MJPEG fixture metadata");
        pool_ = std::make_unique<PacketPool>(metadata_.value("max_packet_bytes", 800U * 600U * 2U));
        metadata_["source"] = paced_ ? "paced_replay" : "unpaced_replay";
        trace::emit(trace::Counter, trace::PoolCapacity, 0, 0, 4, trace::PacketPool);
    }
    std::optional<EncodedFrame> next(const std::atomic<bool>& stop) override {
        std::string line;
        if (stop.load() || !std::getline(stream_, line)) return {};
        const auto row = Json::parse(line);
        auto stamp = parseStamp(row);
        if (!std::isfinite(stamp.timestamp_ms) || stamp.timestamp_ms < 0 || stamp.sequence < 0 ||
            (count_ && (stamp.timestamp_ms <= last_original_ || stamp.sequence <= last_sequence_)))
            throw std::runtime_error("fixture timestamps/sequences must strictly increase");
        if (!count_) { base_ = stamp.timestamp_ms; start_ = nowMs(); }
        last_original_ = stamp.timestamp_ms; last_sequence_ = stamp.sequence;
        // 将夹具相对时间平移到本机 CLOCK_MONOTONIC，保留源帧间隔，模拟实时到达。
        stamp.timestamp_ms = start_ + stamp.timestamp_ms - base_;
        stamp.clock = "replay_monotonic";
        if (paced_) while (!stop.load() && nowMs() < stamp.timestamp_ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (stop.load()) return {};
        auto storage = pool_->acquire(count_ + 1);
        if (!storage) throw std::runtime_error("replay packet pool exhausted");
        const std::filesystem::path filename(row.at("file").get<std::string>());
        if (filename.has_parent_path() || filename.is_absolute()) throw std::runtime_error("fixture file must be a basename");
        const auto path = directory_ / filename;
        const auto bytes = std::filesystem::file_size(path);
        if (bytes < 4 || bytes > storage->dma->size - 256) throw std::runtime_error("invalid replay packet size");
        CpuAccess write(storage->dma, DMA_BUF_SYNC_WRITE);
        std::ifstream file(path, std::ios::binary);
        if (!file.read(static_cast<char*>(storage->dma->data), bytes)) throw std::runtime_error("cannot read fixture JPEG");
        std::memset(static_cast<unsigned char*>(storage->dma->data) + bytes, 0, 256); write.finish();
        ++count_; bytes_ += bytes;
        stamp.frame_id = count_;
        stamp.arrival_ms = nowMs();
        trace::emit(trace::Instant, trace::Capture, count_, 0, stamp.sequence);
        if (trace::enabled()) {
            const double raw_ns = stamp.raw_timestamp_ms * 1e6;
            if (raw_ns >= 0 && raw_ns < std::ldexp(1.0, 64))
                trace::emit(trace::Instant, trace::SourceTime, count_, 0, static_cast<uint64_t>(raw_ns));
        }
        trace::emit(trace::Instant, trace::Valid, count_, 0, stamp.sequence);
        return EncodedFrame{storage, bytes, stamp};
    }
    Json info() const override { return metadata_; }
    Json counters() const override { return {{"dequeued", count_}, {"compressed_copy_bytes", bytes_},
        {"observed_source_fps", count_ > 1 && last_original_ > base_ ? (count_ - 1) * 1000 / (last_original_ - base_) : 0}}; }
private:
    std::filesystem::path directory_;
    std::ifstream stream_;
    std::unique_ptr<PacketPool> pool_;
    Json metadata_;
    bool paced_;
    double base_ = 0, start_ = 0, last_original_ = 0;
    int64_t last_sequence_ = -1;
    uint64_t count_ = 0, bytes_ = 0;
};
}
std::unique_ptr<FrameSource> cameraSource(const std::string& device) {
    auto source = std::make_unique<CameraSource>(); source->initialize(device); return source;
}
std::unique_ptr<FrameSource> replaySource(const std::filesystem::path& manifest, bool paced) {
    return std::make_unique<ReplaySource>(manifest, paced);
}

class MjpegDecoder::Impl {
public:
    explicit Impl(int timeout, size_t pool_size) : surfaces(pool_size) {
        trace::emit(trace::Counter, trace::PoolCapacity, 0, 0, pool_size, trace::SurfacePool);
        try {
            check(mpp_check_support_format(MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG), "MPP MJPEG support");
            // 启动时分配整个 Surface 池。稳态只取已归还所有租约的槽。
            for (auto& slot : surfaces)
                slot = std::make_shared<Surface>();
            check(mpp_create(&ctx, &api), "mpp_create");
            check(mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG), "mpp_init MJPEG");
            RK_S64 timeout_ms = timeout;
            check(api->control(ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout_ms), "MPP output timeout");
            MppFrameFormat format = MPP_FMT_YUV420SP;
            check(api->control(ctx, MPP_DEC_SET_OUTPUT_FORMAT, &format), "MPP output NV12");
        } catch (...) {
            if (ctx) mpp_destroy(ctx);
            throw;
        }
    }
    ~Impl() {
        if (ctx) {
            // 先结束 MPP 访问，再销毁数组持有的 Surface；外部帧仍可保留自己的引用。
            if (api->reset(ctx) != MPP_OK || mpp_destroy(ctx) != MPP_OK) {
                std::cerr << "fatal: MPP cleanup failed before surface release\n"; std::terminate();
            }
        }
    }
    std::optional<DecodedFrame> decode(const EncodedFrame& packet) {
        const auto start = nowMs();
        if (!packet.storage || packet.bytes < 4 || packet.bytes > packet.storage->dma->size - 256)
            throw std::runtime_error("invalid MJPEG packet lease/capacity");
        std::shared_ptr<Surface> surface;
        // 检查的是 Surface 租约，不是 dma 引用数。池独占时为 1；两路消费者
        // 任何一方仍持有 owner 都不复用，耗尽时把丢帧决策交给流水线并计数。
        trace::Scope search(trace::BufferSearch, packet.stamp.frame_id, trace::SurfacePool);
        for (size_t i = 0; i < surfaces.size(); ++i) {
            const auto &slot = surfaces[i];
            if (slot.use_count() == 1) {
                surface = trace::lease(slot, trace::SurfacePool, i, packet.stamp.frame_id);
                break;
            }
        }
        search.end();
        if (!surface) return {};
        MppPacket input = nullptr;
        check(mpp_packet_init_with_buffer(&input, packet.storage->buffer.get()), "mpp_packet_init_with_buffer");
        struct PacketEnd { MppPacket* p; ~PacketEnd() { mpp_packet_deinit(p); } } end{&input};
        mpp_packet_set_length(input, packet.bytes); mpp_packet_set_pts(input, packet.stamp.sequence);
        // 把本次输出明确绑定到选中的外部 Surface；提交成功尚不能归还输入和输出。
        mpp_meta_set_frame(mpp_packet_get_meta(input), KEY_OUTPUT_FRAME, surface->frame);
        trace::Scope submit(trace::Submit, packet.stamp.frame_id);
        check(api->decode_put_packet(ctx, input), "MPP JPEG submit");
        submit.end();
        trace::Scope receive(trace::GetFrame, packet.stamp.frame_id);
        MppFrame output = nullptr;
        const auto rc = api->decode_get_frame(ctx, &output);
        receive.end();
        if (rc != MPP_OK || !output) {
            // 无法确认完成时不展开栈并回收输入/输出，避免设备仍访问已被重用的内存。
            std::cerr << "fatal: MPP JPEG completion uncertain: " << rc << '\n'; std::terminate();
        }
        // 完成通知还要对应本包、本帧、本 buffer 和源序号；不能仅凭非空就向下游发布。
        MppPacket consumed = nullptr;
        check(mpp_meta_get_packet(mpp_frame_get_meta(output), KEY_INPUT_PACKET, &consumed), "MPP returned input packet");
        if (consumed != input) throw std::runtime_error("MPP returned input packet identity mismatch");
        if (output != surface->frame || mpp_frame_get_buffer(output) != surface->buffer.get() ||
            mpp_frame_get_errinfo(output) || mpp_frame_get_discard(output) || mpp_frame_get_info_change(output) ||
            mpp_frame_get_width(output) != 800 || mpp_frame_get_height(output) != 600 ||
            mpp_frame_get_fmt(output) != MPP_FMT_YUV420SP || mpp_frame_get_pts(output) != packet.stamp.sequence)
            throw std::runtime_error("MPP JPEG output identity, NV12, dimensions, PTS or error validation failed");
        // NV12 UV 起点是 ws*hs，不是可见宽高 800*600；描述和容量必须同时满足。
        const int ws = mpp_frame_get_hor_stride(output), hs = mpp_frame_get_ver_stride(output);
        if (ws < 800 || hs < 600 || ws > 16384 || hs > 16384 || ws % 2 || hs % 2 ||
            size_t(ws) * hs * 3 / 2 > surface->dma->size)
            throw std::runtime_error("MPP JPEG invalid output stride/capacity");
        // 本版本 JPEG 解码器不填 color_range；按已用 CPU 夹具核验的 JPEG BT.601 full 契约。
        // owner=surface 保住帧占用，dma=surface->dma 保住分配，二者职责不同。
        return DecodedFrame{{surface, surface->dma, 800, 600, ws, hs, true}, packet.stamp, nowMs() - start, {}};
    }
    MppCtx ctx = nullptr;
    MppApi* api = nullptr;
    std::vector<std::shared_ptr<Surface>> surfaces;
};
MjpegDecoder::MjpegDecoder(int timeout, size_t pool_size)
    : impl_(std::make_unique<Impl>(timeout, pool_size)) {}
MjpegDecoder::~MjpegDecoder() = default;
std::optional<DecodedFrame> MjpegDecoder::decode(const EncodedFrame& p) { return impl_->decode(p); }

// CPU 画框既读取也写入 BGR，用 RW 区间覆盖；它不代替前面的 RGA 设备完成同步。
class RenderAccess {
public:
    explicit RenderAccess(std::shared_ptr<DmaBuffer> buffer) : access(std::move(buffer), DMA_BUF_SYNC_RW) {}
    CpuAccess access;
};
Renderer::Renderer() = default;
Renderer::~Renderer() = default;
void Renderer::finish() { if (access_) { access_->access.finish(); access_.reset(); } }
void Renderer::convertForEncode(const Nv12FrameView& f) {
    if (access_ || !next_ || !f.owner || !f.dma || f.full_range || f.width != 800 || f.height != 600 ||
        f.width_stride != 800 || f.height_stride != 608 || f.dma->size < 800U * 608U * 3 / 2)
        throw std::runtime_error("invalid encode conversion/lifetime");
    const auto& bgr = buffers_[(next_ - 1) % 2];
    if (!bgr || !bgr->healthy || !f.dma->healthy) throw std::runtime_error("encode conversion requires DMA BGR");
    auto src = wrapbuffer_handle(bgr->handle, 800, 600, RK_FORMAT_BGR_888);
    auto dst = wrapbuffer_handle(f.dma->handle, 800, 600, RK_FORMAT_YCbCr_420_SP, 800, 608);
    dst.color_space_mode = IM_RGB_TO_YUV_BT601_LIMIT;
    rgaCheck(imcheck(src, dst, {}, {}, 0), "encode NV12 layout");
    im_opt_t options{}; options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
    trace::Scope rga_call(trace::Rga);
    const auto rc = improcess(src, dst, {}, {}, {}, {}, -1, nullptr, &options, IM_SYNC);
    rga_call.end();
    if (rc != IM_STATUS_SUCCESS && rc != IM_STATUS_NOERROR) {
        std::cerr << "fatal: encode conversion RGA completion uncertain\n"; std::terminate();
    }
}
cv::Mat Renderer::convert(const DecodedFrame& frame) {
    if (access_) throw std::runtime_error("previous render CPU access still active");
    const auto& f = frame.image;
    const auto roi=nv12SourceRegion(f,frame.stabilization.roi);
    if (f.width != 800 || f.height != 600) throw std::runtime_error("invalid render frame dimensions");
    // 输出目标是独立 BGR 双缓冲。NV12 只读源仍由调用方 frame.owner 保持占用。
    auto& buffer = buffers_[next_++ % 2];
    if (!buffer) buffer = std::make_shared<DmaBuffer>(800U * 600U * 3, "/dev/dma_heap/system-uncached");
    auto src = wrapbuffer_handle(f.dma->handle, f.width, f.height, RK_FORMAT_YCbCr_420_SP, f.width_stride, f.height_stride);
    auto dst = wrapbuffer_handle(buffer->handle, 800, 600, RK_FORMAT_BGR_888);
    dst.color_space_mode = f.full_range ? IM_YUV_TO_RGB_BT601_FULL : IM_YUV_TO_RGB_BT601_LIMIT;
    const im_rect sr{roi.x,roi.y,roi.width,roi.height}, dr{0,0,800,600};
    rgaCheck(imcheck(src, dst, sr, dr, 0), "render NV12 layout");
    im_opt_t options{}; options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
    trace::Scope rga_call(trace::Rga);
    const auto rc = improcess(src, dst, {}, sr, dr, {}, -1, nullptr, &options, IM_SYNC);
    rga_call.end();
    if (rc != IM_STATUS_SUCCESS && rc != IM_STATUS_NOERROR) {
        std::cerr << "fatal: render RGA completion uncertain\n"; std::terminate();
    }
    // IM_SYNC 成功后才能 CPU 访问目标；返回 Mat 不拥有这块外部像素内存，
    // 调用者须在 finish() 前完成当前 CPU 访问，不将视图交给异步消费者后立即复用。
    access_ = std::make_unique<RenderAccess>(buffer);
    return cv::Mat(600, 800, CV_8UC3, buffer->data);
}
} // namespace yolo::live

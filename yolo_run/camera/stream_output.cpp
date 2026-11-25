#include "camera/stream_output.h"
#include "image/mpp_dma_buffer.h"
#include "image/dma_buffer.h"
#include "image/frame_lease.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <thread>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <rk_mpi.h>
#include <rk_venc_cfg.h>
#include <mpp_buffer.h>
#include <mpp_frame.h>
#include <mpp_meta.h>
#include <mpp_packet.h>
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/mem.h>
}

namespace yolo::live {
namespace {
constexpr size_t PacketCapacity = 2U * 1024 * 1024;
void check(MPP_RET rc, const char* what) {
    if (rc != MPP_OK) throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
}
void avCheck(int rc, const char* what) {
    if (rc >= 0) return;
    char text[AV_ERROR_MAX_STRING_SIZE]{}; av_strerror(rc, text, sizeof(text));
    throw std::runtime_error(std::string(what) + ": " + text);
}
bool hasNal(const std::vector<uint8_t>& bytes, uint8_t type) {
    for (size_t i = 0; i + 3 < bytes.size(); ++i) {
        if (bytes[i] || bytes[i + 1]) continue;
        size_t n = i + 2;
        if (bytes[n] == 0) ++n;
        if (n + 1 < bytes.size() && bytes[n] == 1 && (bytes[n + 1] & 31) == type) return true;
    }
    return false;
}
struct Histogram {
    std::array<uint64_t, 4097> bins{};
    uint64_t count = 0;
    double sum = 0, maximum = 0;
    void add(double value) {
        if (!std::isfinite(value) || value < 0) return;
        ++count; sum += value; maximum = std::max(maximum, value);
        ++bins[std::min<size_t>(4096, size_t(std::ceil(value)))];
    }
    nlohmann::json json() const {
        auto percentile = [&](double p) { uint64_t n = 0; for (size_t i = 0; i < bins.size(); ++i) {
            n += bins[i]; if (count && n >= std::ceil(count * p)) return i; } return size_t(0); };
        return {{"count", count}, {"mean_ms", count ? sum / count : 0}, {"max_ms", maximum},
            {"p50_ms", percentile(.5)}, {"p95_ms", percentile(.95)}, {"p99_ms", percentile(.99)},
            {"quantile_resolution_ms", 1}, {"overflow_bin_ms", 4096}};
    }
};
}
struct EncodeFramePool::Slot {
    std::shared_ptr<DmaBuffer> dma = std::make_shared<DmaBuffer>(800U * 608U * 3 / 2, "/dev/dma_heap/system-uncached");
    std::weak_ptr<void> lease;
    Slot() {
        CpuAccess access(dma, DMA_BUF_SYNC_WRITE);
        std::memset(dma->data, 16, 800U * 608U);
        std::memset(static_cast<uint8_t*>(dma->data) + 800U * 608U, 128, 800U * 608U / 2);
        access.finish();
    }
};
EncodeFramePool::EncodeFramePool() {
    trace::emit(trace::Counter, trace::PoolCapacity, 0, 0, 3, trace::EncodePool);
    for (auto &slot : slots_)
        slot = std::make_shared<Slot>();
}
std::optional<Nv12FrameView> EncodeFramePool::acquire() {
    // 仅输出生产者调用 acquire；编码 Worker 只归还租约，不参与选槽。
    for (auto& slot : slots_) if (slot->lease.expired() && slot->dma->healthy) {
            auto owner = std::make_shared<FrameLease>(
                trace::lease(slot, trace::EncodePool, &slot - slots_.data(), trace::currentFrame()));
            slot->lease = owner;
            return Nv12FrameView{owner, slot->dma, 800, 600, 800, 608, false};
    }
    return {};
}

struct MppH264Encoder::Impl {
    MppCtx ctx = nullptr; MppApi* api = nullptr; MppEncCfg cfg = nullptr;
    // 长期注册只持有 allocation；active 才表示本次输入仍被设备使用。
    std::map<int, MppDmaBuffer> bindings;
    std::shared_ptr<DmaBuffer> output_dma;
    MppDmaBuffer output_buffer;
    MppPacket offered = nullptr;
    Nv12FrameView active;
    H264Packet output;
    double origin = -1, last_idr = -1;
    int64_t last_pts = -1;

    void init(int bitrate, int timeout) {
        if (bitrate < 64000 || bitrate > 14000000 || timeout < 1 || timeout > 10000)
            throw std::runtime_error("invalid encoder bitrate/timeout");
        check(mpp_check_support_format(MPP_CTX_ENC, MPP_VIDEO_CodingAVC), "MPP H264 support");
        output.bytes.reserve(PacketCapacity);
        output_dma = std::make_shared<DmaBuffer>(PacketCapacity, "/dev/dma_heap/system-uncached");
        output_buffer = MppDmaBuffer(output_dma, "encoder import DMA-BUF");
        check(mpp_create(&ctx, &api), "encoder mpp_create");
        RK_S64 wait = timeout;
        check(api->control(ctx, MPP_SET_INPUT_TIMEOUT, &wait), "encoder input timeout");
        check(api->control(ctx, MPP_SET_OUTPUT_TIMEOUT, &wait), "encoder output timeout");
        check(mpp_init(ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC), "encoder mpp_init");
        check(mpp_enc_cfg_init(&cfg), "encoder cfg init");
        check(api->control(ctx, MPP_ENC_GET_CFG, cfg), "encoder get cfg");
        auto set = [&](const char* name, int value) { check(mpp_enc_cfg_set_s32(cfg, name, value), name); };
        set("codec:type", MPP_VIDEO_CodingAVC);
        set("prep:width", 800); set("prep:height", 600);
        set("prep:hor_stride", 800); set("prep:ver_stride", 608);
        set("prep:format", MPP_FMT_YUV420SP);
        set("prep:range", MPP_FRAME_RANGE_MPEG);
        set("prep:range_out", MPP_FRAME_RANGE_MPEG);
        set("prep:colorspace", MPP_FRAME_SPC_SMPTE170M);
        set("prep:colorprim", MPP_FRAME_PRI_SMPTE170M);
        set("prep:colortrc", MPP_FRAME_TRC_SMPTE170M);
        set("rc:mode", MPP_ENC_RC_MODE_CBR);
        set("rc:bps_target", bitrate); set("rc:bps_max", std::min(bitrate * 5 / 4, 14000000)); set("rc:bps_min", bitrate * 3 / 4);
        set("rc:fps_in_flex", 0); set("rc:fps_in_num", 30); set("rc:fps_in_denom", 1);
        set("rc:fps_out_flex", 0); set("rc:fps_out_num", 30); set("rc:fps_out_denom", 1);
        set("rc:drop_mode", MPP_ENC_RC_DROP_FRM_DISABLED); set("rc:gop", 30);
        set("h264:profile", 66); set("h264:level", 31);
        set("h264:cabac_en", 0); set("h264:trans8x8", 0);
        set("split:mode", 0);
        check(api->control(ctx, MPP_ENC_SET_CFG, cfg), "encoder set cfg");
        MppEncHeaderMode headers = MPP_ENC_HEADER_MODE_EACH_IDR;
        check(api->control(ctx, MPP_ENC_SET_HEADER_MODE, &headers), "encoder each IDR headers");
    }
    bool close() noexcept {
        // 超时不代表 DMA 完成。reset/destroy 无法确认设备停止时，保留整个 Impl，
        // 包括注册和 active 帧租约；析构入口将其隔离至进程退出。
        if (ctx) {
            if (api->reset(ctx) != MPP_OK) return false;
            if (mpp_destroy(ctx) != MPP_OK) return false;
            ctx = nullptr;
        }
        if (offered) mpp_packet_deinit(&offered);
        active = {};
        bindings.clear();
        if (cfg) { mpp_enc_cfg_deinit(cfg); cfg = nullptr; }
        output_buffer.reset();
        output_dma.reset();
        return true;
    }
    const H264Packet& encode(const StreamFrame& frame, bool force) {
        const auto& f = frame.image;
        if (!f.owner || !f.dma || !f.dma->healthy || f.full_range || f.width != 800 || f.height != 600 ||
            f.width_stride != 800 || f.height_stride != 608 || f.dma->size < 800U * 608U * 3 / 2 ||
            !std::isfinite(frame.stamp.timestamp_ms)) throw std::runtime_error("invalid encoder frame/layout/range");
        if (origin < 0) origin = frame.stamp.timestamp_ms;
        const int64_t pts = std::llround((frame.stamp.timestamp_ms - origin) * 90);
        if (pts <= last_pts) throw std::runtime_error("non-monotonic encoder timestamp");
        auto it = bindings.find(f.dma->fd);
        if (it == bindings.end()) {
            if (bindings.size() == 3) throw std::runtime_error("encoder expects a fixed three-slot input pool");
            it = bindings.emplace(f.dma->fd, MppDmaBuffer(f.dma, "encoder import DMA-BUF")).first;
        }
        if (force || last_idr < 0 || frame.stamp.timestamp_ms - last_idr >= 1000)
            check(api->control(ctx, MPP_ENC_SET_IDR_FRAME, nullptr), "encoder request IDR");
        MppFrame input = nullptr;
        check(mpp_frame_init(&input), "encoder frame init");
        mpp_frame_set_width(input, 800); mpp_frame_set_height(input, 600);
        mpp_frame_set_hor_stride(input, 800); mpp_frame_set_ver_stride(input, 608);
        mpp_frame_set_fmt(input, MPP_FMT_YUV420SP);
        mpp_frame_set_color_range(input, MPP_FRAME_RANGE_MPEG);
        mpp_frame_set_colorspace(input, MPP_FRAME_SPC_SMPTE170M);
        mpp_frame_set_pts(input, pts); mpp_frame_set_buffer(input, it->second.get());
        try {
            check(mpp_packet_init_with_buffer(&offered, output_buffer.get()), "encoder packet init");
            mpp_packet_set_length(offered, 0);
            check(mpp_meta_set_packet(mpp_frame_get_meta(input), KEY_OUTPUT_PACKET, offered), "encoder output packet");
            active = f; // 所有分片收齐到 EOI 前保留帧占用；异常时交给 close() 确认停止。
            check(api->encode_put_frame(ctx, input), "encode_put_frame");
        } catch (...) { mpp_frame_deinit(&input); throw; }
        mpp_frame_deinit(&input);
        output.bytes.clear(); output.pts90 = pts;
        bool complete = false;
        const double deadline = nowMs() + 10000; // Bounds even malformed endless partition output.
        while (!complete) {
            MppPacket packet = nullptr;
            check(api->encode_get_packet(ctx, &packet), "encode_get_packet");
            if (!packet) throw std::runtime_error("encoder output timeout");
            if (packet == offered) offered = nullptr;
            const auto length = mpp_packet_get_length(packet);
            complete = !mpp_packet_is_partition(packet) || mpp_packet_is_eoi(packet);
            const auto returned_pts = mpp_packet_get_pts(packet);
            const bool valid = length <= PacketCapacity - output.bytes.size() && returned_pts == pts;
            try { if (valid) {
                // CPU reads compressed output only. No raw-frame upload/copy.
                CpuAccess access(output_dma, DMA_BUF_SYNC_READ);
                const auto* data = static_cast<const uint8_t*>(mpp_packet_get_pos(packet));
                output.bytes.insert(output.bytes.end(), data, data + length);
                access.finish();
            } } catch (...) { mpp_packet_deinit(&packet); throw; }
            mpp_packet_deinit(&packet);
            if (!valid || nowMs() > deadline) throw std::runtime_error("encoder packet PTS/capacity/deadline mismatch");
        }
        active = {}; // 已收齐 EOI，设备不再读取输入；注册继续缓存，但槽位可以归还。
        last_pts = pts;
        if (output.bytes.empty()) throw std::runtime_error("empty H264 access unit");
        output.keyframe = hasNal(output.bytes, 5);
        if (output.keyframe) {
            if (!hasNal(output.bytes, 7) || !hasNal(output.bytes, 8)) throw std::runtime_error("IDR lacks SPS/PPS");
            last_idr = frame.stamp.timestamp_ms;
        }
        return output;
    }
};
MppH264Encoder::MppH264Encoder(int bitrate, int timeout) : impl_(std::make_unique<Impl>()) {
    try { impl_->init(bitrate, timeout); }
    catch (...) { if (!impl_->close()) (void)impl_.release(); throw; }
}
MppH264Encoder::~MppH264Encoder() {
    if (impl_ && !impl_->close()) {
        std::cerr << "encoder DMA completion uncertain: quarantining its fixed buffers until process exit\n";
        (void)impl_.release();
    }
}
const H264Packet& MppH264Encoder::encode(const StreamFrame& f, bool idr) { return impl_->encode(f, idr); }

struct MpegTsOutput::Impl {
    int fd = -1;
    sockaddr_in address{};
    AVFormatContext* format = nullptr;
    AVIOContext* io = nullptr;
    AVPacket* packet = nullptr;
    uint64_t sent = 0, failed = 0;
    bool write_failed = false;
    static int send(void* opaque, const uint8_t* data, int size) {
        auto& self = *static_cast<Impl*>(opaque);
        for (int pos = 0; pos < size; pos += 1316) {
            const int length = std::min(1316, size - pos);
            ssize_t rc;
            do { rc = sendto(self.fd, data + pos, length, MSG_DONTWAIT | MSG_NOSIGNAL,
                reinterpret_cast<const sockaddr*>(&self.address), sizeof(self.address)); } while (rc < 0 && errno == EINTR);
            if (rc == length) self.sent += length;
            else { ++self.failed; self.write_failed = true; }
        }
        // Consume failed datagrams: never retain stale bytes in AVIO or poison
        // all later writes. The next access unit is requested as a fresh IDR.
        return size;
    }
    void init(uint16_t port) {
        if (!port) throw std::runtime_error("invalid stream UDP port");
        fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (fd < 0) throw std::runtime_error("stream UDP socket failed");
        address.sin_family = AF_INET; address.sin_port = htons(port); address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        avCheck(avformat_alloc_output_context2(&format, nullptr, "mpegts", nullptr), "MPEG-TS context");
        auto* buffer = static_cast<uint8_t*>(av_malloc(1316));
        if (!buffer) throw std::bad_alloc();
        io = avio_alloc_context(buffer, 1316, 1, this, nullptr, send, nullptr);
        if (!io) { av_free(buffer); throw std::bad_alloc(); }
        io->max_packet_size = 1316;
        format->pb = io; format->flags |= AVFMT_FLAG_CUSTOM_IO | AVFMT_FLAG_FLUSH_PACKETS;
        format->max_delay = 0; format->avoid_negative_ts = AVFMT_AVOID_NEG_TS_DISABLED;
        auto* stream = avformat_new_stream(format, nullptr);
        if (!stream) throw std::bad_alloc();
        stream->time_base = {1, 90000}; stream->avg_frame_rate = {30, 1};
        auto* c = stream->codecpar; c->codec_type = AVMEDIA_TYPE_VIDEO; c->codec_id = AV_CODEC_ID_H264;
        c->width = 800; c->height = 600; c->format = AV_PIX_FMT_YUV420P;
        c->color_range = AVCOL_RANGE_MPEG; c->color_space = AVCOL_SPC_SMPTE170M;
        c->color_primaries = AVCOL_PRI_SMPTE170M; c->color_trc = AVCOL_TRC_SMPTE170M;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "mpegts_flags", "pat_pmt_at_frames+initial_discontinuity", 0);
        av_dict_set(&opts, "mpegts_copyts", "1", 0);
        const int rc = avformat_write_header(format, &opts); av_dict_free(&opts);
        avCheck(rc, "MPEG-TS header");
        packet = av_packet_alloc(); if (!packet) throw std::bad_alloc();
    }
    ~Impl() {
        if (packet) av_packet_free(&packet);
        if (format) { format->pb = nullptr; avformat_free_context(format); }
        if (io) { av_freep(&io->buffer); avio_context_free(&io); }
        if (fd >= 0) ::close(fd);
    }
};
MpegTsOutput::MpegTsOutput(uint16_t port) : impl_(std::make_unique<Impl>()) { impl_->init(port); }
MpegTsOutput::~MpegTsOutput() = default;
bool MpegTsOutput::write(const H264Packet& data) {
    auto& i = *impl_; i.write_failed = false;
    auto* p = i.packet; p->data = const_cast<uint8_t*>(data.bytes.data()); p->size = static_cast<int>(data.bytes.size());
    p->pts = p->dts = data.pts90; p->duration = 0; p->stream_index = 0; p->flags = data.keyframe ? AV_PKT_FLAG_KEY : 0;
    const int rc = av_write_frame(i.format, p); avio_flush(i.io);
    p->data = nullptr; p->size = 0;
    avCheck(rc, "MPEG-TS packet"); return !i.write_failed;
}
uint64_t MpegTsOutput::bytes() const { return impl_->sent; }
uint64_t MpegTsOutput::errors() const { return impl_->failed; }

struct StreamOutput::Impl {
    EncodeFramePool pool;
    LatestSlot<StreamFrame> pending{trace::EncodeQueue,
                                    [](const StreamFrame &f) { return f.stamp.frame_id; }};
    std::atomic<bool> enabled{true};
    std::thread worker;
    mutable std::mutex mutex;
    std::string status = "starting", error;
    uint64_t submitted = 0, no_buffer = 0, encoded = 0, keys = 0, sent = 0, errors = 0;
    Histogram conversion, encoding, latency;
    double start = nowMs();
    void run(uint16_t port, int bitrate, int timeout, const std::function<void()>& before, bool unpaced) {
        trace::thread(trace::Encoder);
        uint64_t encoding_frame = 0;
        try {
            MppH264Encoder encoder(bitrate, timeout);
            MpegTsOutput output(port);
            { std::lock_guard<std::mutex> lock(mutex); status = "running"; }
            bool force_idr = false;
            while (auto frame = pending.take()) {
                encoding_frame = frame->stamp.frame_id;
                trace::frame(encoding_frame);
                trace::Scope encode(trace::Encode);
                if (before) before();
                const double begin = nowMs();
                const auto& packet = encoder.encode(*frame, force_idr);
                encode.end();
                trace::emit(trace::Instant, trace::Encoded, encoding_frame);
                // MPP 已归还编码结果。后续发送失败记在传输侧，编码终态只记一次。
                encoding_frame = 0;
                const double completed = nowMs();
                force_idr = !output.write(packet);
                std::lock_guard<std::mutex> lock(mutex);
                ++encoded; keys += packet.keyframe; sent = output.bytes(); errors = output.errors();
                encoding.add(completed - begin); latency.add(completed - frame->stamp.latencyOrigin(unpaced));
            }
            std::lock_guard<std::mutex> lock(mutex); status = "stopped";
        } catch (const std::exception& e) {
            if (encoding_frame) trace::emit(trace::Error, trace::Encoded, encoding_frame);
            enabled.store(false); pending.close();
            std::lock_guard<std::mutex> lock(mutex); status = "error"; error = e.what();
            std::cerr << "stream disabled: " << error << '\n';
        }
    }
};
StreamOutput::StreamOutput(uint16_t port, int bitrate, int timeout, std::function<void()> before, bool unpaced)
    : impl_(std::make_unique<Impl>()) {
    impl_->worker = std::thread([this, port, bitrate, timeout, before = std::move(before), unpaced] {
        impl_->run(port, bitrate, timeout, before, unpaced);
    });
}
StreamOutput::~StreamOutput() { stop(); }
bool StreamOutput::submit(Renderer& renderer, const Stamp& stamp) {
    auto& i = *impl_;
    if (!i.enabled.load()) return false;
    auto image = i.pool.acquire();
    if (!image) {
        trace::emit(trace::Drop, trace::Encoded, stamp.frame_id, 0, 3);
        std::lock_guard<std::mutex> lock(i.mutex);
        ++i.no_buffer;
        return false;
    }
    const double begin = nowMs();
    renderer.convertForEncode(*image);
    const bool ok = i.pending.put(StreamFrame{std::move(*image), stamp});
    std::lock_guard<std::mutex> lock(i.mutex); i.submitted += ok; i.conversion.add(nowMs() - begin);
    return ok;
}
void StreamOutput::stop() {
    if (!impl_) return;
    impl_->enabled.store(false); impl_->pending.close();
    if (impl_->worker.joinable()) impl_->worker.join();
}
nlohmann::json StreamOutput::stats() const {
    const auto& i = *impl_; std::lock_guard<std::mutex> lock(i.mutex);
    return {{"status", i.status}, {"error", i.error}, {"submitted", i.submitted}, {"encoded", i.encoded}, {"keyframes", i.keys},
        {"pool_dropped", i.no_buffer}, {"pending_dropped", i.pending.dropped()}, {"shutdown_discarded", i.pending.cancelled()},
        {"sent_bytes", i.sent}, {"send_errors", i.errors}, {"encoded_fps", i.encoded * 1000 / std::max(1.0, nowMs() - i.start)},
        {"pool_capacity", 3}, {"queue_capacity", 1}, {"conversion", i.conversion.json()},
        {"encoding", i.encoding.json()}, {"capture_to_encoded", i.latency.json()}};
}
} // namespace yolo::live

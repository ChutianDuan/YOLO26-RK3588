#pragma once
#include "camera/live_media.h"
#include <functional>
#include <vector>

namespace yolo::live {
struct StreamFrame { Nv12FrameView image; Stamp stamp; };

// 固定三槽；分配存活和当前帧占用分离。MPP 注册可长期保留 allocation，
// 只有外部 Nv12FrameView::owner 尚未释放时，该槽才仍被当前帧占用。
class EncodeFramePool {
public:
    EncodeFramePool();
    std::optional<Nv12FrameView> acquire();
private:
    struct Slot;
    std::array<std::shared_ptr<Slot>, 3> slots_;
};
struct H264Packet {
    std::vector<uint8_t> bytes;
    int64_t pts90 = 0;
    bool keyframe = false;
};
class MppH264Encoder {
public:
    explicit MppH264Encoder(int bitrate = 2000000, int timeout_ms = 1000);
    ~MppH264Encoder();
    // Returned compressed storage is reused by the next call. One caller only.
    const H264Packet& encode(const StreamFrame&, bool force_idr = false);
    MppH264Encoder(const MppH264Encoder&) = delete;
    MppH264Encoder& operator=(const MppH264Encoder&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class MpegTsOutput {
public:
    explicit MpegTsOutput(uint16_t port);
    ~MpegTsOutput();
    bool write(const H264Packet&);
    uint64_t bytes() const;
    uint64_t errors() const;
    MpegTsOutput(const MpegTsOutput&) = delete;
    MpegTsOutput& operator=(const MpegTsOutput&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class StreamOutput {
public:
    // Optional callback is only used by test/ executables for fault injection.
    StreamOutput(uint16_t port, int bitrate, int timeout_ms,
                 std::function<void()> before_encode = {});
    ~StreamOutput();
    bool submit(Renderer&, const Stamp&);
    void stop();
    nlohmann::json stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yolo::live

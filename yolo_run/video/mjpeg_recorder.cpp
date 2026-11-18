#include "video/mjpeg_recorder.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

namespace yolo::multi {
namespace {
void check(int result, const char* operation) {
    if (result >= 0) return;
    char error[AV_ERROR_MAX_STRING_SIZE]{}; av_strerror(result, error, sizeof(error));
    throw std::runtime_error(std::string(operation) + ": " + error);
}
struct FormatDeleter {
    void operator()(AVFormatContext* context) const {
        if (!context) return;
        if (context->pb) avio_closep(&context->pb);
        avformat_free_context(context);
    }
};
}
class MjpegRecorder::Impl {
public:
    Impl(const std::filesystem::path& path, double fps, cv::Size dimensions) : size(dimensions) {
        if (!std::isfinite(fps) || fps <= 0 || size.width < 2 || size.height < 2)
            throw std::runtime_error("invalid MJPG recording parameters");
        AVFormatContext* raw = nullptr;
        const int rc = avformat_alloc_output_context2(&raw, nullptr, "avi", path.c_str());
        format.reset(raw); check(rc, "AVI allocation");
        if (!format) throw std::runtime_error("AVI allocation returned null");
        stream = avformat_new_stream(format.get(), nullptr);
        if (!stream) throw std::runtime_error("AVI stream allocation failed");
        const auto rate = av_d2q(fps, 1000000);
        tick = av_inv_q(rate); stream->time_base = tick; stream->avg_frame_rate = rate;
        stream->r_frame_rate = rate;
        auto& params = *stream->codecpar;
        params.codec_type = AVMEDIA_TYPE_VIDEO; params.codec_id = AV_CODEC_ID_MJPEG;
        params.width = size.width; params.height = size.height;
        params.format = AV_PIX_FMT_YUVJ420P; params.color_range = AVCOL_RANGE_JPEG;
        check(avio_open(&format->pb, path.c_str(), AVIO_FLAG_WRITE), "AVI open");
        check(avformat_write_header(format.get(), nullptr), "AVI header");
        encoded.reserve(size_t(size.width) * size.height * 3 + 1024);
    }
    void write(const cv::Mat& bgr, int64_t index) {
        if (closed || bgr.type() != CV_8UC3 || bgr.size() != size || index <= last_index)
            throw std::runtime_error("invalid AVI frame or output sequence");
        if (!cv::imencode(".jpg", bgr, encoded, {cv::IMWRITE_JPEG_QUALITY, 85}))
            throw std::runtime_error("JPEG encode failed");
        if (encoded.size() > size_t(std::numeric_limits<int>::max())) throw std::runtime_error("JPEG too large");
        AVPacket packet{};
        packet.data = encoded.data(); packet.size = static_cast<int>(encoded.size());
        packet.stream_index = stream->index; packet.flags = AV_PKT_FLAG_KEY;
        packet.pts = packet.dts = av_rescale_q(index, tick, stream->time_base);
        packet.duration = av_rescale_q(1, tick, stream->time_base);
        // Single-stream av_write_frame borrows packet storage for this synchronous call.
        // Encoded bytes are CPU-owned and can be reused once the write returns.
        check(av_write_frame(format.get(), &packet), "AVI frame write");
        avio_flush(format->pb); check(format->pb->error, "AVI flush");
        last_index = index;
    }
    void finish() {
        if (closed) return;
        closed = true;
        const int trailer = av_write_trailer(format.get());
        avio_flush(format->pb);
        const int io_error = format->pb->error;
        const int close_error = avio_closep(&format->pb);
        check(trailer, "AVI trailer"); check(io_error, "AVI final flush"); check(close_error, "AVI close");
    }
    std::unique_ptr<AVFormatContext, FormatDeleter> format;
    AVStream* stream = nullptr;
    AVRational tick{};
    cv::Size size;
    std::vector<unsigned char> encoded;
    int64_t last_index = -1;
    bool closed = false;
};
MjpegRecorder::MjpegRecorder(const std::filesystem::path& path, double fps, cv::Size size)
    : impl_(std::make_unique<Impl>(path, fps, size)) {}
MjpegRecorder::~MjpegRecorder() {
    try { impl_->finish(); }
    catch (const std::exception& e) { std::cerr << "MJPG cleanup: " << e.what() << '\n'; }
}
void MjpegRecorder::write(const cv::Mat& bgr, int64_t index) { impl_->write(bgr, index); }
void MjpegRecorder::finish() { impl_->finish(); }
} // namespace yolo::multi

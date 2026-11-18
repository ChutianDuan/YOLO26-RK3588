#include "camera/live_media.h"
#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <set>
#include <thread>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <opencv2/imgcodecs.hpp>

namespace {
bool reject_request = false, reject_queue = false, corrupt_next = false;
std::set<int> queued_fds;
std::array<int, 4> queue_slots{{-1,-1,-1,-1}};
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
size_t fdCount() {
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"), std::filesystem::directory_iterator{});
}
std::vector<unsigned char> pixels(const yolo::live::EncodedFrame& packet) {
    std::vector<unsigned char> result;
    packet.withCpuRead([&](const unsigned char* data, size_t bytes) { result.assign(data, data + bytes); });
    return result;
}
}
extern "C" int __real_ioctl(int, unsigned long, ...);
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
    va_list values; va_start(values, request);
    void* arg = va_arg(values, void*); va_end(values);
    if (request == VIDIOC_REQBUFS) {
        const auto& b = *static_cast<v4l2_requestbuffers*>(arg);
        require(b.memory == V4L2_MEMORY_DMABUF, "capture attempted a non-DMA fallback");
        queue_slots.fill(-1);
        if (reject_request) { reject_request = false; errno = EINVAL; return -1; }
    }
    if (request == VIDIOC_QUERYBUF || request == VIDIOC_EXPBUF)
        throw std::runtime_error("capture attempted MMAP/export allocation");
    if (request == VIDIOC_QBUF) {
        const auto& b = *static_cast<v4l2_buffer*>(arg);
        require(b.memory == V4L2_MEMORY_DMABUF && b.index < 4 && b.m.fd >= 0 && b.length >= 960256,
                "invalid direct-capture DMA queue descriptor");
        if (reject_queue) { reject_queue = false; errno = EIO; return -1; }
        for (int other : queue_slots) require(other != b.m.fd, "same DMA queued in two V4L2 slots");
        queue_slots[b.index] = b.m.fd;
        queued_fds.insert(b.m.fd);
    }
    const int rc = __real_ioctl(fd, request, arg);
    if (!rc && request == VIDIOC_DQBUF) {
        auto& b = *static_cast<v4l2_buffer*>(arg);
        require(b.index < 4, "invalid real V4L2 index");
        queue_slots[b.index] = -1;
        if (corrupt_next) { corrupt_next = false; b.flags |= V4L2_BUF_FLAG_ERROR; }
    }
    return rc;
}
int main() {
    try {
        using namespace yolo::live;
        const std::string device = "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0";
        // RGA/MPP lazily retain process-wide device FDs after the first import.
        { PacketPool warmup(960000); }
        const auto initial_fds = fdCount();
        for (bool* fault : {&reject_request, &reject_queue}) {
            *fault = true;
            bool failed = false;
            try { auto source = cameraSource(device); }
            catch (const std::runtime_error&) { failed = true; }
            require(failed && !*fault && fdCount() == initial_fds, "capture initialization failure leaked resources or fell back");
        }
        queued_fds.clear();
        std::atomic<bool> stop{false};
        auto source = cameraSource(device);
        require(source->info().at("capture_io") == "dmabuf" && source->info().at("packet_pool_buffers") == 8,
                "unexpected capture mode or pool size");
        corrupt_next = true;
        std::array<std::optional<EncodedFrame>, 4> held;
        std::array<std::vector<unsigned char>, 4> saved;
        for (unsigned i = 0; i < held.size(); ++i) {
            held[i] = source->next(stop);
            require(held[i].has_value(), "capture ended early");
            saved[i] = pixels(*held[i]);
        }
        // Four DMA buffers remain with V4L2 and four are retained by consumers.
        // next() must drop until a consumer releases one, without allocating/copying.
        auto release = std::async(std::launch::async, [&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            held[0].reset();
        });
        auto next = source->next(stop);
        release.get();
        require(next.has_value() && queued_fds.size() == 8, "capture did not rotate exactly eight DMA buffers");
        require(source->counters().at("packet_pool_dropped") > 0 && source->counters().at("corrupt_dropped") == 1 &&
                source->counters().at("compressed_copy_bytes") == 0, "drop/copy accounting mismatch");
        for (unsigned i = 1; i < held.size(); ++i)
            require(pixels(*held[i]) == saved[i], "V4L2 overwrote a retained compressed packet");
        // Requeueing a spare must not change the packet passed to MPP.
        MjpegDecoder decoder;
        auto frame = decoder.decode(*next);
        require(frame.has_value(), "MPP did not decode a directly captured DMA packet");
        cv::Mat reference = cv::imdecode(pixels(*next), cv::IMREAD_COLOR);
        Renderer renderer;
        const auto bgr = renderer.convert(*frame);
        cv::Mat delta; cv::absdiff(bgr, reference, delta);
        const auto average = cv::mean(delta);
        require((average[0]+average[1]+average[2])/3 <= 5, "direct DMA MPP/RGA differs from JPEG reference");
        renderer.finish();
        source.reset();
        for (unsigned i = 1; i < held.size(); ++i)
            require(pixels(*held[i]) == saved[i], "packet did not survive source destruction");
        std::cout << "PASS DMA capture/MPP, eight-buffer rotation, pool pressure, corrupt drop, init failure and retained packets\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

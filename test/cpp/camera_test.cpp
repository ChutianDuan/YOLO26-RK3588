#include "image/dma_buffer.h"
#include "image/image_processing.h"
#include "model/yolo_postprocess.h"
#include "camera/v4l2_camera.h"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace {
using namespace yolo;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid metadata/lease was accepted");
}
cv::Mat readRgb(const TensorInput& t) {
    auto buffer = std::static_pointer_cast<DmaBuffer>(t.dma.allocation);
    CpuAccess read(buffer, DMA_BUF_SYNC_READ);
    cv::Mat output = cv::Mat(t.dma.height, t.dma.width, CV_8UC3, t.dma.data,
                             size_t(t.dma.width_stride) * 3).clone();
    read.finish();
    return output;
}
}
int main(int argc, char** argv) {
    try {
        using namespace yolo;
        if (argc == 2 && std::string(argv[1]) == "--capture") {
            auto camera = std::make_unique<V4l2Camera>("/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0");
            auto frame = camera->next();
            require(frame.has_value() && frame->image.dma, "real camera did not provide DMA frame");
            auto retained = frame->image;
            frame.reset();
            rejects([&] { camera->next(); });
            retained = {};
            frame = camera->next();
            auto memory = frame->image.dma;
            camera.reset(); // STREAMOFF，但被租用的源 allocation 必须继续存活。
            CpuAccess read(memory, DMA_BUF_SYNC_READ);
            require(memory->fd >= 0 && memory->data != MAP_FAILED, "capture allocation died before lease");
            read.finish(); frame.reset();
            std::cout << "PASS real camera lease prevents QBUF reuse; source survives capture destruction\n";
            return 0;
        }
        AppConfig config;
        constexpr int stride = 832;
        auto source = std::make_shared<DmaBuffer>(stride * 600 * 2, "/dev/dma_heap/system-uncached");
        auto lease = std::make_shared<int>(1);
        YuyvFrameView frame{lease, source, static_cast<unsigned char*>(source->data), source->size, 800, 600, stride * 2};
        InputMemoryLayout layout{816, 610, 816U * 610U * 3U};
        ImagePreprocessor direct(config, layout);
        const std::array<std::array<unsigned char, 3>, 5> colors{{{16,128,128}, {235,128,128}, {82,90,240}, {145,54,34}, {41,240,110}}};
        int previous_fd = -1;
        for (unsigned iteration = 0; iteration < 12; ++iteration) {
            {
                CpuAccess write(source, DMA_BUF_SYNC_WRITE);
                std::memset(source->data, 0xCD, source->size);
                for (int y = 0; y < 600; ++y) for (int x = 0; x < 800; x += 2) {
                    const auto color = colors[(x / 160 + iteration) % colors.size()];
                    auto* p = static_cast<unsigned char*>(source->data) + y * stride * 2 + x * 2;
                    p[0] = color[0]; p[1] = color[1]; p[2] = color[0]; p[3] = color[2];
                }
                write.finish();
            }
            auto a = direct.prepareYuyv(frame);
            const auto rgb = readRgb(a);
            require(!cv::countNonZero(rgb.rowRange(0, 4).reshape(1) != 114) &&
                    !cv::countNonZero(rgb.rowRange(604, 608).reshape(1) != 114), "padding is not 114");
            cv::Mat cpu;
            {
                CpuAccess read(source, DMA_BUF_SYNC_READ);
                cv::cvtColor(cv::Mat(600, 800, CV_8UC2, source->data, stride * 2), cpu, cv::COLOR_YUV2RGB_YUY2);
                read.finish();
            }
            const double diff = cv::norm(rgb.rowRange(4, 604), cpu, cv::NORM_INF);
            require(diff <= 3, "YUYV order, RGB channels or BT.601 limited conversion differs from CPU");
            if (previous_fd >= 0) require(a.dma.fd == previous_fd, "RGB allocation not reused");
            previous_fd = a.dma.fd;
            require(a.dma.width_stride == 816 && a.dma.height_stride == 610 && a.dma.capacity == layout.min_capacity,
                    "RKNN layout/capacity not preserved");
            require(a.image_width == 800 && a.image_height == 600 && a.letterbox.pad_h == 4 &&
                    a.letterbox.scale_x == 1 && a.letterbox.scale_y == 1, "source-to-model transform");
            rejects([&] { direct.prepareYuyv(frame); });
            auto alias = a; a = {};
            rejects([&] { direct.prepareYuyv(frame); });
            alias = {};
        }
        {
            auto first = direct.prepareYuyv(frame);
            auto second = first;
            auto last = second;
            const auto allocation_only = first.dma.allocation;
            const int fd = first.dma.fd;
            std::weak_ptr<void> frame_lease = first.owner;
            first = {};
            second = {};
            rejects([&] { direct.prepareYuyv(frame); });
            last = {};
            require(frame_lease.expired(), "last TensorInput copy did not release frame occupancy");
            const auto reused = direct.prepareYuyv(frame);
            require(reused.dma.fd == fd && reused.dma.allocation == allocation_only,
                    "allocation-only reference blocked RGB reuse after last lease");
        }
        auto bad = frame; bad.owner.reset(); rejects([&] { direct.prepareYuyv(bad); });
        bad = frame; bad.capacity = 12; rejects([&] { direct.prepareYuyv(bad); });
        bad = frame; bad.bytes_per_line = 1599; rejects([&] { direct.prepareYuyv(bad); });
        bad = frame; bad.data += 2; rejects([&] { direct.prepareYuyv(bad); });
        bad = frame; bad.width = 799; rejects([&] { direct.prepareYuyv(bad); });
        bad = frame; bad.dma.reset();
        rejects([&] { direct.prepareYuyv(bad); });
        auto input = direct.prepareYuyv(frame);
        const float boxes[]{100,104,300,404};
        const float scores[]{0.9F,0,0,0,0,0,0,0,0,0};
        const auto result = decodeYolo26(boxes, scores, 1, input, config);
        require(result.size() == 1 && result[0].x1 == 100 && result[0].y1 == 100 && result[0].y2 == 400,
                "camera boxes not restored to 800x600 source coordinates");
        TensorInput survivor;
        { ImagePreprocessor temporary(config, layout); survivor = temporary.prepareYuyv(frame); }
        require(!readRgb(survivor).empty(), "tensor allocation did not survive producer");
        std::cout << "PASS YUYV colors/stride/padding, 12 changing DMA frames, pool reuse, leases, metadata, coordinates and ownership\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

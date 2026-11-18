#include "camera/live_media.h"
#include "image/mpp_dma_buffer.h"
#include "image/dma_buffer.h"
#include "tracking/byte_tracker.h"
#include <iostream>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <type_traits>
#include <utility>
#include <fcntl.h>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

void checkMppRegistrationOwnership() {
    using namespace yolo;
    using namespace yolo::live;
    static_assert(!std::is_copy_constructible_v<CpuAccess>);
    static_assert(!std::is_copy_constructible_v<MppDmaBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<MppDmaBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<MppDmaBuffer>);

    auto allocation = std::make_shared<DmaBuffer>(4096, "/dev/dma_heap/system-uncached");
    const int fd = allocation->fd;
    std::weak_ptr<DmaBuffer> observed = allocation;
    MppDmaBuffer imported(allocation);
    const auto handle = imported.get();
    allocation.reset();
    require(!observed.expired(), "MPP registration did not retain allocation");

    MppDmaBuffer moved(std::move(imported));
    require(!imported.get() && !imported.allocation() && moved.get() == handle,
            "MPP move constructor lost or duplicated registration ownership");

    auto previous = std::make_shared<DmaBuffer>(4096, "/dev/dma_heap/system-uncached");
    const int previous_fd = previous->fd;
    std::weak_ptr<DmaBuffer> old_allocation = previous;
    MppDmaBuffer assigned(previous);
    previous.reset();
    assigned = std::move(moved);
    require(!moved.get() && !moved.allocation() && assigned.get() == handle,
            "MPP move assignment lost or duplicated registration ownership");
    require(old_allocation.expired() && fcntl(previous_fd, F_GETFD) < 0 && errno == EBADF,
            "MPP move assignment retained replaced allocation");

    assigned.reset();
    require(!assigned.get() && !assigned.allocation() && observed.expired(),
            "MPP reset did not release registration/allocation");
    require(fcntl(fd, F_GETFD) < 0 && errno == EBADF, "MPP allocation FD survived reset");
}
}
int main(int argc, char** argv) {
    const auto directory = std::filesystem::temp_directory_path() / ("live_media_" + std::to_string(getpid()));
    try {
        using namespace yolo;
        using namespace yolo::live;
        checkMppRegistrationOwnership();
        std::filesystem::create_directory(directory);
        cv::Mat image(600, 800, CV_8UC3);
        const cv::Scalar colors[]{{0,0,0}, {255,255,255}, {0,0,255}, {0,255,0}, {255,0,0}};
        for (int i = 0; i < 5; ++i) image(cv::Rect(i * 160, 0, 160, 600)).setTo(colors[i]);
        require(cv::imwrite((directory / "0.jpg").string(), image, {cv::IMWRITE_JPEG_QUALITY, 100}), "write JPEG");
        {
            std::ofstream out(directory / "manifest.jsonl");
            out << "{\"type\":\"mjpeg_fixture\",\"width\":800,\"height\":600}\n";
            out << "{\"file\":\"0.jpg\",\"sequence\":0,\"timestamp_ms\":0}\n";
        }
        std::atomic<bool> stop{false};
        auto source = replaySource(directory / "manifest.jsonl", false);
        auto packet = source->next(stop); require(packet.has_value(), "missing fixture packet");
        MjpegDecoder decoder;
        auto frame = decoder.decode(*packet);
        cv::Mat reference;
        packet->withCpuRead([&](const unsigned char* data, size_t bytes) {
            reference = cv::imdecode(cv::Mat(1, int(bytes), CV_8UC1, const_cast<unsigned char*>(data)), cv::IMREAD_COLOR);
        });
        require(frame.has_value() && frame->image.width_stride >= 800 && frame->image.height_stride >= 600, "MPP output layout");
        Renderer renderer;
        const auto bgr = renderer.convert(*frame);
        for (int i = 0; i < 5; ++i) {
            const cv::Rect roi(i * 160 + 20, 20, 120, 560);
            require(cv::norm(bgr(roi), reference(roi), cv::NORM_INF) <= 5, "MPP/RGA color or range mismatch");
        }
        renderer.finish();
        ImagePreprocessor preprocess(AppConfig{});
        {
            const auto tensor = preprocess.prepareNv12(frame->image);
            CpuAccess read(std::static_pointer_cast<DmaBuffer>(tensor.dma.allocation), DMA_BUF_SYNC_READ);
            const cv::Mat rgb(608, 800, CV_8UC3, tensor.dma.data, tensor.dma.width_stride * 3);
            require(!cv::countNonZero(rgb.rowRange(0,4).reshape(1) != 114) &&
                    !cv::countNonZero(rgb.rowRange(604,608).reshape(1) != 114), "NV12 letterbox padding");
            require(rgb.at<cv::Vec3b>(100, 400)[0] >= 250, "NV12 RGB channel order");
            read.finish();
        }
        std::vector<DecodedFrame> leases;
        for (int i = 0; i < 7; ++i) { auto f = decoder.decode(*packet); require(f.has_value(), "early surface exhaustion"); leases.push_back(*f); }
        require(!decoder.decode(*packet), "surface reused while leased");
        leases.erase(leases.begin());
        require(decoder.decode(*packet).has_value(), "released surface not reusable");
        auto retained = frame->image;
        frame.reset();
        require(retained.owner && retained.dma->fd >= 0, "consumer lease lost ownership");
        {
            auto producer = std::make_unique<MjpegDecoder>();
            auto survivor = producer->decode(*packet);
            producer.reset();
            require(!renderer.convert(*survivor).empty(), "decoded allocation did not survive producer");
            renderer.finish();
        }
        {
            auto dma = std::make_shared<DmaBuffer>(832U*624U*3U/2U, "/dev/dma_heap/system-uncached");
            {
                CpuAccess write(dma, DMA_BUF_SYNC_WRITE);
                std::memset(dma->data, 70, 832U*624U);
                std::memset(static_cast<unsigned char*>(dma->data)+832U*624U,128,832U*624U/2U);
                write.finish();
            }
            Nv12FrameView padded{dma,dma,800,600,832,624,true};
            ImagePreprocessor padded_processor(AppConfig{}, {816,610,816U*610U*3U});
            auto tensor = padded_processor.prepareNv12(padded);
            CpuAccess read(std::static_pointer_cast<DmaBuffer>(tensor.dma.allocation), DMA_BUF_SYNC_READ);
            const cv::Mat rgb(608,800,CV_8UC3,tensor.dma.data,816*3);
            require(cv::norm(rgb.rowRange(4,604),cv::Mat(600,800,CV_8UC3,cv::Scalar(70,70,70)),cv::NORM_INF)<=2,
                    "non-tight NV12 Y/UV or output stride is incorrect");
            read.finish();
        }
        if (argc == 2) {
            auto real_source = replaySource(argv[1], false);
            MjpegDecoder real_mpp;
            double maximum_mean = 0, maximum_p95 = 0;
            for (int i = 0; i < 8; ++i) {
                auto p = real_source->next(stop); require(p.has_value(), "need eight real JPEG fixture frames");
                auto hardware = real_mpp.decode(*p);
                cv::Mat software;
                p->withCpuRead([&](const unsigned char* data, size_t bytes) {
                    software = cv::imdecode(cv::Mat(1, int(bytes), CV_8UC1, const_cast<unsigned char*>(data)), cv::IMREAD_COLOR);
                });
                const auto pixels = renderer.convert(*hardware);
                cv::Mat difference; cv::absdiff(pixels,software,difference);
                const auto average = cv::mean(difference);
                maximum_mean = std::max(maximum_mean,(average[0]+average[1]+average[2])/3);
                std::array<uint64_t,256> bins{};
                for (int y = 0; y < difference.rows; ++y) {
                    const auto* row = difference.ptr<unsigned char>(y);
                    for (int x = 0; x < difference.cols*3; ++x) ++bins[row[x]];
                }
                uint64_t n = 0;
                for (size_t value = 0; value < bins.size(); ++value) {
                    n += bins[value]; if (n >= 800U*600U*3U*.95) { maximum_p95=std::max(maximum_p95,double(value)); break; }
                }
                renderer.finish();
            }
            require(maximum_mean <= 5 && maximum_p95 <= 12, "real camera JPEG MPP/RGA differs excessively from CPU decode");
            std::cerr << "real JPEG max frame MAE=" << maximum_mean << ", max P95=" << maximum_p95 << '\n';
        }
        std::cerr << "PASS MPP MJPEG NV12 " << retained.width_stride << 'x' << retained.height_stride
                  << ", full-range colors/RGB/padding, eight-slot exhaustion and lease reuse\n";
        std::filesystem::remove_all(directory); return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n'; std::filesystem::remove_all(directory); return 1;
    }
}

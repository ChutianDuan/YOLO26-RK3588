#include "camera/v4l2_camera.h"
#include "image/dma_buffer.h"
#include "../support/cpu_reference.h"
#include "model/yolo_engine.h"
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fcntl.h>
#include <unistd.h>

// All RGB comparisons and checksum reads live in the validation executable.
int main(int argc, char** argv) {
    try {
        using namespace yolo;
        std::string device = "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0", output;
        unsigned frames = 8;
        for (int i = 1; i < argc; ++i) {
            const std::string key(argv[i]);
            if (key == "--help") {
                std::cout << "camera_validation [--device DEVICE] [--frames N>=8] [--output NEW.json]\n";
                return 0;
            }
            if (++i == argc) throw std::runtime_error("missing argument");
            const std::string value(argv[i]);
            if (key == "--device") device = value;
            else if (key == "--output") output = value;
            else if (key == "--frames") {
                auto rc = std::from_chars(value.data(), value.data() + value.size(), frames);
                if (rc.ec != std::errc() || rc.ptr != value.data() + value.size() || frames < 8 || frames > 100000)
                    throw std::runtime_error("frames must be 8..100000");
            } else throw std::runtime_error("unknown argument: " + key);
        }
        if (!output.empty() && std::filesystem::exists(std::filesystem::symlink_status(output)))
            throw std::runtime_error("output must be new");
        auto config = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
        config.model_path = std::string(YOLO_SOURCE_DIR) + "/../yolo_model_rknn/yolo26s_800x608_int8.rknn";
        cv::setNumThreads(config.opencv_threads);
        YoloEngine engine(config);
        ImagePreprocessor direct(config, engine.inputMemoryLayout());
        V4l2Camera camera(device);
        uint64_t upload = 0, read_bytes = 0, diagnostic_bytes = 0, changes = 0, previous = 0;
        for (unsigned i = 0; i < frames; ++i) {
            auto frame = camera.next();
            if (!frame) throw std::runtime_error("camera ended before validation completed");
            auto x = direct.prepareYuyv(frame->image);
            cv::Mat bgr;
            {
                CpuAccess read(frame->image.dma, DMA_BUF_SYNC_READ);
                cv::cvtColor(cv::Mat(600,800,CV_8UC2,const_cast<unsigned char*>(frame->image.data),
                                    frame->image.bytes_per_line), bgr, cv::COLOR_YUV2BGR_YUY2);
                read.finish();
            }
            const auto copied = yolo::test::referenceRgb(bgr, config);
            auto a = std::static_pointer_cast<DmaBuffer>(x.dma.allocation);
            CpuAccess ra(a, DMA_BUF_SYNC_READ);
            cv::Mat rgb(608, 800, CV_8UC3, x.dma.data, size_t(x.dma.width_stride) * 3);
            if (cv::norm(rgb, copied, cv::NORM_INF) > 3 ||
                cv::countNonZero(rgb.rowRange(0, 4).reshape(1) != 114) ||
                cv::countNonZero(rgb.rowRange(604, 608).reshape(1) != 114))
                throw std::runtime_error("RGA/reference pixels or letterbox differ");
            uint64_t hash = 14695981039346656037ULL;
            for (int row = 4; row < 604; row += 8)
                for (int column = 0; column < 2400; column += 7) {
                    hash = (hash ^ rgb.ptr<unsigned char>(row)[column]) * 1099511628211ULL;
                    ++diagnostic_bytes;
                }
            if (i && hash != previous) ++changes;
            previous = hash;
            ra.finish();
            read_bytes += 800U * 608U * 3U;
        }
        camera.stop();
        nlohmann::json result{{"type", "camera_validation"}, {"status", "passed"}, {"frames", frames},
            {"pixels_within_tolerance", true}, {"max_channel_error", 3}, {"padding_114", true}, {"validation_upload_bytes", upload},
            {"validation_rgb_read_bytes", read_bytes}, {"diagnostic_read_bytes", diagnostic_bytes},
            {"checksum_samples", frames}, {"checksum_changes", changes}};
        // A static scene may have zero checksum changes; this is an observation, not a correctness failure.
        if (!output.empty()) {
            const int fd = open(output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
            if (fd < 0) throw std::runtime_error("cannot create validation output");
            FILE* file = fdopen(fd, "w");
            if (!file) { close(fd); throw std::runtime_error("fdopen failed"); }
            const std::string text = result.dump(2) + "\n";
            const bool ok = fwrite(text.data(), 1, text.size(), file) == text.size();
            const int closed = fclose(file);
            if (!ok || closed) throw std::runtime_error("validation output write failed");
        }
        std::cerr << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

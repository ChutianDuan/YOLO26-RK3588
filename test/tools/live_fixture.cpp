#include "fixture_recorder.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
namespace { std::atomic<bool> stopped{false}; void stop(int) { stopped.store(true); } }
int main(int argc, char** argv) {
    try {
        using namespace yolo::live;
        if (argc < 4) throw std::runtime_error("Usage: live_fixture capture NEW_DIR N [DEVICE] | video INPUT NEW_DIR [N]");
        signal(SIGINT, stop); signal(SIGTERM, stop);
        if (std::string(argv[1]) == "capture") {
            auto source = cameraSource(argc > 4 ? argv[4] : "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0");
            recordFixture(*source, argv[2], std::stoull(argv[3]), stopped);
            std::cerr << source->counters().dump(2) << '\n';
        } else if (std::string(argv[1]) == "video") {
            const std::filesystem::path directory(argv[3]);
            if (!std::filesystem::create_directory(directory)) throw std::runtime_error("fixture directory must be new");
            cv::VideoCapture capture(argv[2]);
            if (!capture.isOpened()) throw std::runtime_error("cannot open fixture video");
            const double fps = capture.get(cv::CAP_PROP_FPS);
            if (!std::isfinite(fps) || fps <= 0) throw std::runtime_error("invalid video FPS");
            std::ofstream manifest(directory / "manifest.jsonl");
            manifest << nlohmann::json({{"type", "mjpeg_fixture"}, {"width", 800}, {"height", 600},
                {"negotiated_fps", fps}, {"max_packet_bytes", 960000}, {"input", argv[2]},
                {"resize", "stretch_800x600"}}).dump() << '\n';
            const uint64_t limit = argc > 4 ? std::stoull(argv[4]) : 0;
            cv::Mat frame, resized;
            for (uint64_t i = 0; !stopped.load() && (!limit || i < limit) && capture.read(frame); ++i) {
                cv::resize(frame, resized, {800, 600});
                const auto name = std::to_string(i) + ".jpg";
                if (!cv::imwrite((directory / name).string(), resized, {cv::IMWRITE_JPEG_QUALITY, 90}))
                    throw std::runtime_error("JPEG write failed");
                manifest << nlohmann::json({{"file", name}, {"sequence", i}, {"timestamp_ms", i * 1000.0 / fps}}).dump() << '\n';
                if (!manifest) throw std::runtime_error("manifest write failed");
            }
        } else throw std::runtime_error("unknown fixture mode");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

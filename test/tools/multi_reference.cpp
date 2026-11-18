#include "config/app_config.h"
#include "image/image_processing.h"
#include "model/yolo_engine.h"
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>
namespace {
std::map<std::string, uint64_t> dmaInodes() {
    std::map<std::string,uint64_t> result;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd")) {
        std::error_code ec; auto target = std::filesystem::read_symlink(e.path(), ec).string();
        if (ec || target.find("dmabuf") == std::string::npos) continue;
        struct stat st{};
        if (::stat(e.path().c_str(), &st) == 0) result[e.path().filename().string()] = st.st_ino;
    }
    return result;
}
}
int main(int argc, char** argv) {
    try {
        const int report_fd = dup(STDOUT_FILENO);
        if (report_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot route SDK diagnostics");
        if (argc < 3) throw std::runtime_error("multi_reference CONFIG VIDEO...");
        auto config = yolo::loadAppConfig(argv[1]); cv::setNumThreads(2);
        std::vector<cv::Mat> images;
        size_t capacity = 0;
        for (int i = 2; i < argc; ++i) {
            cv::VideoCapture cap(argv[i], cv::CAP_FFMPEG, {cv::CAP_PROP_N_THREADS,1}); cv::Mat image;
            if (!cap.read(image)) throw std::runtime_error("reference video unreadable");
            capacity = std::max(capacity, size_t((image.cols + 15) & ~15) * image.rows * 3);
            images.push_back(image);
        }
        nlohmann::json report = nlohmann::json::array();
        std::vector<std::vector<yolo::Detection>> baseline;
        for (int mask : {1,2,4}) {
            yolo::YoloEngine engine(config, mask);
            yolo::ImagePreprocessor processor(config, engine.inputMemoryLayout()); processor.reserveBgrSourceCapacity(capacity);
            std::map<std::string,uint64_t> dma;
            std::shared_ptr<void> rgb_allocation;
            for (size_t round = 0; round < 3; ++round) for (size_t i = 0; i < images.size(); ++i) {
                auto tensor = processor.prepare(images[i]);
                if (!rgb_allocation) rgb_allocation = tensor.dma.allocation;
                if (rgb_allocation != tensor.dma.allocation) throw std::runtime_error("RGB buffer reallocated");
                auto result = engine.infer(tensor);
                const auto current = dmaInodes();
                if (current.empty()) throw std::runtime_error("DMA inode inspection unavailable");
                if (dma.empty()) dma = current;
                else if (dma != current) throw std::runtime_error("DMA allocations changed while alternating source dimensions");
                if (mask == 1 && round == 0) {
                    baseline.push_back(result.detections);
                    nlohmann::json detections = nlohmann::json::array();
                    for (const auto& d : result.detections) detections.push_back({{"class_id",d.class_id},
                        {"score",d.score},{"bbox",{d.x1,d.y1,d.x2,d.y2}}});
                    report.push_back({{"channel_id",i},{"width",images[i].cols},{"height",images[i].rows},{"detections",detections}});
                }
                const auto& expected = baseline.at(i);
                if (expected.size() != result.detections.size()) throw std::runtime_error("candidate count differs across cores");
                for (size_t n = 0; n < expected.size(); ++n) {
                    const auto& a = expected[n]; const auto& b = result.detections[n];
                    if (a.class_id != b.class_id || std::abs(a.score-b.score) > .01F ||
                        std::max({std::abs(a.x1-b.x1), std::abs(a.y1-b.y1), std::abs(a.x2-b.x2), std::abs(a.y2-b.y2)}) > 1)
                        throw std::runtime_error("cross-core reference mismatch");
                }
            }
        }
        dprintf(report_fd, "%s\n", report.dump().c_str()); close(report_fd); return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

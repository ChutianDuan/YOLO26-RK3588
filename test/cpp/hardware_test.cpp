#include <cmath>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <rknn_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "image/image_processing.h"
#include "../support/cpu_reference.h"
#include "model/yolo_engine.h"
#include "video/optical_flow_tracker.h"

using namespace yolo;
namespace {
size_t imports = 0, releases = 0, waits = 0, peak_registrations = 0;
bool inject_run_error = false, synthetic_completion = false, wait_held_frame = false;
std::weak_ptr<void> observed_frame;
}
// Link-time hooks count real registrations and simulate an error after a real,
// completed run. No device timeout or driver fault is deliberately induced.
extern "C" {
rknn_tensor_mem* __real_rknn_create_mem_from_fd(rknn_context, int32_t, void*, uint32_t, int32_t);
int __real_rknn_destroy_mem(rknn_context, rknn_tensor_mem*);
int __real_rknn_run(rknn_context, rknn_run_extend*);
int __real_rknn_wait(rknn_context, rknn_run_extend*);
rknn_tensor_mem* __wrap_rknn_create_mem_from_fd(rknn_context ctx, int32_t fd, void* data, uint32_t size, int32_t offset) {
    auto* mem = __real_rknn_create_mem_from_fd(ctx, fd, data, size, offset);
    if (mem) { ++imports; peak_registrations = std::max(peak_registrations, imports - releases); }
    return mem;
}
int __wrap_rknn_destroy_mem(rknn_context ctx, rknn_tensor_mem* mem) {
    const int rc = __real_rknn_destroy_mem(ctx, mem);
    if (rc == RKNN_SUCC) ++releases;
    return rc;
}
int __wrap_rknn_run(rknn_context ctx, rknn_run_extend* run) {
    const int rc = __real_rknn_run(ctx, run);
    if (rc == RKNN_SUCC && inject_run_error) {
        inject_run_error = false;
        synthetic_completion = true;
        return RKNN_ERR_TIMEOUT;
    }
    return rc;
}
int __wrap_rknn_wait(rknn_context ctx, rknn_run_extend* run) {
    ++waits;
    wait_held_frame = !observed_frame.expired();
    if (synthetic_completion) { synthetic_completion = false; return RKNN_SUCC; }
    return __real_rknn_wait(ctx, run);
}
}
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}
void readSync(int fd, uint64_t flags) {
    dma_buf_sync sync{}; sync.flags = flags | DMA_BUF_SYNC_READ;
    int rc;
    do { rc = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync); } while (rc && errno == EINTR);
    if (rc) throw std::runtime_error("test DMA READ sync failed");
}
cv::Mat snapshotRgb(const TensorInput& input) {
    const auto& b = input.dma;
    readSync(b.fd, DMA_BUF_SYNC_START);
    cv::Mat copy;
    try { copy = cv::Mat(b.height, b.width, CV_8UC3, b.data, size_t(b.width_stride) * 3).clone(); }
    catch (...) { readSync(b.fd, DMA_BUF_SYNC_END); throw; }
    readSync(b.fd, DMA_BUF_SYNC_END);
    return copy; // CPU access has ended before this image can be submitted to RKNN.
}
size_t fdCount() {
    return size_t(std::distance(std::filesystem::directory_iterator("/proc/self/fd"), std::filesystem::directory_iterator{}));
}
void equivalent(const InferResult& a, const InferResult& b) {
    require(a.detections.size() == b.detections.size(), "same RGB: DMA/CPU detection count mismatch");
    for (size_t i = 0; i < a.detections.size(); ++i) {
        const auto& x = a.detections[i]; const auto& y = b.detections[i];
        if (x.class_id != y.class_id || std::abs(x.score-y.score) > 1e-6F ||
            std::abs(x.x1-y.x1) > 1e-3F || std::abs(x.y1-y.y1) > 1e-3F ||
            std::abs(x.x2-y.x2) > 1e-3F || std::abs(x.y2-y.y2) > 1e-3F)
            std::cerr << "equivalence mismatch frame=" << a.frame_index << " index=" << i << " class=" << x.class_id << "/" << y.class_id
                      << " scores=" << x.score << "/" << y.score << " x1=" << x.x1 << "/" << y.x1 << '\n';
        require(x.class_id == y.class_id && std::abs(x.score-y.score) <= 1e-6F &&
                std::abs(x.x1-y.x1) <= 1e-3F && std::abs(x.y1-y.y1) <= 1e-3F &&
                std::abs(x.x2-y.x2) <= 1e-3F && std::abs(x.y2-y.y2) <= 1e-3F,
                "same RGB: DMA/CPU class/order/score/coordinates mismatch");
    }
}
double iou(const Detection& a, const Detection& b) {
    const double area = std::max(0.0F, std::min(a.x2,b.x2)-std::max(a.x1,b.x1)) * std::max(0.0F, std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    return area / ((a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-area+1e-9);
}
}
int main() {
    try {
        auto c = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
        cv::setNumThreads(1);
        ImagePreprocessor rga(c);
        cv::Mat storage(202, 402, CV_8UC3, cv::Scalar(10, 70, 200));
        const cv::Mat bgr = storage(cv::Rect(1,1,400,200));
        for (float scale : {1.0F, .75F, .5F}) {
            auto cfg = c; cfg.lk_preprocess = "rga"; cfg.lk_scale = scale;
            TrackingPreprocessor flow(cfg);
            auto a = flow.prepare(bgr);
            require(a.gray.type() == CV_8UC1 && a.gray.size() == cv::Size(cvRound(400*scale), cvRound(200*scale)), "LK RGA dimensions");
            cv::Mat expected(a.gray.size(), CV_8UC1, cv::Scalar(102)); // .114*10 + .587*70 + .299*200
            require(cv::norm(a.gray, expected, cv::NORM_INF) <= 2, "LK grayscale BGR/full-range mismatch");
            auto b = flow.prepare(cv::Mat(200,400,CV_8UC3,cv::Scalar(255,255,255)));
            require(cv::norm(b.gray, cv::Mat(b.gray.size(),CV_8UC1,cv::Scalar(255)),cv::NORM_INF) <= 1, "LK limited-range white");
            require(cv::norm(a.gray, expected, cv::NORM_INF) <= 2, "LK previous DMA frame overwritten");
            bool rejected = false;
            try { flow.prepare(bgr); } catch (...) { rejected = true; }
            require(rejected, "LK pool exceeded two leased frames");
            const auto address = a.gray.data;
            a = {};
            auto reused = flow.prepare(cv::Mat(200,400,CV_8UC3,cv::Scalar(0,0,0)));
            require(reused.gray.data == address && cv::countNonZero(reused.gray) == 0, "LK DMA reuse/full-range black");
        }
        {
            auto cfg = c; cfg.lk_scale = .5F;
            TrackingImage survivor;
            { TrackingPreprocessor temporary(cfg); survivor = temporary.prepare(bgr); }
            require(std::abs(cv::mean(survivor.gray)[0] - 102) <= 2, "LK DMA lease did not retain allocation");
            TrackingPreprocessor flow(cfg);
            cv::Mat texture(240, 320, CV_8UC1), color, shifted;
            cv::RNG rng(3588); rng.fill(texture, cv::RNG::UNIFORM, 0, 255);
            cv::cvtColor(texture, color, cv::COLOR_GRAY2BGR);
            cv::warpAffine(color, shifted, cv::Mat(cv::Matx23d(1,0,4,0,1,2)), color.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
            auto previous = flow.prepare(color), current = flow.prepare(shifted);
            std::vector<cv::Mat> old_pyramid, new_pyramid;
            const auto tracked = trackFlowFrame(previous.gray, current.gray, {{7,{2,.9F,60,50,200,190}}},
                color.size(), 1, old_pyramid, new_pyramid);
            require(tracked.weak.tracks.size() == 1 && tracked.weak.tracks[0].track_id == 7 &&
                std::abs(tracked.weak.tracks[0].detection.x1 - 64) < .8 && tracked.motion.valid &&
                std::abs(tracked.motion.dx - 4) < .8 && std::abs(tracked.motion.dy - 2) < .8, "DMA LK scaled translation/ID/motion");
        }
        {
            auto a = rga.prepare(bgr); const auto b = yolo::test::referenceRgb(bgr, c);
            require(a.dma.allocation != nullptr, "RGA must expose owned DMA input");
            const auto pixels = snapshotRgb(a);
            require(cv::norm(pixels, b, cv::NORM_INF) <= 1, "RGA RGB/letterbox uniform pixel mismatch");
            require(pixels.at<cv::Vec3b>(0,0) == cv::Vec3b(114,114,114), "padding value");
            require(pixels.at<cv::Vec3b>(304,400) == cv::Vec3b(200,70,10), "RGB channel order");
            bool rejected = false;
            try { rga.prepare(bgr); } catch (...) { rejected = true; }
            require(rejected, "leased DMA buffer overwritten");
        }
        {
            auto changed = rga.prepare(cv::Mat(400,200,CV_8UC3,cv::Scalar(0,0,0)));
            const auto pixels = snapshotRgb(changed);
            require(pixels.at<cv::Vec3b>(304,20) == cv::Vec3b(114,114,114), "letterbox padding not reset after geometry change");
            require(pixels.at<cv::Vec3b>(304,400) == cv::Vec3b(0,0,0), "letterbox inner region not overwritten");
        }
        {
            ImagePreprocessor padded(c, {816, 610, 0});
            auto tensor = padded.prepare(bgr);
            const auto expected = yolo::test::referenceRgb(bgr, c);
            require(tensor.dma.capacity == 816U * 610U * 3U &&
                    cv::norm(snapshotRgb(tensor), expected, cv::NORM_INF) <= 1,
                    "RGA destination ignored nonpacked width/height strides");
        }
        const auto bus = cv::imread("/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg");
        require(!bus.empty(), "missing bus.jpg");
        std::vector<std::vector<Detection>> detections;
        for (const std::string precision : {"fp16", "int8"}) {
            c.model_path = (std::filesystem::path(YOLO_SOURCE_DIR).parent_path() / "yolo_model_rknn" / ("yolo26s_800x608_" + precision + ".rknn")).string();
            YoloEngine engine(c);
            yolo::test::CpuReferenceEngine cpu_reference(c);
            std::vector<Detection> reference;
            for (const std::string method : {"opencv", "rga"}) {
                ImagePreprocessor preprocessor(c, engine.inputMemoryLayout());
                auto tensor = preprocessor.prepare(bus);
                const auto result = method == "opencv" ?
                    cpu_reference.infer(yolo::test::referenceRgb(bus, c), tensor, 0) : engine.infer(tensor, 0);
                const auto repeat = method == "opencv" ?
                    cpu_reference.infer(yolo::test::referenceRgb(bus, c), tensor, 0) : engine.infer(tensor, 0);
                require(!result.detections.empty(), "bus inference returned no detections");
                require(result.detections.size() == repeat.detections.size(), "nondeterministic repeated inference");
                for (size_t i = 0; i < result.detections.size(); ++i) {
                    const auto& d = result.detections[i];
                    require(d.class_id >= 0 && d.class_id < 10 && d.score >= .25 && d.score <= 1 &&
                        d.x1 >= 0 && d.y1 >= 0 && d.x2 <= bus.cols && d.y2 <= bus.rows, "invalid decoded detection");
                    if (std::abs(d.score - repeat.detections[i].score) >= 1e-6)
                        std::cerr << "repeat mismatch index=" << i << " class=" << d.class_id << "/" << repeat.detections[i].class_id
                                  << " score=" << d.score << "/" << repeat.detections[i].score << '\n';
                    require(std::abs(d.score - repeat.detections[i].score) < 1e-6, "repeat score mismatch");
                }
                std::cerr << precision << "/" << method << ": " << result.detections.size() << " boxes, infer=" << result.infer_ms << "ms\n";
                if (method == "opencv") reference = result.detections;
                else {
                    size_t matched = 0;
                    for (const auto& a : reference) for (const auto& b : result.detections)
                        if (a.class_id == b.class_id && iou(a,b) > .7) { ++matched; break; }
                    require(matched > 0, "no matching RGA/OpenCV detections");
                    std::cerr << "same-class IoU>0.7 matches " << matched << "/" << reference.size() << '\n';
                }
                detections.push_back(result.detections);
            }
            const size_t imports_before = imports, releases_before = releases;
            {
                ImagePreprocessor producer(c, engine.inputMemoryLayout());
                ImagePreprocessor reference_producer(c, engine.inputMemoryLayout());
                const auto layout = engine.inputMemoryLayout();
                require(layout.width_stride == 800 && layout.height_stride == 608 &&
                        layout.min_capacity == (precision == "fp16" ? 2918400U : 1459200U), "unexpected installed model memory layout");
                cv::Mat flipped;
                cv::flip(bus, flipped, 1);
                size_t steady_fds = 0;
                int previous_fd = -1;
                double cpu_ms = 0, dma_ms = 0;
                for (int i = 0; i < 32; ++i) {
                    const auto& frame = i % 3 == 0 ? bus : i % 3 == 1 ? flipped : bgr;
                    auto tensor = producer.prepare(frame);
                    auto reference_tensor = reference_producer.prepare(frame);
                    require(tensor.dma.capacity >= layout.min_capacity, "producer ignored RKNN capacity");
                    if (i) require(tensor.dma.fd == previous_fd, "destination DMA allocation was not reused");
                    previous_fd = tensor.dma.fd;
                    // Never CPU-map the direct buffer before either FD inference:
                    // a READ sync here would hide missing RGA-to-RKNN invalidation.
                    const auto cpu_input = snapshotRgb(reference_tensor);
                    const auto baseline = cpu_reference.infer(cpu_input, reference_tensor, i);
                    const auto direct = engine.infer(tensor, i);
                    require(cv::norm(snapshotRgb(tensor), cpu_input, cv::NORM_INF) == 0,
                            "reference/direct RGA inputs differ or inference modified input pixels");
                    equivalent(baseline, direct);
                    cpu_ms += baseline.infer_ms; dma_ms += direct.infer_ms;
                    if (i == 0) steady_fds = fdCount();
                    else require(fdCount() == steady_fds, "FD growth during DMA reuse");
                }
                require(imports == imports_before + 1 && releases == releases_before + 1,
                        "registration cache did not replace once and reuse 32 times");
                auto tensor = producer.prepare(bus);
                const auto registered = imports;
                auto bad = tensor; bad.dma.fd = -1;
                rejects([&] { engine.infer(bad); }, "invalid FD accepted");
                const int closed_fd = dup(tensor.dma.fd);
                require(closed_fd >= 0, "dup failed"); close(closed_fd);
                bad = tensor; bad.dma.fd = closed_fd;
                rejects([&] { engine.infer(bad); }, "closed FD accepted");
                bad = tensor; bad.dma.capacity = layout.min_capacity - 1;
                rejects([&] { engine.infer(bad); }, "undersized DMA accepted");
                bad = tensor; bad.dma.capacity += 1024 * 1024;
                rejects([&] { engine.infer(bad); }, "capacity beyond FD allocation accepted");
                bad = tensor; bad.dma.width_stride += 16;
                rejects([&] { engine.infer(bad); }, "wrong DMA width stride accepted");
                bad = tensor; bad.dma.height_stride += 1;
                rejects([&] { engine.infer(bad); }, "wrong DMA height stride accepted");
                bad = tensor; bad.dma.data = nullptr;
                rejects([&] { engine.infer(bad); }, "null DMA mapping accepted");
                bad = tensor; bad.owner.reset();
                rejects([&] { engine.infer(bad); }, "missing frame lease accepted");
                require(imports == registered, "invalid metadata reached RKNN import");
                equivalent(cpu_reference.infer(snapshotRgb(tensor), tensor), engine.infer(tensor));
                // A distinct allocation owner must re-register, even with the same
                // integer FD/address/layout. The new owner still retains the real allocation.
                auto reowned = tensor;
                reowned.dma.allocation = std::make_shared<std::shared_ptr<void>>(tensor.dma.allocation);
                engine.infer(reowned);
                require(imports == registered + 1, "cache keyed only by FD instead of allocation owner");
                const int alias_fd = fcntl(tensor.dma.fd, F_DUPFD_CLOEXEC, 0);
                require(alias_fd >= 0, "DMA FD duplication failed");
                auto alias = tensor;
                alias.dma.fd = alias_fd;
                alias.dma.allocation = std::shared_ptr<void>(new std::shared_ptr<void>(tensor.dma.allocation),
                    [alias_fd](void* p) { close(alias_fd); delete static_cast<std::shared_ptr<void>*>(p); });
                equivalent(cpu_reference.infer(snapshotRgb(tensor), tensor), engine.infer(alias));
                require(imports == registered + 2, "duplicated FD was not independently registered");
                std::cerr << precision << ": 32 same-RGB CPU-reference/FD comparisons PASS; mean CPU input="
                          << cpu_ms / 32 << "ms, FD input=" << dma_ms / 32 << "ms, stable FDs=" << steady_fds << '\n';
            }
            TensorInput survivor;
            {
                ImagePreprocessor temporary(c, engine.inputMemoryLayout());
                survivor = temporary.prepare(bus);
            }
            equivalent(cpu_reference.infer(snapshotRgb(survivor), survivor), engine.infer(survivor));
            std::weak_ptr<void> retained_allocation = survivor.dma.allocation;
            survivor = {};
            require(!retained_allocation.expired(), "registration did not retain allocation after producer/frame destruction");
        }
        require(imports == releases, "RKNN registrations leaked after engine destruction");
        {
            auto engine = std::make_unique<YoloEngine>(c);
            auto producer = std::make_unique<ImagePreprocessor>(c, engine->inputMemoryLayout());
            auto tensor = producer->prepare(bus);
            const int owned_fd = tensor.dma.fd;
            observed_frame = tensor.owner;
            std::weak_ptr<void> allocation = tensor.dma.allocation;
            inject_run_error = true;
            rejects([&] { engine->infer(tensor); }, "injected execution failure swallowed");
            tensor = {};
            rejects([&] { producer->prepare(bus); }, "failed execution released its frame before draining");
            rejects([&] { engine->infer({}); }, "failed engine accepted another inference");
            producer.reset();
            require(!allocation.expired(), "failed engine released allocation prematurely");
            engine.reset();
            require(waits == 1 && wait_held_frame && observed_frame.expired() && allocation.expired(),
                    "execution failure drain/lease/allocation cleanup order");
            require(fcntl(owned_fd, F_GETFD) < 0 && errno == EBADF, "DMA FD remained open after engine cleanup");
        }
        require(imports == releases, "error path leaked RKNN registrations");
        require(peak_registrations == 1, "more than one external registration was alive at once");
        std::cout << "PASS RGA DMA/cache/stride/RGB/letterbox, FP16/INT8 FD binding/equivalence/reuse/lifetime/validation/error cleanup; imports="
                  << imports << ", releases=" << releases << ", peak=" << peak_registrations << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

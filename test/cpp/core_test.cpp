#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "config/app_config.h"
#include "image/image_processing.h"
#include "io/result_writer.h"
#include "model/yolo_postprocess.h"
#include "tracking/byte_tracker.h"

using namespace yolo;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
void fixture(const std::string& path, int count, bool use_bus = false) {
    cv::Mat base(120, 160, CV_8UC3);
    cv::RNG rng(123);
    rng.fill(base, cv::RNG::UNIFORM, 30, 220);
    if (use_bus) {
        const auto bus = cv::imread("/opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg");
        require(!bus.empty(), "missing bus fixture");
        cv::resize(bus, base, cv::Size(320, 240));
    }
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('M','J','P','G'), 30, base.size());
    require(writer.isOpened(), "fixture VideoWriter failed");
    cv::Mat frame;
    for (int i = 0; i < count; ++i) {
        cv::warpAffine(base, frame, cv::Mat(cv::Matx23d(1, 0, use_bus ? 0 : i, 0, 1, 0)), base.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
        writer.write(frame);
    }
}
void configTest(const std::filesystem::path& temp) {
    auto c = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
    require(c.class_names[1] == "rider" && c.class_names[9] == "traffic sign", "label order");
    require(std::filesystem::path(c.model_path).is_absolute(), "relative config paths");
    c.preprocess = "opencv";
    rejects([&] { validateConfig(c); }, "removed CPU preprocessing accepted");
    c.preprocess = "rga"; c.lk_preprocess = "opencv";
    rejects([&] { validateConfig(c); }, "removed CPU LK preprocessing accepted");
    c.lk_preprocess = "rga";
    c.lk_scale = 0; rejects([&] { validateConfig(c); }, "invalid LK scale accepted");
    c.lk_scale = 1; c.lk_preprocess = "bad";
    rejects([&] { validateConfig(c); }, "invalid LK backend accepted");
    c.lk_preprocess = "rga";
    c.conf_threshold = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { validateConfig(c); }, "NaN threshold accepted");
    const auto bad = temp / "bad.yaml";
    std::ofstream(bad) << "high_model_threads: 4\n";
    rejects([&] { loadAppConfig(bad.string()); }, "removed config accepted");
    std::ofstream(bad) << "video_detect_fps: 4garbage\n";
    rejects([&] { loadAppConfig(bad.string()); }, "partial number accepted");
}
void layoutTest() {
    AppConfig c;
    rknn_tensor_attr input{};
    input.n_dims = 4; input.dims[0] = 1; input.dims[1] = 608; input.dims[2] = 800; input.dims[3] = 3;
    input.fmt = RKNN_TENSOR_NHWC; input.type = RKNN_TENSOR_FLOAT16; input.n_elems = 608 * 800 * 3;
    std::vector<rknn_tensor_attr> outputs(2);
    for (int i = 0; i < 2; ++i) {
        auto& a = outputs[i]; a.n_dims = 3; a.dims[0] = 1; a.dims[1] = i ? 10 : 4; a.dims[2] = 9975;
        a.n_elems = a.dims[1] * a.dims[2]; a.fmt = RKNN_TENSOR_UNDEFINED;
        a.type = RKNN_TENSOR_INT8; a.qnt_type = RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC; a.scale = .01F; a.zp = -128;
    }
    validateModelLayout(input, outputs, c);
    std::swap(outputs[0], outputs[1]); validateModelLayout(input, outputs, c);
    outputs[0].scale = 0;
    rejects([&] { validateModelLayout(input, outputs, c); }, "invalid quantization accepted");
    outputs[0].scale = .01F; outputs[0].dims[1] = 80;
    rejects([&] { validateModelLayout(input, outputs, c); }, "wrong classes accepted");
    input.fmt = RKNN_TENSOR_NCHW;
    rejects([&] { validateModelLayout(input, outputs, c); }, "wrong input layout accepted");
}
void decoderTest() {
    AppConfig c;
    TensorInput input;
    input.image_width = 400; input.image_height = 200;
    input.letterbox = resizeGeometry(400, 200, c).transform;
    const float boxes[] = {20, 22, 124, 126, 200, 202, 304, 306};
    std::vector<float> scores(20, 0);
    scores[2 * 2] = .9F; scores[2 * 2 + 1] = .8F;
    auto d = decodeYolo26(boxes, scores.data(), 2, input, c);
    require(d.size() == 2, "YOLO26 must retain overlapping one-to-one boxes without NMS");
    scores[3 * 2] = .85F;
    d = decodeYolo26(boxes, scores.data(), 2, input, c);
    require(d.size() == 2 && d[0].class_id == 2 && d[1].class_id == 3 && d[0].x1 == d[1].x1,
            "two-stage TopK must retain multiple classes on the same anchor");
    scores[3 * 2] = 0;
    d = decodeYolo26(boxes, scores.data(), 2, input, c);
    require(d[0].class_id == 2 && std::abs(d[0].x1 - 10) < .001 && std::abs(d[0].y1 - 10) < .001, "letterbox inverse");
    c.class_conf_thresholds.assign(10, -1); c.class_conf_thresholds[2] = .85F;
    require(decodeYolo26(boxes, scores.data(), 2, input, c).size() == 1, "per-class threshold");
    scores[4] = std::numeric_limits<float>::quiet_NaN();
    require(decodeYolo26(boxes, scores.data(), 2, input, c).empty(), "NaN score accepted");
    c.class_conf_thresholds.clear();
    const size_t n = 350;
    std::vector<float> many_boxes(n * 4), many_scores(n * 10, 0);
    for (size_t i = 0; i < n; ++i) {
        many_boxes[i] = 10; many_boxes[n + i] = 110; many_boxes[2*n + i] = 40; many_boxes[3*n + i] = 150;
        many_scores[i] = .5F + float(i) / 1000;
    }
    d = decodeYolo26(many_boxes.data(), many_scores.data(), n, input, c);
    require(d.size() == 300 && d.front().score >= d.back().score, "top-k ordering");
}
void flowAndTrackerTest() {
    ByteTracker tracker;
    Detection d{2, .95F, 30, 25, 95, 95};
    auto tracks = tracker.update({d}, 0);
    require(tracks.size() == 1, "tracker bootstrap");
    const int id = tracks.front().track_id;
    cv::Mat previous(120, 160, CV_8UC1), current;
    cv::RNG rng(17); rng.fill(previous, cv::RNG::UNIFORM, 0, 255);
    cv::warpAffine(previous, current, cv::Mat(cv::Matx23d(1,0,1,0,1,0)), previous.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    const auto weak = weakTrackWithOpticalFlow(previous, current, tracks, previous.cols, previous.rows);
    require(weak.tracks.size() == 1 && std::abs(weak.quality.median_dx - 1) < .2F, "LK translation");
    std::vector<cv::Mat> old_pyramid, new_pyramid;
    auto large_tracks = tracks;
    for (auto& t : large_tracks) {
        t.detection.x1 *= 2; t.detection.x2 *= 2;
        t.detection.y1 *= 2; t.detection.y2 *= 2;
    }
    const auto scaled = trackFlowFrame(previous, current, large_tracks, {320, 240}, 1, old_pyramid, new_pyramid);
    require(scaled.weak.tracks.size() == 1 && scaled.weak.tracks[0].track_id == id &&
        std::abs(scaled.weak.tracks[0].detection.x1 - (large_tracks[0].detection.x1 + 2)) < .4 &&
        scaled.motion.valid && std::abs(scaled.motion.dx - 2) < .4, "scaled LK box/ID/motion restoration");
    cv::Mat diagonal;
    cv::warpAffine(previous, diagonal, cv::Mat(cv::Matx23d(1,0,1,0,1,1)), previous.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    const auto unequal = trackFlowFrame(previous, diagonal, large_tracks, {321,243}, 1, old_pyramid, new_pyramid);
    require(unequal.motion.valid && std::abs(unequal.motion.dx - 321.0F/160) < .4 &&
        std::abs(unequal.motion.dy - 243.0F/120) < .4, "actual independent X/Y scale restoration");
    tracks = tracker.updateTracked(weak.tracks, 1);
    require(tracks.size() == 1 && tracks[0].track_id == id, "LK lost ID");
    d.x1 += 1; d.x2 += 1;
    tracks = tracker.update({d}, 1);
    require(tracks.size() == 1 && tracks[0].track_id == id, "same-frame detection changed ID");
    DetectionPolicy policy(30, 4, true);
    WeakTrackQuality q;
    q.previous_track_count = q.tracked_track_count = 1; q.has_flow = true;
    q.tracked_track_ratio = q.valid_point_ratio = 1; q.min_score = .9F;
    for (int i = 0; i < 5; ++i) policy.observe(q, tracks, tracks);
    require(policy.current > policy.base, "stable cadence did not relax");
    q.tracked_track_count = 0;
    policy.observe(q, tracks, {});
    require(policy.current <= 2, "bad flow did not request urgent detection");
}
void historyTest() {
    MotionHistory h;
    for (int i = 0; i < 1000; ++i) { FrameMotion m; m.frame_index = i; m.dx = 1; m.valid = true; h.push(m); }
    require(h.size() == 64, "unbounded motion history");
    Detection d{2, .9F, 10, 20, 50, 70};
    auto moved = h.compensate({d}, 935, 999, 1000, 1000);
    require(moved && std::abs((*moved)[0].x1 - 74) < .01F, "64-frame compensation");
    require(!h.compensate({d}, 934, 999, 1000, 1000), "expired result accepted");
    require(!h.compensate({d}, 1000, 999, 1000, 1000), "future result accepted");
    FrameMotion m; m.frame_index = 1001; m.valid = true; h.push(m);
    require(!h.compensate({d}, 999, 1001, 1000, 1000), "missing motion accepted");
}
void workerTest() {
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    AsyncInferWorker worker([&](const cv::Mat& image, int64_t i) {
        entered.set_value(); gate.wait();
        require(image.at<cv::Vec3b>(0, 0)[0] == 17, "worker borrowed decoder memory");
        InferResult r; r.frame_index = i; return r;
    });
    cv::Mat image(30, 30, CV_8UC3, cv::Scalar::all(17));
    require(worker.submit(image, 7), "first submit");
    entered.get_future().wait();
    image.setTo(cv::Scalar::all(99));
    require(!worker.submit(image, 8), "second queued task accepted");
    release.set_value();
    auto result = worker.pop(true);
    require(result && !result->error && result->frame_index == 7 && !worker.pending(), "worker source routing");
    AsyncInferWorker broken([](const cv::Mat&, int64_t) -> InferResult { throw std::runtime_error("injected failure"); });
    require(broken.submit(image, 9), "error task submit");
    auto failed = broken.pop(true);
    require(failed && failed->error && !broken.pending(), "worker error not delivered");
}
void videoTest(const std::filesystem::path& temp) {
    fixture((temp / "in.avi").string(), 3);
    auto c = loadAppConfig(std::string(YOLO_SOURCE_DIR) + "/config.yaml");
    c.lk_preprocess = "rga";
    c.lk_scale = 1.0F;
    c.video_detect_fps = 15; c.video_stride_mode = "fixed";
    ResultWriter writer(c, temp / "out.jsonl", temp / "out.avi", true, 30);
    int64_t expected = 0;
    FrameResult last;
    const auto stats = inferVideoFile(temp / "in.avi", c, [](const cv::Mat&, int64_t index) {
        InferResult r; r.frame_index = index;
        r.detections.push_back({2, .9F, float(25 + index), 25, float(75 + index), 90}); return r;
    }, [&](const cv::Mat& frame, const FrameResult& result) {
        require(result.frame_index == expected++, "out-of-order frame");
        writer.frame(frame, result); last = result;
    });
    writer.finish(stats);
    require(stats.frames == 3 && stats.inference_count == 2 && stats.async_submissions == 1 && stats.async_corrections == 1,
            "video scheduling counters");
    require(last.is_detection_frame && last.corrected_from_frame_index == 2 && stats.pending_at_end == 0, "EOF did not drain/correct last frame");
    require(stats.flow.count == 3 && stats.tracker.count == 3 && stats.decode.count == 3,
            "flow, ByteTrack and successful decode must use per-frame timing counts");
    require(stats.flow_prepare.count == 3 && stats.flow_motion.count == 3 && stats.flow_compensation.count == 3 &&
        stats.flow_object[3].count == 3, "flow detail denominators");
    require(!last.tracks.empty() && last.tracks[0].track_id == 1, "video ID continuity");
    cv::VideoCapture read((temp / "out.avi").string());
    int count = 0; cv::Mat frame;
    while (read.read(frame)) ++count;
    require(count == 3, "annotation video frame count");
    std::ifstream jsonl(temp / "out.jsonl"); std::string line; count = 0;
    while (std::getline(jsonl, line)) { auto j = nlohmann::json::parse(line); ++count; }
    require(count == 4, "JSONL frame and summary count");
    c.video_model_async = false;
    rejects([&] { inferVideoFile(temp / "in.avi", c, [](const cv::Mat&, int64_t) -> InferResult { throw std::runtime_error("injected"); }, [](const cv::Mat&, const FrameResult&) {}); }, "video inference failure swallowed");
}
}
int main(int argc, char** argv) {
    cv::setNumThreads(1);
    const auto temp = std::filesystem::temp_directory_path() / ("yolo_core_" + std::to_string(getpid()));
    try {
        if (argc == 4 && std::string(argv[1]) == "--fixture") { fixture(argv[2], std::stoi(argv[3]), true); return 0; }
        if (argc == 4 && std::string(argv[1]) == "--verify-media") {
            const int expected = std::stoi(argv[3]);
            if (!cv::imread(argv[2]).empty()) { require(expected == 1, "unexpected image"); return 0; }
            cv::VideoCapture capture(argv[2]); cv::Mat frame;
            int count = 0;
            while (capture.read(frame) && count <= expected) ++count;
            require(count == expected, "media reread frame count mismatch");
            return 0;
        }
        std::filesystem::create_directory(temp);
        configTest(temp); layoutTest(); decoderTest(); flowAndTrackerTest(); historyTest(); workerTest(); videoTest(temp);
        std::filesystem::remove_all(temp);
        std::cout << "PASS config, layout, decoder, LK/ByteTrack, cadence, bounded history/worker, EOF/JSONL/AVI\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << " (fixtures " << temp << ")\n"; return 1; }
}

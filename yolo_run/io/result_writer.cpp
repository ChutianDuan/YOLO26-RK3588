#include "io/result_writer.h"
#include <cmath>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {
namespace {
using Json = nlohmann::json;
Json detectionJson(const Detection& d, const AppConfig& c) {
    return {{"class_id", d.class_id}, {"class_name", c.class_names.at(d.class_id)},
            {"score", d.score}, {"bbox", {d.x1, d.y1, d.x2, d.y2}}};
}
Json timingJson(const TimingAggregate& a) {
    return {{"count", a.count}, {"total_ms", a.total_ms}, {"mean_ms", a.mean()}, {"max_ms", a.max_ms}};
}
}
ResultWriter::ResultWriter(const AppConfig& c, const std::filesystem::path& jsonl,
    const std::filesystem::path& annotated, bool video, double fps)
    : config_(c), annotated_(annotated), video_(video), fps_(fps) {
    jsonl_.exceptions(std::ios::badbit | std::ios::failbit);
    jsonl_.open(jsonl);
    if (!annotated.empty()) {
        auto ext = annotated.extension().string();
        if (video && ext != ".avi") throw std::runtime_error("annotated video must use .avi (MJPG)");
        if (!video && ext != ".jpg" && ext != ".jpeg" && ext != ".png")
            throw std::runtime_error("annotated image must use .jpg, .jpeg or .png");
        if (video && (!std::isfinite(fps) || fps <= 0)) throw std::runtime_error("invalid annotation FPS");
    }
}
void ResultWriter::frame(const cv::Mat& image, const FrameResult& r) {
    // 到这里框和轨迹都已处于原图坐标；序列化/画框不再做 letterbox 逆变换。
    Json detections = Json::array(), tracks = Json::array();
    for (const auto& d : r.detections) detections.push_back(detectionJson(d, config_));
    for (const auto& t : r.tracks) {
        auto j = detectionJson(t.detection, config_);
        j["track_id"] = t.track_id;
        tracks.push_back(std::move(j));
    }
    Json value = {{"type", "frame"}, {"frame_index", r.frame_index}, {"timestamp_ms", r.timestamp_ms},
        {"width", r.width}, {"height", r.height}, {"is_detection_frame", r.is_detection_frame},
        {"tracks_source", r.tracks_source}, {"corrected_from_frame_index", r.corrected_from_frame_index},
        {"detections", std::move(detections)}, {"tracks", std::move(tracks)}};
    if (!annotated_.empty()) {
        // 可选标注只绘制到可复用副本，避免污染仍要参与 LK 或检测的原始 BGR 帧。
        image.copyTo(drawing_);
        auto draw = [&](const Detection& d, int id) {
            const cv::Scalar color(40 + (d.class_id * 67) % 180, 200, 60 + (d.class_id * 29) % 180);
            cv::rectangle(drawing_, cv::Point(cvRound(d.x1), cvRound(d.y1)), cv::Point(cvRound(d.x2), cvRound(d.y2)), color, 2);
            std::string label = config_.class_names.at(d.class_id) + " " + std::to_string(int(std::round(d.score * 100))) + "%";
            if (id >= 0) label += " #" + std::to_string(id);
            cv::putText(drawing_, label, cv::Point(std::max(0, cvRound(d.x1)), std::max(18, cvRound(d.y1) - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        };
        if (video_) {
            for (const auto& t : r.tracks) draw(t.detection, t.track_id);
            if (!video_writer_.isOpened()) {
                if (!video_writer_.open(annotated_.string(), cv::VideoWriter::fourcc('M','J','P','G'), fps_, drawing_.size()))
                    throw std::runtime_error("cannot create annotated video: " + annotated_.string());
            }
            video_writer_.write(drawing_);
        } else {
            for (const auto& d : r.detections) draw(d, -1);
            if (!cv::imwrite(annotated_.string(), drawing_)) throw std::runtime_error("cannot write annotated image");
        }
    }
    // 逐帧写出，不在内存中积累整段结果；frame_index 保留源视频时序。
    jsonl_ << value.dump() << '\n';
}
void ResultWriter::finish(const RunStats& s) {
    // 汇总中的 processing_fps 包含 LK 帧，不能当作 NPU 每秒推理次数。
    video_writer_.release();
    Json summary = {{"type", "summary"}, {"model_path", config_.model_path}, {"preprocess", config_.preprocess},
        {"lk_preprocess", config_.lk_preprocess}, {"lk_scale", config_.lk_scale},
        {"lk_width", s.flow_width}, {"lk_height", s.flow_height},
        {"frame_count", s.frames}, {"inference_count", s.inference_count}, {"source_fps", s.source_fps},
        {"video_model_async", video_ && config_.video_model_async && s.max_stride > 1},
        {"stride_mode", config_.video_stride_mode}, {"min_stride", s.min_stride}, {"max_stride", s.max_stride}, {"final_stride", s.final_stride},
        {"async_submissions", s.async_submissions}, {"async_corrections", s.async_corrections},
        {"discarded_results", s.discarded_results}, {"skipped_detections", s.skipped_detections}, {"forced_detections", s.forced_detections},
        {"max_pending", s.max_pending}, {"pending_at_end", s.pending_at_end}, {"history_peak", s.history_peak},
        {"elapsed_ms", s.elapsed_ms}, {"processing_fps", s.elapsed_ms > 0 ? s.frames * 1000.0 / s.elapsed_ms : 0},
        {"cpu_percent", s.cpu_percent}, {"peak_rss_mb", s.peak_rss_mb},
        {"timing", {{"decode", timingJson(s.decode)}, {"preprocess", timingJson(s.preprocess)},
                    {"infer", timingJson(s.infer)}, {"postprocess", timingJson(s.postprocess)},
                    {"flow", timingJson(s.flow)}, {"tracker", timingJson(s.tracker)}, {"queue_wait", timingJson(s.queue_wait)}}}};
    summary["flow_detail"] = {{"prepare", timingJson(s.flow_prepare)},
        {"frame_diff", timingJson(s.flow_object[0])}, {"features", timingJson(s.flow_object[1])},
        {"pyramid", timingJson(s.flow_object[2])}, {"forward", timingJson(s.flow_object[3])},
        {"backward", timingJson(s.flow_object[4])}, {"quality", timingJson(s.flow_object[5])},
        {"motion", timingJson(s.flow_motion)}, {"compensation", timingJson(s.flow_compensation)}};
    jsonl_ << summary.dump() << '\n';
    jsonl_.flush();
}
}  // namespace yolo

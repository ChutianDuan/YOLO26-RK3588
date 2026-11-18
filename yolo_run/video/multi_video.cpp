#include "video/multi_video.h"
#include "video/fair_scheduler.h"
#include "video/multi_stats.h"
#include "video/mjpeg_recorder.h"
#include "image/image_processing.h"
#include "model/yolo_engine.h"
#include "tracking/byte_tracker.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <thread>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rknn_api.h>

namespace yolo::multi {
namespace {
constexpr size_t pool_size = 6, event_capacity = 64, history_capacity = 64;
constexpr double recording_lag_limit_ms = 250;
struct Frame {
    std::shared_ptr<const cv::Mat> image;
    int64_t sequence = 0;
    double media_ms = 0, due_ms = 0, published_ms = 0;
};
using Scheduler = FairScheduler<Frame>;
struct Channel {
    size_t id;
    std::filesystem::path path, directory;
    cv::VideoCapture capture;
    std::unique_ptr<MjpegRecorder> recording;
    cv::Size size;
    double fps = 0, expected_frames = 0, period = 0;
    std::array<std::shared_ptr<cv::Mat>, pool_size> pool;
    std::atomic<bool> source_done{false}, output_done{false}, output_failed{false}, record_failed{false};
    std::atomic<int64_t> last_source{-1};
    std::atomic<uint64_t> grabbed{0}, decoded{0}, late_dropped{0}, pool_dropped{0}, inferred{0}, accepted{0};
    std::atomic<uint64_t> output_frames{0}, recorded{0}, repeated{0}, event_dropped{0}, stale_results{0}, output_skipped{0};
    std::atomic<uint64_t> rate_skipped{0}, prediction_frames{0}, correction_frames{0}, predicted_boxes{0}, lost_boxes{0};
    std::array<std::atomic<uint64_t>, 3> cores{};
    Distribution decode_ms, queue_ms, preprocess_ms, infer_ms, postprocess_ms, result_ms, output_ms, age_ms, record_ms;
    Distribution tracking_ms, detection_age_ms;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::optional<Frame> latest;
    std::optional<Frame> initial;
    ByteTracker tracker; // Serialized by the scheduler's one-in-flight rule AND mutex.
    std::deque<TrackerSnapshot> history;
    std::deque<Json> events;
    size_t event_peak = 0;
    int64_t watermark = -1;
    cv::Mat preview;
    std::string source_error, output_error, record_error;

    Channel(size_t number, const std::filesystem::path& file) : id(number), path(file) {
        if (!std::filesystem::is_regular_file(file)) throw std::runtime_error("input must be a local regular file: " + file.string());
        if (!capture.open(file.string(), cv::CAP_FFMPEG, {cv::CAP_PROP_N_THREADS, 1,
                cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 1000, cv::CAP_PROP_READ_TIMEOUT_MSEC, 1000}))
            throw std::runtime_error("cannot open video: " + file.string());
        fps = capture.get(cv::CAP_PROP_FPS); expected_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
        if (!std::isfinite(fps) || fps <= 0 || fps > 240) throw std::runtime_error("video FPS must be in (0,240]");
        period = 1000 / fps;
        pool[0] = std::make_shared<cv::Mat>();
        if (!capture.read(*pool[0]) || pool[0]->empty() || pool[0]->type() != CV_8UC3)
            throw std::runtime_error("video has no readable BGR frames: " + file.string());
        size = pool[0]->size();
        if (size.width < 2 || size.height < 2 || size.width > 8192 || size.height > 8192)
            throw std::runtime_error("unsupported decoded video dimensions");
        for (size_t i = 1; i < pool.size(); ++i) pool[i] = std::make_shared<cv::Mat>(size, CV_8UC3);
        preview = cv::Mat::zeros(360, 640, CV_8UC3);
    }
    std::shared_ptr<cv::Mat> acquire() {
        // Decoder alone examines this array; every external reference is a read lease.
        for (auto& slot : pool) if (slot.use_count() == 1) return slot;
        return {};
    }
    void event(Json row) {
        std::lock_guard<std::mutex> lock(mutex);
        if (output_failed || events.size() == event_capacity) {
            ++event_dropped;
            if (!output_failed) { output_error = "JSONL event queue full (64)"; output_failed = true; }
            changed.notify_all(); return;
        }
        events.push_back(std::move(row)); event_peak = std::max(event_peak, events.size()); changed.notify_all();
    }
    void recordFailure(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex);
        record_failed = true; if (record_error.empty()) record_error = error;
    }
};
Json detectionJson(const Detection& d, const AppConfig& config) {
    return {{"class_id", d.class_id}, {"class_name", config.class_names.at(d.class_id)},
        {"score", d.score}, {"bbox", {d.x1, d.y1, d.x2, d.y2}}};
}
void drawTracks(cv::Mat& canvas, const std::vector<TrackedDetection>& tracks, size_t channel, const AppConfig& config) {
    for (const auto& t : tracks) {
        const auto& d = t.detection;
        const cv::Scalar color(40 + (d.class_id * 67) % 180, 200, 60 + (d.class_id * 29) % 180);
        cv::rectangle(canvas, {cvRound(d.x1), cvRound(d.y1)}, {cvRound(d.x2), cvRound(d.y2)}, color, 2);
        const std::string label = config.class_names.at(d.class_id) + " " + std::to_string(channel) + ":" + std::to_string(t.track_id);
        cv::putText(canvas, label, {std::max(0, cvRound(d.x1)), std::max(18, cvRound(d.y1) - 4)},
                    cv::FONT_HERSHEY_SIMPLEX, .5, color, 1, cv::LINE_AA);
    }
}
Json channelStats(const Channel& c, const Scheduler::Stats& s, double seconds) {
    std::lock_guard<std::mutex> lock(c.mutex);
    return {{"channel_id", c.id}, {"input", c.path.string()}, {"width", c.size.width}, {"height", c.size.height},
        {"source_fps", c.fps}, {"advertised_frames", c.expected_frames}, {"grabbed", c.grabbed.load()},
        {"decoded", c.decoded.load()}, {"decode_late_dropped", c.late_dropped.load()}, {"pool_dropped", c.pool_dropped.load()},
        {"inference_count", c.inferred.load()}, {"accepted_results", c.accepted.load()},
        {"stale_results", c.stale_results.load()}, {"output_frames", c.output_frames.load()},
        {"recorded_frames", c.recorded.load()}, {"repeated_frames", c.repeated.load()},
        {"output_ticks_skipped", c.output_skipped.load()},
        {"detection_rate_skipped", c.rate_skipped.load()},
        {"tracking", {{"prediction_frames", c.prediction_frames.load()}, {"correction_frames", c.correction_frames.load()},
            {"predicted_boxes", c.predicted_boxes.load()}, {"lost_predictions", c.lost_boxes.load()}}},
        {"detection_fps", seconds > 0 ? c.inferred / seconds : 0},
        {"output_fps", seconds > 0 ? c.output_frames / seconds : 0},
        {"core_results", {c.cores[0].load(), c.cores[1].load(), c.cores[2].load()}},
        {"scheduler", {{"dispatched", s.dispatched}, {"replaced", s.replaced}, {"cancelled", s.cancelled},
            {"pending", s.pending}, {"busy", s.busy}, {"eof", s.eof}}},
        {"event_queue_peak", c.event_peak}, {"event_dropped", c.event_dropped.load()},
        {"source_done", c.source_done.load()}, {"source_error", c.source_error},
        {"output_error", c.output_error}, {"record_error", c.record_error},
        {"timing", {{"decode", c.decode_ms.json()}, {"queue_wait", c.queue_ms.json()},
            {"preprocess", c.preprocess_ms.json()}, {"infer", c.infer_ms.json()}, {"postprocess", c.postprocess_ms.json()},
            {"source_to_result", c.result_ms.json()}, {"output_latency", c.output_ms.json()},
            {"result_age", c.age_ms.json()}, {"record", c.record_ms.json()},
            {"tracking", c.tracking_ms.json()}, {"track_detection_age", c.detection_age_ms.json()}}}};
}
} // namespace

Json run(const AppConfig& config, const Options& options, const volatile sig_atomic_t* signal_stop, const Callbacks& callbacks) {
    validateConfig(config);
    if (options.inputs.empty() || options.inputs.size() > 4 || options.output_dir.empty()) throw std::runtime_error("requires 1..4 videos and new output directory");
    if (!std::isfinite(options.duration_seconds) || options.duration_seconds < 0) throw std::runtime_error("invalid duration");
    if (!std::isfinite(options.detect_fps) || options.detect_fps < 0 || options.detect_fps > 240)
        throw std::runtime_error("--detect-fps must be in [0,240]");
    if (!std::isfinite(options.track_max_age_ms) || options.track_max_age_ms < 1 || options.track_max_age_ms > 1000)
        throw std::runtime_error("--track-max-age-ms must be in [1,1000]");
    if (options.preview != "auto" && options.preview != "on" && options.preview != "off") throw std::runtime_error("invalid preview mode");
    const char* x_display = std::getenv("DISPLAY");
    const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
    const bool display = (x_display && *x_display) || (wayland_display && *wayland_display);
    if (options.preview == "on" && !display) throw std::runtime_error("preview needs a graphical session; use --preview off");
    const bool preview = options.preview == "on" || (options.preview == "auto" && display);
    if (std::filesystem::exists(options.output_dir)) throw std::runtime_error("output directory must be new: " + options.output_dir.string());
    cv::setNumThreads(config.opencv_threads);
    std::vector<std::unique_ptr<Channel>> channels;
    size_t upload_capacity = 0;
    for (size_t i = 0; i < options.inputs.size(); ++i) {
        channels.push_back(std::make_unique<Channel>(i, std::filesystem::absolute(options.inputs[i])));
        const auto size = channels.back()->size;
        if (options.record && (size.width % 2 || size.height % 2))
            throw std::runtime_error("MJPG recording requires even decoded dimensions; use --record off");
        upload_capacity = std::max(upload_capacity, size_t((size.width + 15) & ~15) * size.height * 3);
    }
    if (!options.output_dir.parent_path().empty()) std::filesystem::create_directories(options.output_dir.parent_path());
    if (!std::filesystem::create_directory(options.output_dir)) throw std::runtime_error("cannot exclusively create output directory");
    std::array<std::unique_ptr<YoloEngine>, 3> engines;
    std::array<std::unique_ptr<ImagePreprocessor>, 3> processors;
    constexpr std::array<int, 3> masks{RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
    for (size_t i = 0; i < engines.size(); ++i) {
        engines[i] = std::make_unique<YoloEngine>(config, masks[i]);
        processors[i] = std::make_unique<ImagePreprocessor>(config, engines[i]->inputMemoryLayout());
        processors[i]->reserveBgrSourceCapacity(upload_capacity);
    }
    for (auto& c : channels) {
        c->directory = options.output_dir / ("stream_" + std::to_string(c->id));
        std::filesystem::create_directory(c->directory);
        // Encoder construction belongs to startup, outside the shared playback epoch.
        if (options.record) {
            try { c->recording = std::make_unique<MjpegRecorder>(c->directory / "annotated.avi", c->fps, c->size); }
            catch (const std::exception& e) { c->recordFailure(e.what()); }
        }
    }
    Scheduler scheduler(channels.size());
    std::atomic<bool> stop{false};
    std::mutex wait_mutex, failure_mutex;
    std::condition_variable wake;
    std::string fatal_error, stop_reason = "eof";
    auto requestStop = [&] {
        stop = true; scheduler.cancel(); wake.notify_all();
        for (auto& c : channels) c->changed.notify_all();
    };
    auto fail = [&](const std::string& error) {
        { std::lock_guard<std::mutex> lock(failure_mutex); if (fatal_error.empty()) fatal_error = error; }
        requestStop();
    };
    auto waitUntil = [&](double deadline) {
        std::unique_lock<std::mutex> lock(wait_mutex);
        wake.wait_until(lock, Steady::time_point(std::chrono::duration_cast<Steady::duration>(
            std::chrono::duration<double, std::milli>(deadline))), [&] { return stop.load(); });
    };
    // One shared epoch, established only after all contexts and buffers are ready.
    const double start = nowMs(), cpu_start = cpuSeconds();
    for (auto& c : channels) { c->latest = Frame{c->pool[0], 0, 0, start, start}; c->initial = c->latest; }
    std::vector<std::thread> decoders, outputs, workers;
    auto join = [&] {
        for (auto& t : decoders) if (t.joinable()) t.join();
        for (auto& t : workers) if (t.joinable()) t.join();
        for (auto& c : channels) c->changed.notify_all();
        for (auto& t : outputs) if (t.joinable()) t.join();
    };
    std::ofstream resource_file(options.output_dir / "resources.jsonl");
    resource_file.exceptions(std::ios::badbit | std::ios::failbit);
    ResourceSampler sampler;
    Json metadata{{"type", "multi_video"}, {"schema_version", 1}, {"channels", channels.size()}, {"workers", 3},
        {"core_masks", masks}, {"model", config.model_path}, {"input_width", config.input_width}, {"input_height", config.input_height},
        {"preprocess", "rga"}, {"decode", "opencv_ffmpeg_cpu"}, {"decoder_threads_per_channel", 1},
        {"record_requested", options.record}, {"preview", preview}, {"epoch_monotonic_ms", start},
        {"recorder", "opencv_jpeg85_system_avformat_avi"},
        {"tracking", "bytetrack_native_kalman"}, {"detect_fps_limit", options.detect_fps},
        {"frame_pool_per_channel", pool_size}, {"pending_per_channel", 1}, {"inflight_per_channel", 1},
        {"event_capacity_per_channel", event_capacity}, {"snapshot_capacity_per_channel", history_capacity},
        {"prediction_horizon_ms", options.track_max_age_ms}, {"recording_lag_limit_ms", recording_lag_limit_ms},
        {"bgr_upload_reserved_bytes_per_worker", upload_capacity}, {"timing_policy", "source_index/source_fps; shared steady clock epoch"}};
    {
        std::ofstream file(options.output_dir / "metadata.json"); file.exceptions(std::ios::badbit | std::ios::failbit);
        file << metadata.dump(2) << '\n';
    }
    try {
        for (size_t w = 0; w < engines.size(); ++w) workers.emplace_back([&, w] {
            while (auto task = scheduler.take()) {
                auto& c = *channels[task->channel];
                try {
                    auto frame = std::move(task->value);
                    const double began = nowMs();
                    c.queue_ms.add(began - frame.published_ms);
                    auto tensor = processors[w]->prepare(*frame.image);
                    const double prepared = nowMs() - began;
                    frame.image.reset(); // RGA IM_SYNC has finished reading the CPU upload source.
                    if (callbacks.before_infer) callbacks.before_infer(w, c.id, frame.sequence);
                    auto result = engines[w]->infer(tensor, frame.sequence);
                    const double completed = nowMs();
                    ++c.inferred; ++c.cores[w];
                    c.preprocess_ms.add(prepared); c.infer_ms.add(result.infer_ms);
                    c.postprocess_ms.add(result.postprocess_ms); c.result_ms.add(completed - frame.due_ms);
                    bool accepted = false;
                    {
                        std::lock_guard<std::mutex> lock(c.mutex);
                        if (frame.sequence > c.watermark) {
                            c.history.push_back(c.tracker.updateAt(result.detections, frame.sequence, frame.media_ms));
                            if (c.history.size() > history_capacity) c.history.pop_front();
                            c.watermark = frame.sequence; ++c.accepted; accepted = true;
                        } else ++c.stale_results;
                    }
                    Json detections = Json::array();
                    for (const auto& d : result.detections) detections.push_back(detectionJson(d, config));
                    c.event({{"type", "detection"}, {"channel_id", c.id}, {"source_frame_index", frame.sequence},
                        {"media_timestamp_ms", frame.media_ms}, {"scheduled_monotonic_ms", frame.due_ms},
                        {"completed_monotonic_ms", completed}, {"worker", w}, {"core_mask", masks[w]},
                        {"width", c.size.width}, {"height", c.size.height}, {"accepted", accepted},
                        {"queue_wait_ms", began - frame.published_ms}, {"preprocess_ms", prepared},
                        {"infer_ms", result.infer_ms}, {"postprocess_ms", result.postprocess_ms},
                        {"source_to_result_ms", completed - frame.due_ms}, {"detections", std::move(detections)}});
                    // tensor's lease is destroyed before this worker can reuse its RGB target.
                } catch (const std::exception& e) { fail("worker " + std::to_string(w) + ": " + e.what()); }
                scheduler.complete(c.id); c.changed.notify_all();
            }
        });
        for (auto& owner : channels) {
            Channel& c = *owner;
            decoders.emplace_back([&, ptr = &c] {
                auto& channel = *ptr;
                int64_t index = 0;
                int64_t submitted = -1;
                double last_submit_ms = -1e30;
                try {
                    while (!stop) {
                        if (callbacks.before_decode) callbacks.before_decode(channel.id, index);
                        const double began = nowMs();
                        if (index && !channel.capture.grab()) {
                            if (std::isfinite(channel.expected_frames) && index + 1 < channel.expected_frames)
                                throw std::runtime_error("decode ended before advertised EOF");
                            break;
                        }
                        ++channel.grabbed; channel.last_source = index;
                        const double due = start + index * channel.period;
                        if (nowMs() > due + channel.period) { ++channel.late_dropped; ++index; continue; }
                        std::shared_ptr<cv::Mat> image;
                        if (!index) image = channel.pool[0];
                        else {
                            image = channel.acquire();
                            if (!image) { ++channel.pool_dropped; ++index; continue; }
                            if (!channel.capture.retrieve(*image) || image->size() != channel.size || image->type() != CV_8UC3)
                                throw std::runtime_error("decode failed or frame dimensions/type changed");
                        }
                        channel.decode_ms.add(nowMs() - began); ++channel.decoded;
                        waitUntil(due);
                        if (stop) break;
                        Frame frame{std::move(image), index, index * channel.period, due, nowMs()};
                        {
                            std::lock_guard<std::mutex> lock(channel.mutex);
                            channel.latest = frame;
                        }
                        const bool last = std::isfinite(channel.expected_frames) && index + 1 >= channel.expected_frames;
                        if (!options.detect_fps || index == 0 || last ||
                            frame.media_ms - last_submit_ms + .001 >= 1000 / options.detect_fps) {
                            last_submit_ms = frame.media_ms; submitted = index;
                            scheduler.put(channel.id, std::move(frame));
                        } else ++channel.rate_skipped;
                        channel.changed.notify_all();
                        ++index;
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(channel.mutex); channel.source_error = e.what();
                }
                // Drain the last actually published image even for unknown frame counts
                // or when the decoder dropped an overdue tail. No extra image history.
                if (!stop) {
                    std::lock_guard<std::mutex> lock(channel.mutex);
                    if (channel.decoded && channel.latest && channel.latest->sequence > submitted) {
                        scheduler.put(channel.id, *channel.latest); --channel.rate_skipped;
                    }
                }
                channel.capture.release(); channel.source_done = true;
                scheduler.finish(channel.id); channel.changed.notify_all();
            });
        }
        for (auto& owner : channels) {
            Channel& c = *owner;
            outputs.emplace_back([&, ptr = &c] {
                auto& channel = *ptr;
                try {
                    std::ofstream jsonl(channel.directory / "results.jsonl");
                    jsonl.exceptions(std::ios::badbit | std::ios::failbit);
                    Json info = metadata;
                    info.update({{"channel_id", channel.id}, {"input", channel.path.string()}, {"source_fps", channel.fps},
                                 {"width", channel.size.width}, {"height", channel.size.height}});
                    jsonl << info.dump() << '\n'; jsonl.flush();
                    auto& video = channel.recording;
                    int64_t tick = 0, last_drawn = -1;
                    int64_t last_correction = -1;
                    std::optional<Frame> displayed;
                    { std::lock_guard<std::mutex> lock(channel.mutex); displayed = std::move(channel.initial); channel.initial.reset(); }
                    cv::Mat drawing, resized;
                    while (true) {
                        if (callbacks.before_write) callbacks.before_write(channel.id);
                        if (channel.output_failed) throw std::runtime_error("JSONL output failed or queue overflowed");
                        std::deque<Json> events;
                        {
                            std::lock_guard<std::mutex> lock(channel.mutex); events.swap(channel.events);
                        }
                        for (const auto& row : events) jsonl << row.dump() << '\n';
                        jsonl.flush();
                        const bool drained = channel.source_done && scheduler.finished(channel.id);
                        const int64_t last_source = channel.last_source;
                        if (drained && (stop || tick > last_source)) {
                            // complete() follows event publication, so a final queue check is required.
                            std::lock_guard<std::mutex> lock(channel.mutex);
                            if (channel.events.empty()) break;
                            continue;
                        }
                        if (stop || (channel.source_done && tick > last_source) || nowMs() < start + tick * channel.period) {
                            std::unique_lock<std::mutex> lock(channel.mutex);
                            channel.changed.wait_for(lock, std::chrono::milliseconds(2)); continue;
                        }
                        // Repeat the previously drawn image for missed output ticks. Never attach a
                        // newer source image to an older tick. On the current tick take the latest image.
                        const int64_t current_tick = int64_t(std::floor((nowMs() - start) / channel.period));
                        if ((!options.record || channel.record_failed) && current_tick > tick) {
                            const int64_t advance = channel.source_done ? std::min(current_tick, last_source) : current_tick;
                            if (advance > tick) { channel.output_skipped += advance - tick; tick = advance; }
                        }
                        std::optional<Frame> frame;
                        std::optional<TrackerSnapshot> snapshot;
                        {
                            std::lock_guard<std::mutex> lock(channel.mutex);
                            if (channel.latest && channel.latest->sequence <= tick) frame = channel.latest;
                        }
                        // A missed tick can occur between two decoder publications. Reuse the last
                        // drawing and its original source identity rather than inventing a source frame.
                        if (frame) displayed = frame;
                        else frame = displayed;
                        if (!frame) throw std::runtime_error("no initial output frame");
                        {
                            std::lock_guard<std::mutex> lock(channel.mutex);
                            for (auto it = channel.history.rbegin(); it != channel.history.rend(); ++it)
                                if (it->timestamp_ms <= frame->media_ms) { snapshot = *it; break; }
                        }
                        const double before_tracking = nowMs();
                        const auto predictions = snapshot ? snapshot->predictDetailedAt(frame->media_ms, options.track_max_age_ms,
                            channel.size.width, channel.size.height) : std::vector<TrackPrediction>{};
                        const double tracking_time = nowMs() - before_tracking;
                        channel.tracking_ms.add(tracking_time);
                        std::vector<TrackedDetection> tracks;
                        size_t predicted_count = 0, lost_count = 0;
                        for (const auto& p : predictions) {
                            tracks.push_back(p.track);
                            if (p.detection_age_ms > 1e-6) ++predicted_count;
                            if (p.lost) ++lost_count;
                        }
                        const bool corrected = snapshot && snapshot->source_sequence > last_correction;
                        if (corrected) { last_correction = snapshot->source_sequence; ++channel.correction_frames; }
                        const bool duplicate = frame->sequence == last_drawn;
                        if (duplicate) ++channel.repeated;
                        else {
                            if (predicted_count) ++channel.prediction_frames;
                            channel.predicted_boxes += predicted_count; channel.lost_boxes += lost_count;
                            for (const auto& p : predictions) channel.detection_age_ms.add(p.detection_age_ms);
                        }
                        if (snapshot) channel.age_ms.add(frame->media_ms - snapshot->timestamp_ms);
                        if (preview || (options.record && !channel.record_failed)) {
                            frame->image->copyTo(drawing);
                            drawTracks(drawing, tracks, channel.id, config);
                        }
                        if (preview) {
                            const double scale = std::min(640.0 / drawing.cols, 360.0 / drawing.rows);
                            cv::resize(drawing, resized, {std::max(1, int(std::round(drawing.cols * scale))),
                                                        std::max(1, int(std::round(drawing.rows * scale)))});
                            std::lock_guard<std::mutex> lock(channel.mutex);
                            channel.preview.setTo(cv::Scalar(0,0,0));
                            resized.copyTo(channel.preview(cv::Rect((640-resized.cols)/2, (360-resized.rows)/2, resized.cols, resized.rows)));
                            const std::string age = snapshot ? "result age " + std::to_string(int(frame->media_ms - snapshot->timestamp_ms)) + "ms" : "waiting for detection";
                            cv::putText(channel.preview, age, {8, 345}, cv::FONT_HERSHEY_SIMPLEX, .5, {0,255,255}, 1);
                        }
                        if (options.record && !channel.record_failed) {
                            if (nowMs() - (start + tick * channel.period) > recording_lag_limit_ms) {
                                channel.recordFailure("recording more than 250 ms behind playback"); if (video) video->finish();
                            } else {
                                try {
                                    const double before_record = nowMs();
                                    video->write(drawing, tick); ++channel.recorded;
                                    channel.record_ms.add(nowMs() - before_record);
                                    if (nowMs() - (start + tick * channel.period) > recording_lag_limit_ms) {
                                        channel.recordFailure("recording write exceeded 250 ms lag limit"); video->finish();
                                    }
                                } catch (const std::exception& e) { channel.recordFailure(e.what()); }
                            }
                        }
                        Json values = Json::array();
                        for (const auto& prediction : predictions) {
                            const auto& t = prediction.track;
                            auto row = detectionJson(t.detection, config); row["track_id"] = t.track_id;
                            row["track_uid"] = std::to_string(channel.id) + ":" + std::to_string(t.track_id);
                            row["last_detection_source_frame_index"] = prediction.last_detection_sequence;
                            row["last_detection_timestamp_ms"] = prediction.last_detection_ms;
                            row["detection_age_ms"] = prediction.detection_age_ms;
                            row["prediction_ms"] = prediction.prediction_ms;
                            row["center_stddev_px"] = prediction.center_stddev_px;
                            row["state"] = prediction.lost ? "lost" : "tracked";
                            row["predicted"] = prediction.detection_age_ms > 1e-6;
                            values.push_back(std::move(row));
                        }
                        const double emitted = nowMs(); channel.output_ms.add(emitted - frame->due_ms);
                        Json row{{"type", "frame"}, {"channel_id", channel.id}, {"output_index", tick},
                            {"output_timestamp_ms", tick * channel.period}, {"source_frame_index", frame->sequence},
                            {"media_timestamp_ms", frame->media_ms}, {"emitted_monotonic_ms", emitted},
                            {"width", channel.size.width}, {"height", channel.size.height}, {"duplicate", duplicate},
                            {"tracks_source", snapshot ? "kalman" : "empty"},
                            {"tracking_ms", tracking_time}, {"detection_correction_applied", corrected},
                            {"predicted_boxes", predicted_count}, {"lost_predictions", lost_count},
                            {"detection_source_frame_index", snapshot ? snapshot->source_sequence : -1},
                            {"detection_age_ms", snapshot ? Json(frame->media_ms - snapshot->timestamp_ms) : Json(nullptr)},
                            {"tracks", std::move(values)}, {"record_active", options.record && !channel.record_failed}};
                        jsonl << row.dump() << '\n'; jsonl.flush();
                        last_drawn = frame->sequence; ++channel.output_frames; ++tick;
                    }
                    if (video) {
                        try { video->finish(); }
                        catch (const std::exception& e) { channel.recordFailure(e.what()); }
                    }
                    jsonl << Json{{"type", "output_end"}, {"channel_id", channel.id},
                        {"recorded_frames", channel.recorded.load()}, {"output_frames", channel.output_frames.load()}}.dump() << '\n';
                    jsonl.flush();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(channel.mutex);
                    if (channel.output_error.empty()) channel.output_error = e.what();
                    channel.output_failed = true; channel.events.clear();
                }
                channel.output_done = true;
            });
        }
        if (preview) cv::namedWindow("RK3588 four video", cv::WINDOW_AUTOSIZE);
        double sampled = 0, shown = 0;
        cv::Mat mosaic(720, 1280, CV_8UC3);
        while (true) {
            const double time = nowMs();
            if (!stop && ((signal_stop && *signal_stop) ||
                    (options.duration_seconds && time - start >= options.duration_seconds * 1000))) {
                stop_reason = signal_stop && *signal_stop ? "signal" : "duration"; requestStop();
            }
            if (time - sampled >= 1000) {
                auto row = sampler.sample(); row["elapsed_ms"] = time - start; row["channels"] = Json::array();
                auto stats = scheduler.stats();
                for (size_t i = 0; i < channels.size(); ++i) row["channels"].push_back(channelStats(*channels[i], stats[i], (time - start) / 1000));
                resource_file << row.dump() << '\n'; resource_file.flush(); sampled = time;
            }
            if (preview && time - shown >= 1000.0 / 30) {
                mosaic.setTo(cv::Scalar(0,0,0));
                for (size_t i = 0; i < channels.size(); ++i) {
                    auto& channel = *channels[i];
                    const cv::Rect rect(int(i % 2) * 640, int(i / 2) * 360, 640, 360);
                    std::lock_guard<std::mutex> lock(channel.mutex);
                    channel.preview.copyTo(mosaic(rect));
                    const double seconds = (time - start) / 1000;
                    const std::string label = "CH" + std::to_string(i) + " out " +
                        std::to_string(int(channel.output_frames / std::max(.001, seconds))) + " det " +
                        std::to_string(int(channel.inferred / std::max(.001, seconds))) +
                        (channel.source_done ? " EOF" : "") + (channel.output_failed || channel.record_failed || !channel.source_error.empty() ? " ERROR" : "");
                    cv::putText(mosaic, label, {rect.x + 8, rect.y + 24}, cv::FONT_HERSHEY_SIMPLEX, .55, {0,255,255}, 1);
                }
                cv::imshow("RK3588 four video", mosaic);
                const int key = cv::waitKey(1);
                if (!stop && (key == 27 || key == 'q' || key == 'Q' || cv::getWindowProperty("RK3588 four video", cv::WND_PROP_VISIBLE) < 1)) {
                    stop_reason = "preview_closed"; requestStop();
                }
                shown = time;
            }
            bool done = true;
            for (auto& c : channels) if (!c->output_done) done = false;
            if (done) break;
            std::unique_lock<std::mutex> lock(wait_mutex);
            wake.wait_for(lock, std::chrono::milliseconds(5));
        }
    } catch (const std::exception& e) { fail(e.what()); }
    requestStop(); join();
    if (preview) cv::destroyWindow("RK3588 four video");
    const double elapsed = nowMs() - start;
    Json summary{{"schema_version", 1}, {"status", "completed"}, {"stop_reason", stop_reason},
        {"tracking", "bytetrack_native_kalman"}, {"detect_fps_limit", options.detect_fps},
        {"track_max_age_ms", options.track_max_age_ms},
        {"fatal_error", fatal_error}, {"elapsed_ms", elapsed}, {"model", config.model_path},
        {"cpu_percent", elapsed > 0 ? (cpuSeconds() - cpu_start) * 100000 / elapsed : 0}, {"channels", Json::array()}};
    const auto dispatch_stats = scheduler.stats();
    uint64_t total = 0;
    for (size_t i = 0; i < channels.size(); ++i) {
        auto& c = *channels[i];
        auto stats = channelStats(c, dispatch_stats[i], elapsed / 1000);
        if (!stats["source_error"].get<std::string>().empty() || c.output_failed || c.record_failed) summary["status"] = "failed";
        total += c.inferred; summary["channels"].push_back(stats);
        std::ofstream file(c.directory / "summary.json"); file.exceptions(std::ios::badbit | std::ios::failbit);
        file << stats.dump(2) << '\n';
    }
    if (!fatal_error.empty()) summary["status"] = "failed";
    else if (summary["status"] != "failed" && stop_reason != "eof") summary["status"] = "stopped";
    summary["inference_count"] = total;
    summary["inference_fps"] = elapsed > 0 ? total * 1000 / elapsed : 0;
    for (auto& row : summary["channels"]) row["dispatch_share"] = total ? row["inference_count"].get<double>() / total : 0;
    std::ofstream file(options.output_dir / "summary.json"); file.exceptions(std::ios::badbit | std::ios::failbit);
    file << summary.dump(2) << '\n';
    return summary;
}
} // namespace yolo::multi

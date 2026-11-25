#include "camera/live_pipeline.h"
#include "camera/packet_channel.h"
#include "camera/stream_output.h"
#include "model/yolo_engine.h"
#include "tracking/byte_tracker.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pthread.h>
#include <rknn_api.h>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>

namespace yolo::live {
using Json = nlohmann::json;
namespace {
// 固定 1 ms 桶统计分位数，避免为每帧耗时保存无限增长的样本数组。
class Distribution {
public:
    void add(double ms) {
        if (!std::isfinite(ms) || ms < 0) return;
        std::lock_guard<std::mutex> lock(mutex_);
        ++(*bins_)[std::min<size_t>(4096, size_t(std::ceil(ms)))]; ++count_; sum_ += ms; maximum_ = std::max(maximum_, ms);
    }
    Json json() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto percentile = [&](double p) { uint64_t n = 0; for (size_t i = 0; i < bins_->size(); ++i) {
            n += (*bins_)[i]; if (n >= std::ceil(count_ * p)) return double(i); } return 0.0; };
        return {{"count", count_}, {"mean_ms", count_ ? sum_ / count_ : 0}, {"max_ms", maximum_},
            {"p50_ms", count_ ? percentile(.5) : 0}, {"p95_ms", count_ ? percentile(.95) : 0},
            {"p99_ms", count_ ? percentile(.99) : 0}, {"quantile_resolution_ms", 1}, {"overflow_bin_ms", 4096}};
    }
private:
    mutable std::mutex mutex_;
    // Histograms are fixed-size but numerous; keep them off the live pipeline's stack.
    std::unique_ptr<std::array<uint64_t, 4097>> bins_ = std::make_unique<std::array<uint64_t,4097>>();
    uint64_t count_ = 0;
    double sum_ = 0, maximum_ = 0;
};
// 检测结果和 JSONL 都有容量上限；满 250 ms 触发停止，避免下游阻塞导致积压。
// 与 LatestSlot 不同：正常运行中不主动替换旧结果，finish 保留已排队的数据。
template<class T> class BoundedQueue {
public:
    using Id = uint64_t (*)(const T&);
    explicit BoundedQueue(size_t capacity, Id id = nullptr) : capacity_(capacity), id_(id) {}
    void event(trace::Kind k, const T& v) {
        if (trace::enabled() && id_)
            trace::emit(k, trace::ResultQueue, id_(v));
    }
    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!space_.wait_for(lock, std::chrono::milliseconds(250), [&] { return closed_ || queue_.size() < capacity_; }))
            throw std::runtime_error("bounded queue full for 250 ms");
        if (closed_) throw std::runtime_error("bounded queue closed");
        event(trace::Enter, value);
        queue_.push_back(std::move(value));
        peak_ = std::max(peak_, queue_.size());
        data_.notify_one();
    }
    std::optional<T> take() {
        std::unique_lock<std::mutex> lock(mutex_);
        data_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return {};
        event(trace::Leave, queue_.front());
        T value = std::move(queue_.front());
        queue_.pop_front();
        space_.notify_one();
        return value;
    }
    std::vector<T> drainReady() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> result;
        while (!queue_.empty()) {
            event(trace::Leave, queue_.front());
            result.push_back(std::move(queue_.front()));
            queue_.pop_front();
        }
        space_.notify_all(); return result;
    }
    void finish() { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; data_.notify_all(); space_.notify_all(); }
    void cancel() { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; queue_.clear(); data_.notify_all(); space_.notify_all(); }
    size_t peak() const { std::lock_guard<std::mutex> lock(mutex_); return peak_; }
private:
    mutable std::mutex mutex_;
    std::condition_variable data_, space_;
    std::deque<T> queue_;
    size_t capacity_, peak_ = 0;
    Id id_;
    bool closed_ = false;
};
// 单独线程顺序写 JSONL。队列只保存检测/轨迹等数值，不延长图像 DMA-BUF 的租约。
class Writer {
public:
    Writer(const LiveOptions& options, std::function<void(std::string)> fail, std::function<void()> before_write)
        : before_write_(std::move(before_write)), fail_(std::move(fail)) {
        const int fd = open(options.output.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd < 0) throw std::runtime_error("cannot create new JSONL: " + options.output + ": " + std::strerror(errno));
        file_ = fdopen(fd, "w");
        if (!file_) { close(fd); throw std::runtime_error("fdopen JSONL failed"); }
        try {
            thread_ = std::thread([this] {
                trace::thread(trace::Writer);
                run();
            });
        } catch (...) {
            fclose(file_);
            file_ = nullptr;
            throw;
        }
    }
    ~Writer() { queue_.finish(); if (thread_.joinable()) thread_.join(); if (file_) fclose(file_); }
    void push(Json row) {
        { std::lock_guard<std::mutex> lock(mutex_); if (!error_.empty()) throw std::runtime_error(error_); }
        queue_.push(std::move(row));
    }
    void finish() {
        queue_.finish(); if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
    }
    size_t peak() const { return queue_.peak(); }
    void cancelPending() { queue_.cancel(); }
private:
    void run() {
        try {
            while (auto row = queue_.take()) {
                if (before_write_) before_write_();
                trace::Scope write(trace::Write, trace::enabled() ? row->value("frame_id", uint64_t(0)) : 0);
                const auto text = row->dump();
                if (fwrite(text.data(), 1, text.size(), file_) != text.size() || fputc('\n', file_) == EOF || fflush(file_))
                    throw std::runtime_error("camera_live JSONL write failed");
            }
        } catch (const std::exception& e) {
            { std::lock_guard<std::mutex> lock(mutex_); error_ = e.what(); }
            queue_.finish(); fail_(e.what());
        }
    }
    BoundedQueue<Json> queue_{64};
    std::function<void()> before_write_;
    std::function<void(std::string)> fail_;
    FILE* file_ = nullptr;
    std::thread thread_;
    std::mutex mutex_;
    std::string error_;
};
Json detectionJson(const Detection& d) {
    return {{"class_id", d.class_id}, {"score", d.score}, {"bbox", {d.x1,d.y1,d.x2,d.y2}}};
}
Json stabilizationJson(const StabilizationInfo& s) {
    return {{"enabled",s.enabled},{"roi",{s.roi.x,s.roi.y,s.roi.width,s.roi.height}},
        {"motion_valid",s.valid},{"reference_reset",s.reference_reset},{"clamped",s.clamped},
        {"delta_px",{s.delta.x,s.delta.y}},{"compensation_px",{s.compensation.x,s.compensation.y}},
        {"sampled_points",s.sampled_points},{"valid_points",s.valid_points},{"inliers",s.inliers},
        {"background_cells",s.background_cells},{"inlier_ratio",s.inlier_ratio},
        {"mask_source_timestamp_ms",s.mask_source_timestamp_ms<0 ? Json(nullptr) : Json(s.mask_source_timestamp_ms)},
        {"mask_target_timestamp_ms",s.mask_target_timestamp_ms<0 ? Json(nullptr) : Json(s.mask_target_timestamp_ms)},
        {"gray_ms",s.gray_ms},{"motion_ms",s.motion_ms},{"total_ms",s.total_ms}};
}
struct Resources {
    size_t first_fd = 0, last_fd = 0, max_fd = 0;
    double first_rss = 0, last_rss = 0, max_rss = 0, max_temp = 0;
    uint64_t samples = 0;
    Json sample() {
        size_t fd = 0;
        for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd")) { (void)e; ++fd; }
        uint64_t size = 0, resident = 0; std::ifstream("/proc/self/statm") >> size >> resident;
        const double rss = resident * double(sysconf(_SC_PAGESIZE)) / 1048576;
        double temp = 0;
        for (const auto& e : std::filesystem::directory_iterator("/sys/class/thermal")) {
            if (e.path().filename().string().rfind("thermal_zone", 0)) continue;
            double value = 0; std::ifstream(e.path() / "temp") >> value; temp = std::max(temp, value / 1000);
        }
        std::string load; std::getline(std::ifstream("/sys/kernel/debug/rknpu/load"), load);
        if (!samples++) { first_fd = fd; first_rss = rss; }
        last_fd = fd; max_fd = std::max(max_fd, fd); last_rss = rss; max_rss = std::max(max_rss, rss); max_temp = std::max(max_temp, temp);
        return {{"fd_count", fd}, {"rss_mb", rss}, {"temperature_c", temp}, {"npu_load", load}};
    }
    Json json() const { return {{"samples", samples}, {"first_fd", first_fd}, {"last_fd", last_fd}, {"max_fd", max_fd},
        {"first_rss_mb", first_rss}, {"last_rss_mb", last_rss}, {"max_rss_mb", max_rss}, {"max_temperature_c", max_temp}}; }
};
// 检测框已恢复为原图坐标；源序号/时间用于统一跟踪，不把 Worker 编号当轨迹 ID。
struct Completed { Stamp stamp; InferResult result; double completed_ms = 0; int worker = 0; StabilizationInfo stabilization; };
}

Json runLive(FrameSource& source, const AppConfig& config, const LiveOptions& options,
             const volatile sig_atomic_t* signal_stop, const LiveCallbacks& callbacks) {
    if (options.preview && !std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
        throw std::runtime_error("preview needs a logged-in graphical session; use --preview off for headless validation");
    if (options.npu_mode != "default" && options.npu_mode != "all" && options.npu_mode != "split")
        throw std::runtime_error("invalid NPU mode");
    if (!std::isfinite(options.detect_fps) || options.detect_fps < 0 || options.detect_fps > 1000 ||
        !std::isfinite(options.max_predict_ms) || options.max_predict_ms < 0 || options.max_predict_ms > 1000 ||
        !std::isfinite(options.stabilize_alpha) || options.stabilize_alpha <= 0 || options.stabilize_alpha > 1)
        throw std::runtime_error("invalid detection frequency/prediction horizon");
    // 每个显式掩码对应一个独立 Context 和一条追踪轨道。默认 split 使用 1、2、4。
    if (options.masks.size() > 3) throw std::invalid_argument("at most three NPU workers supported");
    const int worker_count =
        options.masks.empty() ? (options.npu_mode == "split" ? 3 : 1) : int(options.masks.size());
    trace::emit(trace::Counter, trace::PoolCapacity, 0, 0, worker_count, trace::RgbPool);
    std::vector<std::unique_ptr<YoloEngine>> engines;
    std::vector<std::unique_ptr<ImagePreprocessor>> processors;
    for (int i = 0; i < worker_count; ++i) {
        // 位掩码 1/2/4 分别选择 Core0/1/2；不是 CPU 线程绑核，也不是帧轮询编号。
        const int mask = !options.masks.empty() ? options.masks.at(i)
                         : options.npu_mode == "split"
                             ? (1 << i)
                             : (options.npu_mode == "all" ? int(RKNN_NPU_CORE_0_1_2) : -1);
        trace::emit(trace::Instant, trace::WorkerMask, 0, 0, uint32_t(mask), trace::Worker0 + i);
        engines.push_back(std::make_unique<YoloEngine>(config, mask, false, options.cpu_copy));
        // 每套 Context 独占预处理器和 RGB 目标，按该模型查询到的 stride/容量分配。
        // 各 Worker 不共用一个可写 RGB 输入，也不并发调用同一个预处理实例。
        processors.push_back(std::make_unique<ImagePreprocessor>(config, engines.back()->inputMemoryLayout()));
    }
    MjpegDecoder decoder(config.inference_timeout_ms, options.surface_count);
    Renderer renderer;
    FrameStabilizer stabilizer(options.stabilize_alpha);
    if (options.stream && (!options.stream_port || options.stream_bitrate < 64000 || options.stream_bitrate > 14000000))
        throw std::runtime_error("invalid streaming options");
    std::unique_ptr<StreamOutput> stream;
    if (options.stream) stream = std::make_unique<StreamOutput>(options.stream_port, options.stream_bitrate,
        config.inference_timeout_ms, callbacks.before_encode, options.unpaced);
    // 解码后的 NV12 分给推理与输出队列，各自通过阻塞或替换等待项控制积压。
    // 替换等待项释放该项租约；正在执行的线程仍持有自己的副本，不会被替换覆盖。
    PacketChannel packets(options.packet_spsc, options.packet_fifo || options.unpaced);
    auto decoded_id = [](const DecodedFrame &f) { return f.stamp.frame_id; };
    LatestSlot<DecodedFrame> infer_frames(trace::InferQueue, decoded_id, options.infer_depth,
                                          options.queue_policy == "fifo",
                                          [](DecodedFrame &f, uint64_t n) { f.stamp.dispatch = n; });
    LatestSlot<DecodedFrame> output_frames(trace::OutputQueue, decoded_id, 1, options.unpaced);
    std::mutex order_mutex;
    std::condition_variable order_changed;
    uint64_t next_dispatch = 1;
    BoundedQueue<Completed> results(
        3, [](const Completed &c) { return c.stamp.frame_id; }); // 各推理结果汇入唯一的轨迹校正线程。
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> decoded{0}, pool_dropped{0}, inference_count{0}, late_results{0}, accepted{0}, detect_rate_skipped{0};
    std::atomic<int> remaining_workers{worker_count};
    std::mutex error_mutex, snapshot_mutex;
    std::string error;
    std::deque<TrackerSnapshot> snapshots;
    auto snapshotAt = [&](double timestamp) -> std::optional<TrackerSnapshot> {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        for (auto it=snapshots.rbegin(); it!=snapshots.rend(); ++it)
            if (it->timestamp_ms<=timestamp) return *it;
        return {};
    };
    // close 舍弃尚未开始的帧，并唤醒阻塞消费者；不会撤销已提交的硬件任务。
    auto stopNewWork = [&] {
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            stop.store(true);
        }
        order_changed.notify_all();
        packets.close();
        infer_frames.close();
        output_frames.close();
    };
    auto fail = [&](std::string text) {
        trace::emit(trace::Error, trace::FrameError, trace::currentFrame());
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (error.empty())
                error = std::move(text);
        }
        stopNewWork();
    };
    Writer writer(options, fail, callbacks.before_write);
    auto metadata = source.info();
    const auto geometry = resizeGeometry(800, 600, config);
    metadata.update({{"type", "camera_live"},
                     {"schema_version", 2},
                     {"decoder", "mpp"},
                     {"npu_mode", options.npu_mode},
                     {"workers", worker_count},
                     {"preview", options.preview},
                     {"stream",
                      {{"enabled", options.stream},
                       {"port", options.stream_port},
                       {"bitrate", options.stream_bitrate},
                       {"codec", "h264"},
                       {"transport", "loopback_udp_mpegts"}}},
                     {"stabilization",
                      {{"enabled", options.stabilize},
                       {"alpha", options.stabilize_alpha},
                       {"analysis_width", 320},
                       {"analysis_height", 240},
                       {"crop_width", 720},
                       {"crop_height", 540},
                       {"loss_policy", "smooth_recenter"},
                       {"transform", "view=(raw-roi_origin)*view_size/roi_size"}}},
                     {"max_predict_ms", options.max_predict_ms},
                     {"detect_fps", options.detect_fps},
                     {"model", config.model_path},
                     {"model_width", config.input_width},
                     {"model_height", config.input_height},
                     {"num_classes", config.num_classes},
                     {"labels_path", config.labels_path},
                     {"bbox_coordinates", options.stabilize ? "stabilized_800x600" : "source_800x600"},
                     {"padding",
                      {{"top", geometry.top},
                       {"bottom", config.input_height - geometry.height - geometry.top},
                       {"left", geometry.left},
                       {"right", config.input_width - geometry.width - geometry.left},
                       {"rgb", {114, 114, 114}}}},
                     {"scale_x", geometry.transform.scale_x},
                     {"scale_y", geometry.transform.scale_y},
                     {"packet_pool", metadata.value("packet_pool_buffers", 4)},
                     {"decoded_pool", options.surface_count},
                     {"snapshot_history", 64},
                     {"json_queue_capacity", 64}});
    writer.push(metadata);
    Distribution decode_ms, preprocess_ms, infer_ms, postprocess_ms, result_age_ms, output_latency_ms, correction_age_ms, prediction_ms;
    Distribution stabilize_ms, gray_ms, motion_ms;
    uint64_t stabilization_frames=0, stabilization_valid=0, stabilization_clamped=0, stabilization_reset=0;
    Resources resources;
    rusage usage_start{}; getrusage(RUSAGE_SELF, &usage_start);
    const double start = nowMs();
    double capture_origin = 0, capture_last = 0;
    uint64_t window_inputs = 0;
    uint64_t output_count = 0, empty_snapshot_frames = 0;
    double first_output_ts = 0, last_output_ts = 0;
    std::vector<std::thread> workers;
    std::thread capture_thread, decode_thread, correction_thread;
    // 先等生产者和推理全部退出，再结束结果消费；避免硬件尚在读写就析构帧池。
    auto join = [&] {
        if (capture_thread.joinable()) capture_thread.join();
        if (decode_thread.joinable()) decode_thread.join();
        for (auto& t : workers) if (t.joinable()) t.join();
        results.finish();
        if (correction_thread.joinable()) correction_thread.join();
    };
    try {
        correction_thread = std::thread([&] {
            trace::thread(trace::Tracker);
            try {
                // 全流水线只有这一份权威跟踪状态：跨 Worker 统一关联并分配 ID。
                // 每个 Worker 只做检测，不能各自创建 tracker，否则同一目标会产生多套 ID。
                ByteTracker tracker;
                int64_t watermark = -1;
                double last_source_ms = -1;
                while (auto first = results.take()) {
                    // 只对当前已完成的一批排序，不等待更早的慢帧；已经越过时间水位的
                    // 迟到结果随后丢弃，防止旧检测让轨迹回退到过去。
                    auto ready = results.drainReady();
                    ready.push_back(std::move(*first));
                    std::sort(ready.begin(), ready.end(), [](const auto& a, const auto& b) { return a.stamp.timestamp_ms < b.stamp.timestamp_ms; });
                    for (auto& item : ready) {
                        trace::frame(item.stamp.frame_id);
                        if (item.stamp.sequence <= watermark || item.stamp.timestamp_ms <= last_source_ms) {
                            trace::emit(trace::Drop, trace::Correction, item.stamp.frame_id, 0, 4);
                            if (options.benchmark_seconds)
                                writer.push({{"type", "late_detection"},
                                             {"frame_id", item.stamp.frame_id},
                                             {"sequence", item.stamp.sequence},
                                             {"arrival_ms", item.stamp.arrival_ms},
                                             {"completed_ms", item.completed_ms}});
                            ++late_results;
                            continue;
                        }
                        trace::Scope tracking(trace::Track);
                        auto snapshot = tracker.updateAt(item.result.detections, item.stamp.sequence, item.stamp.timestamp_ms);
                        tracking.end();
                        snapshot.trace_frame_id = item.stamp.frame_id;
                        snapshot.trace_arrival_ms = item.stamp.arrival_ms;
                        watermark = item.stamp.sequence;
                        last_source_ms = item.stamp.timestamp_ms;
                        ++accepted;
                        {
                            // 输出线程只拷贝不可变快照，再自行预测；不共享可变 tracker。
                            std::lock_guard<std::mutex> lock(snapshot_mutex);
                            snapshots.push_back(std::move(snapshot));
                            trace::emit(trace::Instant, trace::Correction, item.stamp.frame_id);
                            if (snapshots.size() > 64)
                                snapshots.pop_front();
                        }
                        Json values = Json::array();
                        for (const auto &d : item.result.detections)
                            values.push_back(detectionJson(d));
                        writer.push({{"type", "detection"},
                                     {"frame_id", item.stamp.frame_id},
                                     {"arrival_ms", item.stamp.arrival_ms},
                                     {"sequence", item.stamp.sequence},
                                     {"timestamp_ms", item.stamp.timestamp_ms},
                                     {"completed_ms", item.completed_ms},
                                     {"worker", item.worker},
                                     {"detections", values},
                                     {"preprocess_ms", item.result.preprocess_ms},
                                     {"infer_ms", item.result.infer_ms},
                                     {"postprocess_ms", item.result.postprocess_ms},
                                     {"stabilization", stabilizationJson(item.stabilization)}});
                    }
                }
            } catch (const std::exception& e) {
                fail(e.what());
            }
        });
        // 线程只在启动时创建。按值捕获 i，保证每个线程始终使用自己的 engine/processor。
        // 各 Worker 竞争同一推理队列。take() 在锁内分配派发序号，每帧只交给一个 Worker。
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back([&, i] {
                trace::thread(static_cast<trace::Lane>(trace::Worker0 + i));
                try {
                    if (options.affinity) {
                        cpu_set_t cpus;
                        CPU_ZERO(&cpus);
                        CPU_SET(4 + i, &cpus);
                        if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus))
                            throw std::runtime_error("worker affinity failed");
                    }
                    for (;;) {
                        trace::Scope waiting(trace::WorkerWait, 0);
                        auto frame = infer_frames.take();
                        waiting.end();
                        if (!frame)
                            break;
                        trace::frame(frame->stamp.frame_id);
                        if (stop.load()) {
                            trace::emit(trace::Drop, trace::Detection, frame->stamp.frame_id, 0, 2);
                            break;
                        }
                        trace::Scope task(trace::WorkerTask);
                        const auto stamp = frame->stamp;
                        const double before = nowMs();
                        // MPP 输出带共享租约的 NV12。
                        // prepare 内 RGA 用 IM_SYNC，返回时独立 RGB 已就绪。
                        const auto stabilization = frame->stabilization;
                        auto input = processors[i]->prepareNv12(frame->image, stabilization.roi);
                        const double prepare = nowMs() - before;
                        // 归还推理侧 NV12 源租约；另一输出侧若仍持有该 Surface，它仍不可复用。
                        // input 继续持有 RGB 租约，覆盖后面的 RKNN 绑定、执行和后处理。
                        frame.reset();
                        if (callbacks.before_infer)
                            callbacks.before_infer(i);
                        auto result = engines[i]->infer(input, stamp.sequence);
                        result.preprocess_ms = prepare;
                        const double completed = nowMs();
                        trace::emit(trace::Instant, trace::Detection, stamp.frame_id);
                        ++inference_count;
                        preprocess_ms.add(prepare);
                        infer_ms.add(result.infer_ms);
                        postprocess_ms.add(result.postprocess_ms);
                        result_age_ms.add(completed -
                                          stamp.latencyOrigin(options.unpaced));
                        std::unique_lock<std::mutex> order(order_mutex, std::defer_lock);
                        if (options.ordered) {
                            trace::Scope reorder(trace::Reorder);
                            order.lock();
                            order_changed.wait(order,
                                               [&] { return stop || stamp.dispatch == next_dispatch; });
                            if (stop) {
                                trace::emit(trace::Drop, trace::Correction, stamp.frame_id, 0, 2);
                                break;
                            }
                        }
                        results.push({stamp, std::move(result), completed, i, stabilization});
                        if (options.ordered) {
                            ++next_dispatch;
                            order.unlock();
                            order_changed.notify_all();
                        }
                    }
                } catch (const std::exception &e) {
                    fail(e.what());
                }
                // 只有最后一个 Worker 退出才关闭结果队列，让校正线程收完已完成结果。
                if (--remaining_workers == 0)
                    results.finish();
            });
        decode_thread = std::thread([&] {
            trace::thread(trace::Decoder);
            try {
                double last_submit = -1;
                while (auto packet = packets.take()) {
                    trace::frame(packet->stamp.frame_id);
                    if (stop.load()) {
                        trace::emit(trace::Drop, trace::Decode, packet->stamp.frame_id, 0, 2);
                        break;
                    }
                    // decode 返回前确认 MPP 完成；之后压缩输入可归还固定 DMA 包池。
                    // 返回的 NV12 则由独立 Surface owner 继续保护，两种占用互不捆绑。
                    trace::Scope decode(trace::Decode);
                    auto frame = decoder.decode(*packet);
                    decode.end();
                    packet.reset();
                    if (!frame) {
                        trace::emit(trace::Drop, trace::Decode, trace::currentFrame(), 0, 3);
                        ++pool_dropped;
                        continue;
                    }
                    ++decoded;
                    decode_ms.add(frame->decode_ms);
                    if (options.stabilize) {
                        trace::Scope stabilization_span(trace::Stabilize);
                        const double previous_time=stabilizer.previousTimestamp();
                        const auto mask_snapshot=snapshotAt(previous_time);
                        const auto excluded=mask_snapshot ? mask_snapshot->predictAt(previous_time,options.max_predict_ms,800,600)
                                                          : std::vector<TrackedDetection>{};
                        auto info=stabilizer.update(frame->image,frame->stamp.timestamp_ms,excluded);
                        info.mask_target_timestamp_ms=previous_time;
                        if (mask_snapshot)
                            info.mask_source_timestamp_ms = mask_snapshot->timestamp_ms;
                        frame->stabilization=info;
                        ++stabilization_frames;
                        stabilization_valid += info.valid;
                        stabilization_clamped+=info.clamped;
                        stabilization_reset += info.reference_reset;
                        stabilize_ms.add(info.total_ms);
                        gray_ms.add(info.gray_ms);
                        motion_ms.add(info.motion_ms);
                    }
                    if (last_submit < 0 || options.detect_fps == 0 || frame->stamp.timestamp_ms - last_submit >= 1000 / options.detect_fps) {
                        // 这里只复制帧描述/shared_ptr，不复制 NV12 像素。后面移交输出侧后，
                        // 同一 Surface 的两路只读消费者分别归还自己的 owner。
                        last_submit = frame->stamp.timestamp_ms;
                        infer_frames.put(*frame);
                    } else {
                        trace::emit(trace::Drop, trace::Detection, frame->stamp.frame_id, 0, 5);
                        ++detect_rate_skipped;
                    }
                    output_frames.put(std::move(*frame));
                }
            } catch (const std::exception& e) {
                fail(e.what());
            }
            // 正常 EOF 保留最后的等待帧；与主动停止时 close() 丢弃等待项的语义不同。
            infer_frames.finish();
            output_frames.finish();
        });
        capture_thread = std::thread([&] {
            trace::thread(trace::Camera);
            try {
                while (!stop.load()) {
                    // CameraSource 已轮换 DMA-BUF 并 QBUF；队列保留出队压缩包租约。
                    auto packet = source.next(stop);
                    if (!packet)
                        break;
                    capture_last = packet->stamp.arrival_ms;
                    if (!capture_origin)
                        capture_origin = packet->stamp.arrival_ms;
                    const double age = (packet->stamp.arrival_ms - capture_origin) / 1000;
                    if (age >= options.warmup_seconds &&
                        (!options.benchmark_seconds ||
                         age < options.warmup_seconds + options.benchmark_seconds))
                        ++window_inputs;
                    packets.put(std::move(*packet));
                    if (options.benchmark_seconds &&
                        age >= options.warmup_seconds + options.benchmark_seconds)
                        break;
                }
            } catch (const std::exception& e) {
                fail(e.what());
            }
            packets.finish();
        });
        // 调用线程独立消费输出帧，不等待当前帧的检测完成，以保持视频输出及时。
        if (options.preview) cv::namedWindow("RK3588 camera_live", cv::WINDOW_AUTOSIZE);
        bool eof = false;
        while (!stop.load()) {
            if (signal_stop && *signal_stop) break;
            auto frame = output_frames.take(false);
            if (!frame) { // 非阻塞轮询，让采集超时期间信号和窗口事件仍能被处理。
                if (options.preview) {
                    const int idle_key = cv::waitKey(1);
                    if (idle_key == 27 || idle_key == 'q' || idle_key == 'Q' ||
                        cv::getWindowProperty("RK3588 camera_live", cv::WND_PROP_VISIBLE) < 1) break;
                }
                // 在同一锁内检查关闭状态和等待项，避免另一个完成标记使最后一帧丢失。
                if (output_frames.finished()) {
                    eof = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue;
            }
            const auto stamp = frame->stamp;
            trace::frame(stamp.frame_id);
            const auto snapshot=snapshotAt(stamp.timestamp_ms);
            const auto stabilization=frame->stabilization;
            trace::emit(trace::Instant, trace::Snapshot, stamp.frame_id, 0,
                        snapshot ? snapshot->trace_frame_id : 0);
            trace::Scope render(trace::Render);
            auto image = renderer.convert(*frame);
            // RGA 同步转换完成后归还输出侧 NV12 租约；CPU 只在独立 BGR 上画框。
            frame.reset();
            // 把快照预测到当前图像时间；max_predict_ms 限制距最后真实匹配的年龄。
            const auto tracks = snapshot ? snapshot->predictAt(stamp.timestamp_ms, options.max_predict_ms, 800, 600) : std::vector<TrackedDetection>{};
            Json track_rows = Json::array();
            for (const auto& track : tracks) {
                const auto& d = track.detection;
                auto row = detectionJson(d); row["track_id"] = track.track_id;
                for (const auto& s : snapshot->states) if (s.track.track_id == track.track_id) row["last_match_age_ms"] = stamp.timestamp_ms - s.last_match_ms;
                track_rows.push_back(std::move(row));
                cv::rectangle(image, cv::Point(cvRound(d.x1), cvRound(d.y1)), cv::Point(cvRound(d.x2), cvRound(d.y2)), {40,220,40}, 2);
                cv::putText(image, std::to_string(track.track_id) + " " + config.class_names.at(d.class_id),
                    {cvRound(d.x1), std::max(18, cvRound(d.y1) - 4)}, cv::FONT_HERSHEY_SIMPLEX, .5, {40,220,40}, 1);
            }
            cv::putText(image, "NPU " + options.npu_mode + "  seq " + std::to_string(stamp.sequence), {10,590}, cv::FONT_HERSHEY_SIMPLEX, .5, {0,220,255}, 1);
            if (callbacks.before_output) callbacks.before_output();
            int key = -1;
            if (options.preview) { cv::imshow("RK3588 camera_live", image); key = cv::waitKey(1); }
            // image 是 DMA-BUF 的借用 Mat：CPU 画框/同步预览调用后结束 RW 访问区间。
            // 不能将该 Mat 留给异步消费者并立即复用 Renderer 双缓冲。
            renderer.finish();
            const bool stream_submitted = stream && stream->submit(renderer, stamp);
            render.end();
            trace::emit(trace::Instant, trace::Output, stamp.frame_id);
            if (stream)
                trace::emit(trace::Instant, trace::StreamSubmit, stamp.frame_id, 0, stream_submitted);
            const double output_time = nowMs();
            const double latency = output_time - stamp.latencyOrigin(options.unpaced);
            output_latency_ms.add(latency);
            if (snapshot) {
                correction_age_ms.add(
                    output_time - (options.unpaced ? snapshot->trace_arrival_ms : snapshot->timestamp_ms));
                prediction_ms.add(stamp.timestamp_ms - snapshot->timestamp_ms);
            } else
                ++empty_snapshot_frames;
            if (!output_count) first_output_ts = stamp.timestamp_ms;
            last_output_ts = stamp.timestamp_ms;
            writer.push({{"type", "frame"},
                         {"frame_id", stamp.frame_id},
                         {"arrival_ms", stamp.arrival_ms},
                         {"frame_index", output_count},
                         {"sequence", stamp.sequence},
                         {"timestamp_ms", stamp.timestamp_ms},
                         {"raw_timestamp_ms", stamp.raw_timestamp_ms},
                         {"timestamp_clock", stamp.clock},
                         {"timestamp_flags", stamp.flags},
                         {"width", 800},
                         {"height", 600},
                         {"output_monotonic_ms", output_time},
                         {"output_latency_ms", latency},
                         {"stream_submitted", stream_submitted},
                         {"stabilization", stabilizationJson(stabilization)},
                         {"corrected_from_sequence", snapshot ? snapshot->source_sequence : -1},
                         {"correction_timestamp_ms", snapshot ? Json(snapshot->timestamp_ms) : Json(nullptr)},
                         {"prediction_ms",
                          snapshot ? Json(stamp.timestamp_ms - snapshot->timestamp_ms) : Json(nullptr)},
                         {"correction_age_ms",
                          snapshot ? Json(output_time - (options.unpaced ? snapshot->trace_arrival_ms
                                                                         : snapshot->timestamp_ms))
                                   : Json(nullptr)},
                         {"tracks", track_rows}});
            ++output_count;
            if (output_count == 1 || output_count % 30 == 0) {
                auto row = resources.sample(); row["type"] = "resource"; row["frame_count"] = output_count;
                if (stream) row["stream"] = stream->stats();
                writer.push(row);
            }
            if ((options.max_frames && output_count >= options.max_frames) || key == 27 || key == 'q' || key == 'Q' ||
                (options.preview && cv::getWindowProperty("RK3588 camera_live", cv::WND_PROP_VISIBLE) < 1)) break;
        }
        if (!eof)
            stopNewWork();
    } catch (const std::exception& e) { fail(e.what()); }
    // 停提交 → 等线程及已开始任务退出 → 结束 CPU 访问。函数返回后的 RAII 析构
    // 再清理 MPP/RKNN 注册和 Context；底层 allocation 由剩余共享所有者延长存活期。
    join();
    packets.discardAfterJoin();
    stabilizer.finish();
    renderer.finish();
    if (stream) stream->stop();
    if (options.preview) cv::destroyAllWindows();
    const double elapsed = nowMs() - start;
    rusage usage{}; getrusage(RUSAGE_SELF, &usage);
    auto cpuMs = [](const rusage& r) { return r.ru_utime.tv_sec * 1000.0 + r.ru_utime.tv_usec / 1000.0 + r.ru_stime.tv_sec * 1000.0 + r.ru_stime.tv_usec / 1000.0; };
    Json summary{{"type", "summary"},
                 {"status", "completed"},
                 {"error", ""},
                 {"npu_mode", options.npu_mode},
                 {"frame_count", output_count},
                 {"capture_origin_ms", capture_origin},
                 {"capture_last_ms", capture_last},
                 {"window_inputs", window_inputs},
                 {"decoded_count", decoded.load()},
                 {"inference_count", inference_count.load()},
                 {"accepted_results", accepted.load()},
                 {"late_results_dropped", late_results.load()},
                 {"decoded_pool_dropped", pool_dropped.load()},
                 {"packet_pending_dropped", packets.dropped()},
                 {"inference_pending_dropped", infer_frames.dropped()},
                 {"output_pending_dropped", output_frames.dropped()},
                 {"detect_rate_skipped", detect_rate_skipped.load()},
                 {"packet_shutdown_discarded", packets.cancelled()},
                 {"inference_shutdown_discarded", infer_frames.cancelled()},
                 {"output_shutdown_discarded", output_frames.cancelled()},
                 {"empty_snapshot_frames", empty_snapshot_frames},
                 {"result_queue_peak", results.peak()},
                 {"json_queue_peak", writer.peak()},
                 {"elapsed_ms", elapsed},
                 {"output_fps", elapsed > 0 ? output_count * 1000 / elapsed : 0},
                 {"output_timeline_fps", last_output_ts > first_output_ts
                                             ? (output_count - 1) * 1000 / (last_output_ts - first_output_ts)
                                             : 0},
                 {"inference_fps", elapsed > 0 ? inference_count.load() * 1000 / elapsed : 0},
                 {"cpu_percent", elapsed > 0 ? (cpuMs(usage) - cpuMs(usage_start)) * 100 / elapsed : 0},
                 {"resources", resources.json()},
                 {"source", source.counters()},
                 {"stabilization",
                  {{"enabled", options.stabilize},
                   {"frames", stabilization_frames},
                   {"valid_frames", stabilization_valid},
                   {"lost_frames", stabilization_frames - stabilization_valid - stabilization_reset},
                   {"reference_resets", stabilization_reset},
                   {"clamped_frames", stabilization_clamped},
                   {"total", stabilize_ms.json()},
                   {"gray", gray_ms.json()},
                   {"motion", motion_ms.json()}}},
                 {"stream", stream ? stream->stats() : Json{{"status", "disabled"}}},
                 {"timing",
                  {{"decode", decode_ms.json()},
                   {"preprocess", preprocess_ms.json()},
                   {"infer", infer_ms.json()},
                   {"postprocess", postprocess_ms.json()},
                   {"capture_to_result", result_age_ms.json()},
                   {"output_latency", output_latency_ms.json()},
                   {"correction_age", correction_age_ms.json()},
                   {"prediction", prediction_ms.json()}}}};
    if (trace::enabled())
        summary["rknn"] = engines.front()->runtimeInfo();
    { std::lock_guard<std::mutex> lock(error_mutex); if (!error.empty()) { summary["status"] = "error"; summary["error"] = error; }
      else if (signal_stop && *signal_stop) summary["status"] = "interrupted"; }
    // 写入失败时队列可能已满，舍弃待写记录后收尾；此时以 stderr 汇总为准，
    // 不保证 JSONL 含有最后一条 summary，也不让慢写入无限拖延异常退出。
    try {
        if (summary["status"] == "error") { writer.cancelPending(); writer.finish(); }
        else { writer.push(summary); writer.finish(); }
    }
    catch (const std::exception& e) { summary["status"] = "error"; summary["error"] = e.what(); }
    return summary;
}
} // namespace yolo::live

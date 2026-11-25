#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "model/inference_types.h"

namespace yolo {

struct TimedTrackState {
    TrackedDetection track;
    std::array<float, 8> mean{}; // velocities per second
    double last_match_ms = 0;
    std::array<std::array<float, 8>, 8> covariance{};
    int64_t last_detection_sequence = -1;
    bool lost = false;
};
struct TrackPrediction {
    TrackedDetection track;
    int64_t last_detection_sequence = -1;
    double last_detection_ms = 0, prediction_ms = 0, detection_age_ms = 0;
    std::array<float, 2> center_stddev_px{};
    bool lost = false;
};
struct TrackerSnapshot {
    int64_t source_sequence = -1;
    double timestamp_ms = 0;
    std::vector<TimedTrackState> states;
    uint64_t trace_frame_id = 0;
    double trace_arrival_ms = 0;
    // Pure projection: never modifies the snapshot or authoritative tracker.
    std::vector<TrackedDetection> predictAt(double timestamp, double max_age_ms,
                                          int width, int height) const;
    // Uses the same time-based Kalman transition/noise as the authoritative tracker.
    // Ages refer to the real detection match, never renewed by display predictions.
    std::vector<TrackPrediction> predictDetailedAt(double timestamp, double max_age_ms,
                                                  int width, int height) const;
};

// 单模型跟踪器：Kalman 预测结合两阶段检测关联，维持跨帧目标 ID。
class ByteTracker {
public:
    ByteTracker();
    TrackerSnapshot updateAt(const std::vector<Detection>& detections,
                             int64_t sequence, double timestamp_ms);

    std::vector<TrackedDetection> update(const std::vector<Detection>& detections);
    // Observations on the same accepted source frame share one prediction/tick.
    std::vector<TrackedDetection> update(
        const std::vector<Detection>& detections, int64_t frame_index
    );
    // 光流已提供目标 ID 时按 ID 更新既有轨迹，不创建新目标。
    std::vector<TrackedDetection> updateTracked(
        const std::vector<TrackedDetection>& tracked_detections
    );
    std::vector<TrackedDetection> updateTracked(
        const std::vector<TrackedDetection>& tracked_detections, int64_t frame_index
    );

private:
    enum class TrackState {
        Tracked,
        Lost,
        Removed,
    };

    struct Track {
        int id = -1;
        Detection detection;
        // 状态依次为中心 x/y、宽高比、高度，以及这四个量的速度。
        std::array<float, 8> mean{};
        std::array<std::array<float, 8>, 8> covariance{};
        TrackState state = TrackState::Tracked;
        bool activated = false;
        bool matched = false;
        int frame_id = 0;
        int start_frame = 0;
        int tracklet_len = 0;
        double last_match_ms = 0;
        int64_t last_detection_sequence = -1;
    };

    struct Match {
        size_t track_index = 0;
        size_t detection_index = 0;
        float cost = 0.0F;
    };

    static float boxIou(const Detection& a, const Detection& b);
    static Detection stateToDetection(
        const std::array<float, 8>& mean,
        int class_id,
        float score
    );
    static std::array<float, 4> detectionToMeasurement(const Detection& detection);

    Track createTrack(const Detection& detection);
    void beginFrame(int64_t frame_index);
    void predictTrack(Track& track) const;
    void updateTrack(Track& track, const Detection& detection);
    void markLost(Track& track);
    void markRemoved(Track& track);

    std::vector<Match> assignDetections(
        const std::vector<size_t>& track_indices,
        const std::vector<Detection>& detections,
        float match_threshold,
        bool fuse_score
    ) const;
    std::vector<TrackedDetection> currentTrackedDetections() const;
    void removeDuplicateTracks();
    void pruneTracks();

    int frame_id_ = 0;
    std::optional<int64_t> last_frame_index_;
    int next_track_id_ = 1;
    int track_buffer_ = 30;
    std::vector<Track> tracks_;
    bool time_mode_ = false, applying_time_ = false;
    std::optional<double> last_timestamp_ms_;
    double current_timestamp_ms_ = 0, dt_seconds_ = 0;
};

}  // namespace yolo

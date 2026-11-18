#include "tracking/byte_tracker.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace yolo;
namespace {
void require(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
Detection moving(double ms, float offset=0) {
    const float x=80+float(ms*.06)+offset;
    return {0,.9F,x,100,x+80,220};
}
TrackerSnapshot train(ByteTracker& tracker) {
    TrackerSnapshot snapshot;
    for (int i=0;i<=20;++i) snapshot=tracker.updateAt({moving(i*100)},i*3,i*100);
    return snapshot;
}
}
int main() {
    try {
        ByteTracker tracker;
        auto source=train(tracker);
        const auto saved=source.states;
        auto predictions=source.predictDetailedAt(2250,500,800,600);
        require(predictions.size()==1,"confirmed track missing during asynchronous inference");
        const auto& p=predictions[0];
        require(p.last_detection_sequence==60 && p.last_detection_ms==2000 &&
                p.prediction_ms==250 && p.detection_age_ms==250 && !p.lost,"incorrect source identity/age");
        require(std::abs(p.track.detection.x1-moving(2250).x1)<3,"actual-time Kalman prediction inaccurate");
        auto repeated=source.predictDetailedAt(2250,500,800,600);
        require(repeated[0].track.detection.x1==p.track.detection.x1 &&
                repeated[0].center_stddev_px==p.center_stddev_px,"duplicate output advanced prediction twice");
        require(source.states[0].mean==saved[0].mean && source.states[0].covariance==saved[0].covariance,
                "display prediction modified inference tracker state");
        auto at_source=source.predictDetailedAt(2000,500,800,600);
        require(p.center_stddev_px[0]>at_source[0].center_stddev_px[0],"prediction did not propagate Kalman covariance");

        // The display path must agree with ByteTracker's own native prediction,
        // including covariance, rather than a separate motion implementation.
        auto missed=tracker.updateAt({},67,2250);
        require(missed.states.size()==1 && missed.states[0].lost,"missed detection did not retain a lost track");
        const auto& native=missed.states[0];
        require(std::abs(native.track.detection.x1-p.track.detection.x1)<.001F &&
            std::abs(std::sqrt(native.covariance[0][0])-p.center_stddev_px[0])<.001F,
            "display and native tracker Kalman transitions differ");
        auto held=missed.predictDetailedAt(2400,500,800,600);
        require(held.size()==1 && held[0].lost && held[0].last_detection_sequence==60 &&
            held[0].detection_age_ms==400,"prediction falsely refreshed the last real detection");
        require(missed.predictDetailedAt(2500.01,500,800,600).empty(),"expired unmatched track was drawn");
        require(source.predictDetailedAt(1999,500,800,600).empty(),"future result contaminated an older image");

        const auto id=p.track.track_id;
        auto corrected=tracker.updateAt({moving(2400,12)},72,2400);
        auto next=corrected.predictDetailedAt(2500,500,800,600);
        auto uncorrected=source.predictDetailedAt(2500,500,800,600);
        require(next.size()==1 && next[0].track.track_id==id && !next[0].lost &&
            next[0].last_detection_sequence==72 && next[0].detection_age_ms==100,
            "RKNN correction failed to preserve ID/renew detection match");
        require(next[0].track.detection.x1>uncorrected[0].track.detection.x1+2,
            "new measurement failed to adjust the predicted trajectory");
        bool old_rejected=false;
        try { tracker.updateAt({moving(2200)},66,2200); } catch (const std::invalid_argument&) { old_rejected=true; }
        require(old_rejected,"late old result rolled tracker time backwards");
        ByteTracker other;
        const auto isolated=other.updateAt({{0,.9F,20,20,80,100}},0,0);
        require(isolated.states[0].track.track_id==1 && isolated.states[0].last_detection_sequence==0,
            "channel tracker shared IDs or watermarks");
        std::cout<<"PASS: native Kalman/covariance equivalence, asynchronous prediction/correction, duplicate frames, actual-time motion, TTL, occlusion, source ownership and independent channels\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

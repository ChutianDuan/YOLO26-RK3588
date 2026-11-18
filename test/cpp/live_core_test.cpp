#include "tracking/byte_tracker.h"
#include "camera/live_media.h"
#include <cmath>
#include <iostream>
#include <thread>
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool threw = false; try { f(); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "invalid time/legacy mix accepted");
}
}
int main() {
    try {
        using namespace yolo;
        ByteTracker tracker;
        TrackerSnapshot snapshot;
        double time = 10000;
        for (int i = 0; i < 40; ++i) {
            time += i % 3 == 0 ? 100 : (i % 3 == 1 ? 27 : 65);
            const float x = 100 + float((time - 10000) * .06);
            snapshot = tracker.updateAt({{0,.9F,x,100,x+80,220}}, i * 3, time);
        }
        const auto id = snapshot.states.at(0).track.track_id;
        const auto before = snapshot.states;
        const auto predicted = snapshot.predictAt(time + 200, 500, 800, 600);
        require(predicted.size() == 1 && predicted[0].track_id == id, "missing predicted track");
        const double expected = 100 + (time + 200 - 10000) * .06;
        require(std::abs(predicted[0].detection.x1 - expected) < 3, "actual-dt constant velocity prediction inaccurate");
        require(snapshot.states[0].mean == before[0].mean, "projection mutated snapshot");
        require(snapshot.predictAt(time - 1, 500, 800, 600).empty(), "future snapshot used on past image");
        require(!snapshot.predictAt(time + 500, 500, 800, 600).empty(), "500 ms boundary hidden early");
        require(snapshot.predictAt(time + 500.01, 500, 800, 600).empty(), "expired prediction displayed");
        rejects([&] { tracker.updateAt({}, 117, time + 1); });
        rejects([&] { tracker.updateAt({}, 999, time); });
        rejects([&] { tracker.updateAt({}, 999, std::nan("")); });
        rejects([&] { tracker.update({}, 999); });
        auto missed = tracker.updateAt({}, 120, time + 100);
        require(missed.predictAt(time + 200, 500, 800, 600).size() == 1, "missed detection immediately hides confirmed track");
        const float x = 100 + float((time + 400 - 10000) * .06);
        auto reacquired = tracker.updateAt({{0,.9F,x,100,x+80,220}}, 123, time + 400);
        require(reacquired.states.at(0).track.track_id == id, "short occlusion changed ID");
        auto expired = tracker.updateAt({}, 126, time + 1501);
        require(expired.states.empty(), "one-second track retention not enforced");
        live::LatestSlot<std::shared_ptr<int>> slot;
        auto lease = std::make_shared<int>(42);
        slot.put(lease); auto in_flight = slot.take();
        for (int i = 0; i < 100; ++i) slot.put(std::make_shared<int>(i));
        require(**in_flight == 42 && slot.dropped() == 99, "latest replacement corrupted in-flight lease");
        slot.finish(); require(!slot.finished(), "EOF reports drained while final frame pending");
        require(**slot.take() == 99 && !slot.take() && slot.finished(), "EOF lost final pending frame");
        live::LatestSlot<int> blocked;
        std::thread consumer([&] { require(!blocked.take(), "close must wake blocked consumer"); });
        blocked.close(); consumer.join();
        std::cout << "PASS real dt, immutable prediction, future/stale rejection, 500ms expiry, occlusion ID, retention, latest-slot leases/EOF/close\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

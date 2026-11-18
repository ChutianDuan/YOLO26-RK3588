#pragma once
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace yolo::multi {
// One replaceable pending item and one exclusive in-flight item per channel.
// The shared cursor advances only on successful dispatch, never on publication.
template<class T> class FairScheduler {
public:
    struct Task { size_t channel; T value; };
    struct Stats {
        uint64_t dispatched = 0, replaced = 0, cancelled = 0;
        bool pending = false, busy = false, eof = false;
    };
    explicit FairScheduler(size_t channels) : slots_(channels) {
        if (!channels || channels > 4) throw std::invalid_argument("scheduler requires 1..4 channels");
    }
    bool put(size_t channel, T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = slots_.at(channel);
        if (cancelled_ || s.stats.eof) { ++s.stats.cancelled; return false; }
        if (s.value) ++s.stats.replaced;
        s.value = std::move(value); s.stats.pending = true;
        ready_.notify_all(); return true;
    }
    std::optional<Task> take() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return cancelled_ || eligible() || allFinished(); });
        if (cancelled_) return {};
        for (size_t n = 0; n < slots_.size(); ++n) {
            const size_t i = (cursor_ + n) % slots_.size();
            auto& s = slots_[i];
            // EOF keeps its last pending frame eligible for draining.
            if (!s.value || s.stats.busy) continue;
            Task task{i, std::move(*s.value)}; s.value.reset();
            s.stats.pending = false; s.stats.busy = true; ++s.stats.dispatched;
            cursor_ = (i + 1) % slots_.size(); return task;
        }
        return {};
    }
    void complete(size_t channel) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& s = slots_.at(channel);
        if (!s.stats.busy) throw std::logic_error("completing an idle channel");
        s.stats.busy = false; ready_.notify_all();
    }
    void finish(size_t channel) {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.at(channel).stats.eof = true; ready_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        for (auto& s : slots_) {
            if (s.value) ++s.stats.cancelled;
            s.value.reset(); s.stats.pending = false; s.stats.eof = true;
        }
        ready_.notify_all();
    }
    bool finished(size_t channel) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& s = slots_.at(channel);
        return s.stats.eof && !s.value && !s.stats.busy;
    }
    std::vector<Stats> stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Stats> result;
        for (const auto& s : slots_) result.push_back(s.stats);
        return result;
    }
private:
    struct Slot { std::optional<T> value; Stats stats; };
    bool eligible() const {
        for (const auto& s : slots_) if (s.value && !s.stats.busy) return true;
        return false;
    }
    bool allFinished() const {
        for (const auto& s : slots_) if (!s.stats.eof || s.value || s.stats.busy) return false;
        return true;
    }
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<Slot> slots_;
    size_t cursor_ = 0;
    bool cancelled_ = false;
};
} // namespace yolo::multi

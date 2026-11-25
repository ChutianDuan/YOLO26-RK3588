#pragma once
#include "camera/live_media.h"
#include <thread>
namespace yolo::live {
// FIFO reference and SPSC comparison share capacity one and preserve EOF drain.
// Only the producer writes an empty slot. Only the consumer clears a full slot.
class PacketChannel {
  public:
    PacketChannel(bool spsc, bool fifo)
        : spsc_(spsc),
          mutex_queue_(
              trace::PacketQueue, [](const EncodedFrame &p) { return p.stamp.frame_id; }, 1, fifo) {}
    void put(EncodedFrame packet) {
        if (!spsc_) {
            mutex_queue_.put(std::move(packet));
            return;
        }
        while (full_.load(std::memory_order_acquire) && !closed_)
            pause();
        if (closed_) {
            ++cancelled_;
            trace::emit(trace::Evict, trace::PacketQueue, packet.stamp.frame_id, 0, 2);
            return;
        }
        trace::emit(trace::Enter, trace::PacketQueue, packet.stamp.frame_id);
        packet_.emplace(std::move(packet));
        full_.store(true, std::memory_order_release);
    }
    std::optional<EncodedFrame> take() {
        if (!spsc_)
            return mutex_queue_.take();
        while (!full_.load(std::memory_order_acquire)) {
            if (closed_ && !full_.load(std::memory_order_acquire))
                return {};
            pause();
        }
        EncodedFrame packet = std::move(*packet_);
        packet_.reset();
        trace::emit(trace::Leave, trace::PacketQueue, packet.stamp.frame_id);
        full_.store(false, std::memory_order_release);
        return packet;
    }
    void close() {
        if (spsc_)
            closed_ = true;
        else
            mutex_queue_.close();
    }
    void finish() {
        if (spsc_)
            closed_ = true;
        else
            mutex_queue_.finish();
    }
    void discardAfterJoin() {
        if (spsc_ && packet_) {
            trace::emit(trace::Evict, trace::PacketQueue, packet_->stamp.frame_id, 0, 2);
            packet_.reset();
            ++cancelled_;
        }
    }
    uint64_t dropped() const { return spsc_ ? 0 : mutex_queue_.dropped(); }
    uint64_t cancelled() const { return spsc_ ? cancelled_.load() : mutex_queue_.cancelled(); }

  private:
    static void pause() { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    bool spsc_;
    LatestSlot<EncodedFrame> mutex_queue_;
    std::optional<EncodedFrame> packet_;
    alignas(64) std::atomic<bool> full_{false};
    alignas(64) std::atomic<bool> closed_{false};
    std::atomic<uint64_t> cancelled_{0};
};
} // namespace yolo::live

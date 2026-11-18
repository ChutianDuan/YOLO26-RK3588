#pragma once
#include <memory>
#include <utility>

namespace yolo {
// 独立于底层 allocation 的“当前帧占用”。生产者用 weak_ptr 观察本对象，
// 消费者复制 owner 共享它；最后一个副本释放后，生产者才能覆盖该帧。
// keep_alive 可以是 DmaBuffer 或池槽；不持有生产者，也不等待设备完成。
class FrameLease {
public:
    explicit FrameLease(std::shared_ptr<void> keep_alive)
        : keep_alive_(std::move(keep_alive)) {}

    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;

private:
    std::shared_ptr<void> keep_alive_;
};
} // namespace yolo

#pragma once
#include <memory>
#include <mpp_buffer.h>

namespace yolo {
class DmaBuffer;

// MPP 对共享 DMA-BUF 的导入适配，与摄像头或文件等视频源类型无关。
// 一个 MPP EXT_DMA 注册及其底层分配。只延长 allocation 存活期，不占用帧槽。
// 所属设备任务必须先完成；本对象不 reset/destroy Context，也不等待设备。
class MppDmaBuffer {
public:
    MppDmaBuffer() = default;
    explicit MppDmaBuffer(std::shared_ptr<DmaBuffer> allocation,
                          const char* operation = "MPP import DMA-BUF");
    ~MppDmaBuffer();

    MppDmaBuffer(const MppDmaBuffer&) = delete;
    MppDmaBuffer& operator=(const MppDmaBuffer&) = delete;
    MppDmaBuffer(MppDmaBuffer&& other) noexcept;
    MppDmaBuffer& operator=(MppDmaBuffer&& other) noexcept;

    MppBuffer get() const noexcept { return buffer_; }
    const std::shared_ptr<DmaBuffer>& allocation() const noexcept { return allocation_; }
    void reset() noexcept;

private:
    std::shared_ptr<DmaBuffer> allocation_;
    MppBuffer buffer_ = nullptr;
};
} // namespace yolo

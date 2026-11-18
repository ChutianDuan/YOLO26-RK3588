#include "image/mpp_dma_buffer.h"
#include "image/dma_buffer.h"
#include <stdexcept>
#include <string>
#include <utility>

namespace yolo {
MppDmaBuffer::MppDmaBuffer(std::shared_ptr<DmaBuffer> allocation, const char* operation)
    : allocation_(std::move(allocation)) {
    MppBufferInfo info{};
    // 固定版本 EXT_DMA 只按 FD 导入；应用 mmap 地址不是 MPP 自己建立的映射。
    info.type = MPP_BUFFER_TYPE_EXT_DMA;
    info.fd = allocation_->fd;
    info.size = allocation_->size;
    const auto rc = mpp_buffer_import(&buffer_, &info);
    if (rc != MPP_OK)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(rc));
}

MppDmaBuffer::~MppDmaBuffer() {
    reset();
}

MppDmaBuffer::MppDmaBuffer(MppDmaBuffer&& other) noexcept
    : allocation_(std::move(other.allocation_)), buffer_(std::exchange(other.buffer_, nullptr)) {}

MppDmaBuffer& MppDmaBuffer::operator=(MppDmaBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        allocation_ = std::move(other.allocation_);
        buffer_ = std::exchange(other.buffer_, nullptr);
    }
    return *this;
}

void MppDmaBuffer::reset() noexcept {
    // 先撤销 MPP 注册，再释放 allocation 引用，不能先关闭 FD。
    if (buffer_) {
        mpp_buffer_put(buffer_);
        buffer_ = nullptr;
    }
    allocation_.reset();
}
} // namespace yolo

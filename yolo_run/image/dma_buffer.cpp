#include "image/dma_buffer.h"
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace yolo {
DmaBuffer::DmaBuffer(size_t bytes, const char* heap_path) : size(bytes) {
    // 1. DMA Heap 只在分配时打开；真正的缓冲 FD 有独立生命周期。
    const int heap = open(heap_path, O_RDWR | O_CLOEXEC);
    if (heap < 0) fail("open DMA heap");

    dma_heap_allocation_data allocation{};
    allocation.len = bytes;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    const int rc = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
    const int saved_errno = errno;
    close(heap);
    if (rc) {
        errno = saved_errno;
        fail("allocate DMA buffer");
    }
    fd = static_cast<int>(allocation.fd);

    // 2. mmap 不复制像素；CPU 通过 data 访问 fd 对应的同一块 DMA-BUF。
    data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        fd = -1;
        fail("mmap DMA buffer");
    }

    // 3. 只建立 RGA 注册，不分配另一张图。构造失败时析构不会运行，须就地清理。
    handle = importbuffer_fd(fd, static_cast<int>(bytes));
    if (!handle) {
        munmap(data, bytes);
        close(fd);
        data = MAP_FAILED;
        fd = -1;
        throw std::runtime_error("RGA importbuffer_fd failed");
    }
}

DmaBuffer::~DmaBuffer() {
    // 上层已确认设备结束访问：先撤销 RGA 导入，再解除映射，最后关闭底层 FD。
    if (handle) releasebuffer_handle(handle);
    if (data != MAP_FAILED) munmap(data, size);
    if (fd >= 0) close(fd);
}

void DmaBuffer::sync(uint64_t flags) {
    dma_buf_sync request{};
    request.flags = flags;
    int rc;
    do {
        rc = ioctl(fd, DMA_BUF_IOCTL_SYNC, &request);
    } while (rc && errno == EINTR);

    if (rc) {
        healthy = false;
        fail("DMA cache sync");
    }
}

[[noreturn]] void DmaBuffer::fail(const char* label) {
    throw std::runtime_error(std::string(label) + ": " + std::strerror(errno));
}

CpuAccess::CpuAccess(std::shared_ptr<DmaBuffer> b, uint64_t a)
    : buffer(std::move(b)), access(a) {
    buffer->sync(DMA_BUF_SYNC_START | access);
}

CpuAccess::~CpuAccess() {
    if (!active) return;
    try {
        buffer->sync(DMA_BUF_SYNC_END | access);
    } catch (const std::exception& e) {
        std::cerr << "DMA sync cleanup error: " << e.what() << '\n';
    }
}

void CpuAccess::finish() {
    buffer->sync(DMA_BUF_SYNC_END | access);
    active = false;
}
} // namespace yolo

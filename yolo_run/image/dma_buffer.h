#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <linux/dma-buf.h>
#include <sys/mman.h>
#include <im2d.hpp>

namespace yolo {
// 对借用的 DMA-BUF FD 执行一次缓存同步；可重试的中断在内部处理，失败保留 errno。
int syncDmaFd(int fd, uint64_t flags) noexcept;

// 一次分配的唯一资源所有者：DMA-BUF FD、CPU 映射、RGA 导入 Handle 一起管理。
// 析构前由上层保证设备已经结束访问；共享所有权只延长存活期，不负责等待设备。
class DmaBuffer {
public:
    explicit DmaBuffer(size_t bytes, const char* heap_path = "/dev/dma_heap/system");
    ~DmaBuffer();

    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;

    // START/END 仅定义 CPU 访问区间；失败后 healthy=false，调用方不得继续复用。
    void sync(uint64_t flags);

    // 描述同一块分配，调用方只借用这些值，不自行 close/munmap/release。
    size_t size;                      // 分配容量，单位为字节。
    int fd = -1;                      // 供 V4L2、MPP、RKNN 等设备导入。
    void* data = MAP_FAILED;           // mmap 的 CPU 地址，不是像素副本。
    rga_buffer_handle_t handle = 0;    // RGA 持久导入，不是当前帧占用标记。
    bool healthy = true;

private:
    [[noreturn]] static void fail(const char* label);
};

// 成对管理 CPU 缓存同步；它不代替 RGA IM_SYNC 或 RKNN 的执行完成等待。
// 正常路径显式 finish() 传播 END 错误，析构仅负责异常路径的兜底收尾。
// 同一访问区间只有一个守卫；需要共享读租约时共享其 shared_ptr，不复制守卫。
struct CpuAccess {
    CpuAccess(std::shared_ptr<DmaBuffer> buffer, uint64_t access);
    ~CpuAccess();

    CpuAccess(const CpuAccess&) = delete;
    CpuAccess& operator=(const CpuAccess&) = delete;

    void finish();

    std::shared_ptr<DmaBuffer> buffer;
    uint64_t access;
    bool active = true;
};
} // namespace yolo

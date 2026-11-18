#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace yolo {
// 记录“原图 → 模型输入”的实际缩放和填充，后处理用它恢复原图坐标。
// 缩放尺寸经过整数取整，因此 X/Y 比例分别保存，不能只保存理论 ratio。
struct LetterBoxInfo {
    float scale_x = 1;
    float scale_y = 1;
    float pad_w = 0;
    float pad_h = 0;
};
// engine 查询模型后交给预处理器的分配要求；逻辑图像尺寸仍由 AppConfig 给出。
struct InputMemoryLayout {
    int width_stride = 0;   // 单位是像素；0 表示使用图像可见宽度，不能与 Mat::step 的字节数混用。
    int height_stride = 0;
    size_t min_capacity = 0; // 单位是字节；模型报告的容量可能大于可见 RGB UINT8 像素大小。
};
// 描述同一块 DMA-BUF：fd 给设备导入，data 是 mmap 得到的 CPU 地址，不是像素副本。
// 当前契约：offset=0、未缓存映射、RGB UINT8/NHWC；CPU 访问仍需 DMA_BUF_IOCTL_SYNC。
// allocation 保持 FD/mmap/RGA Handle 存活，但不表示“当前这一帧仍被占用”。
struct DmaTensorInput {
    std::shared_ptr<void> allocation;
    int fd = -1;
    void* data = nullptr;
    size_t capacity = 0;
    int width = 0, height = 0;
    int width_stride = 0, height_stride = 0;
};
// 单帧交接对象：模型输入只接受 RGA 输出的 RGB DMA-BUF。
// owner 是独立帧租约：所有 TensorInput 副本释放前，预处理器禁止覆盖该帧。
// 与 allocation 分离后，RKNN 可长期缓存注册，同时允许下一帧复用同一块内存。
struct TensorInput {
    std::shared_ptr<void> owner;
    DmaTensorInput dma;
    int image_width = 0;
    int image_height = 0;
    LetterBoxInfo letterbox;
};
struct Detection {
    int class_id = -1;
    float score = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};
struct TrackedDetection {
    int track_id = -1;
    Detection detection;
};
struct InferResult {
    int64_t frame_index = -1;
    std::vector<Detection> detections;
    double preprocess_ms = 0;
    double infer_ms = 0;
    double postprocess_ms = 0;
    double queue_wait_ms = 0;
};
}  // namespace yolo

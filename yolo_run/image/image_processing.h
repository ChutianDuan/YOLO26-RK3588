#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include "config/app_config.h"
#include "model/inference_types.h"

namespace yolo {
class DmaBuffer;
// owner 是采集帧租约，dma 是分配所有权；只接受 DMA-BUF 源。
// 当前仅接受逐行 Y0/U/Y1/V、BT.601 limited、非交错图像。
struct YuyvFrameView {
    std::shared_ptr<void> owner;
    std::shared_ptr<DmaBuffer> dma;
    const unsigned char* data = nullptr;
    size_t capacity = 0;
    int width = 0, height = 0, bytes_per_line = 0;
};
struct Nv12FrameView {
    std::shared_ptr<void> owner;
    std::shared_ptr<DmaBuffer> dma;
    int width = 0, height = 0, width_stride = 0, height_stride = 0;
    bool full_range = true; // MJPEG/JFIF BT.601 full range, validated by decoder
};
// Empty ROI means the full visible image; NV12 crop coordinates/sizes must be even.
cv::Rect nv12SourceRegion(const Nv12FrameView& frame, const cv::Rect& roi = {});
struct ResizeGeometry {
    int width = 0, height = 0, left = 0, top = 0;
    LetterBoxInfo transform;
};
ResizeGeometry resizeGeometry(int width, int height, const AppConfig& config);

// 同一实例只由一个调用线程使用；内部复用源、目标缓冲。
// 推理时传入 engine.inputMemoryLayout()，并让返回的 TensorInput 存活到推理结束。
class ImagePreprocessor {
public:
    explicit ImagePreprocessor(const AppConfig& config, InputMemoryLayout layout = {});
    ~ImagePreprocessor();
    TensorInput prepare(const cv::Mat& bgr);
    // Before starting Workers, reserve the largest BGR upload required by all sources.
    // No active TensorInput may be leased; subsequent smaller images reuse this allocation.
    void reserveBgrSourceCapacity(size_t bytes);
    TensorInput prepareYuyv(const YuyvFrameView& frame);
    TensorInput prepareNv12(const Nv12FrameView& frame, const cv::Rect& source_roi = {});
    ImagePreprocessor(const ImagePreprocessor&) = delete;
    ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct TrackingImage {
    // 持有 DMA READ 租约；gray 及其金字塔视图销毁前不能释放 owner。
    std::shared_ptr<void> owner;
    cv::Mat gray;
};

// 同一实例只由一个调用线程使用；固定双槽保存 LK 前后帧，不覆盖仍被租用的图像。
class TrackingPreprocessor {
public:
    explicit TrackingPreprocessor(const AppConfig& config);
    ~TrackingPreprocessor();
    TrackingImage prepare(const cv::Mat& bgr);
    TrackingPreprocessor(const TrackingPreprocessor&) = delete;
    TrackingPreprocessor& operator=(const TrackingPreprocessor&) = delete;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace yolo

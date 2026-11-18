#include "image/image_processing.h"
#include "image/dma_buffer.h"
#include "image/frame_lease.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <im2d.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {
namespace {
void checkRga(IM_STATUS status, const char* operation) {
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR)
        throw std::runtime_error(std::string(operation) + ": " + imStrError(status));
}

// Mat 可以是 ROI；按真实行地址上传到对齐后的 DMA 行，并清零每行尾部。
// 检测和 LK 共用此步骤；它是明确存在的 CPU 像素复制。
void uploadBgr(const cv::Mat& bgr, const std::shared_ptr<DmaBuffer>& destination, int stride) {
    CpuAccess write(destination, DMA_BUF_SYNC_WRITE);
    auto* rows = static_cast<unsigned char*>(destination->data);
    for (int y = 0; y < bgr.rows; ++y) {
        auto* row = rows + size_t(y) * stride * 3;
        std::memcpy(row, bgr.ptr(y), size_t(bgr.cols) * 3);
        std::memset(row + bgr.cols * 3, 0, size_t(stride - bgr.cols) * 3);
    }
    write.finish(); // WRITE_END 成功后才允许 RGA 读取。
}
}

cv::Rect nv12SourceRegion(const Nv12FrameView& f, const cv::Rect& requested) {
    if (!f.owner || !f.dma || !f.dma->healthy ||
        f.width < 68 || f.height < 2 || f.width > 16384 || f.height > 16384 ||
        f.width % 2 || f.height % 2 || f.width_stride < f.width || f.height_stride < f.height ||
        f.width_stride > 16384 || f.height_stride > 16384 || f.width_stride % 2 || f.height_stride % 2 ||
        f.dma->size < size_t(f.width_stride) * f.height_stride * 3 / 2)
        throw std::runtime_error("invalid NV12 layout, allocation or lease");
    const auto roi=requested==cv::Rect{} ? cv::Rect(0,0,f.width,f.height) : requested;
    if (roi.x<0 || roi.y<0 || roi.width<2 || roi.height<2 ||
        (roi.x|roi.y|roi.width|roi.height)&1 || roi.x>f.width || roi.y>f.height ||
        roi.width>f.width-roi.x || roi.height>f.height-roi.y)
        throw std::runtime_error("invalid NV12 crop bounds or alignment");
    return roi;
}

ResizeGeometry resizeGeometry(int width, int height, const AppConfig& c) {
    if (width <= 0 || height <= 0) throw std::runtime_error("empty input image");
    ResizeGeometry g;
    g.width = c.input_width;
    g.height = c.input_height;
    if (c.use_letterbox) {
        // 保持比例缩放，把有效图像居中放入模型画布；其余位置填 114。
        const double ratio = std::min(double(c.input_width) / width, double(c.input_height) / height);
        g.width = std::clamp(static_cast<int>(std::round(width * ratio)), 1, c.input_width);
        g.height = std::clamp(static_cast<int>(std::round(height * ratio)), 1, c.input_height);
        g.left = (c.input_width - g.width) / 2;
        g.top = (c.input_height - g.height) / 2;
    }
    // 记录取整后的真实比例，供模型坐标恢复；非 letterbox 时也支持独立 X/Y 缩放。
    g.transform.scale_x = float(g.width) / width;
    g.transform.scale_y = float(g.height) / height;
    g.transform.pad_w = float(g.left);
    g.transform.pad_h = float(g.top);
    return g;
}

class ImagePreprocessor::Impl {
public:
    Impl(const AppConfig& c, InputMemoryLayout l) : config(c), layout(l) {
        if (layout.width_stride == 0) layout.width_stride = c.input_width;
        if (layout.height_stride == 0) layout.height_stride = c.input_height;
        if (c.input_width <= 0 || c.input_height <= 0 ||
            layout.width_stride < c.input_width || layout.height_stride < c.input_height ||
            layout.width_stride > 16384 || layout.height_stride > 16384)
            throw std::runtime_error("invalid RGB DMA stride/dimensions");
        // RGB 每像素 3 字节，同时保留 engine 根据模型查询得到的额外容量要求。
        const size_t rgb_bytes = size_t(layout.width_stride) * layout.height_stride * 3;
        layout.min_capacity = std::max(layout.min_capacity, rgb_bytes);
        if (layout.min_capacity > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("RGB DMA capacity exceeds RGA import limit");
    }
    TensorInput prepare(const cv::Mat& bgr) {
        // cv::Mat 是 CPU 侧 BGR 图像；它可以是 ROI，不能假设各行紧密连续。
        if (bgr.empty() || bgr.type() != CV_8UC3) throw std::runtime_error("preprocess requires nonempty BGR UINT8 image");
        if (bgr.cols > 16384 || bgr.rows > 16384) throw std::runtime_error("input dimensions exceed 16384");
        const auto g = resizeGeometry(bgr.cols, bgr.rows, config);
        requireAvailableInput();
        // RGA 的行 stride 以像素计，分配容量和 memcpy 偏移再乘 RGB/BGR 的 3 字节。
        const int stride = (bgr.cols + 15) & ~15;
        const size_t source_bytes = size_t(stride) * bgr.rows * 3;
        if (!source || source->size < source_bytes) source = std::make_shared<DmaBuffer>(source_bytes);
        ensureRgbDestination("DMA buffer has an earlier cache sync failure");
        if (!source->healthy) throw std::runtime_error("DMA buffer has an earlier cache sync failure");
        uploadBgr(bgr, source, stride);
        // wrapbuffer_handle 只构造图像视图；可见尺寸与内存 stride 分别传入，不复制像素。
        const auto src = wrapbuffer_handle(source->handle, bgr.cols, bgr.rows, RK_FORMAT_BGR_888, stride, bgr.rows);
        const auto dst = wrapbuffer_handle(destination->handle, config.input_width, config.input_height,
                                           RK_FORMAT_RGB_888, layout.width_stride, layout.height_stride);
        initializePadding(g);
        const im_rect sr{0, 0, bgr.cols, bgr.rows};
        const im_rect dr{g.left, g.top, g.width, g.height};
        checkRga(imcheck(src, dst, sr, dr, 0), "RGA layout check");
        im_opt_t options{};
        // 按任务限定到 RGA3，避免分配到受地址/格式限制的 RGA2；不修改全局调度配置。
        options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
        checkRga(improcess(src, dst, {}, sr, dr, {}, -1, nullptr, &options, IM_SYNC), "RGA resize/BGR to RGB");
        // IM_SYNC 返回后 RGA 已完成写入，再把同一个目标 FD 交给 RKNN。
        return makeDmaInput(bgr.cols, bgr.rows, g);
    }
    void reserveBgrSourceCapacity(size_t bytes) {
        if (!bytes || bytes > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("invalid BGR DMA reservation");
        requireAvailableInput();
        if (!source || source->size < bytes) source = std::make_shared<DmaBuffer>(bytes);
        ensureRgbDestination();
    }
    TensorInput prepareYuyv(const YuyvFrameView& f) {
        if (!f.owner || !f.dma || !f.data || f.width < 68 || f.width > 16384 || f.height < 2 || f.height > 16384 ||
            f.width % 2 || f.height % 2 || f.bytes_per_line < f.width * 2 || f.bytes_per_line % 2 ||
            f.bytes_per_line / 2 > 16384 || f.capacity < size_t(f.bytes_per_line) * f.height)
            throw std::runtime_error("invalid YUYV frame, capacity, stride or lease");
        if (f.data != f.dma->data || f.capacity > f.dma->size || !f.dma->healthy)
            throw std::runtime_error("YUYV DMA descriptor does not match its allocation");
        requireAvailableInput();
        const auto g = resizeGeometry(f.width, f.height, config);
        const int stride = f.bytes_per_line / 2;
        ensureRgbDestination();
        initializePadding(g);
        auto src = wrapbuffer_handle(f.dma->handle, f.width, f.height, RK_FORMAT_YUYV_422, stride, f.height);
        auto dst = wrapbuffer_handle(destination->handle, config.input_width, config.input_height,
                                     RK_FORMAT_RGB_888, layout.width_stride, layout.height_stride);
        dst.color_space_mode = IM_YUV_TO_RGB_BT601_LIMIT;
        const im_rect sr{0, 0, f.width, f.height}, dr{g.left, g.top, g.width, g.height};
        checkRga(imcheck(src, dst, sr, dr, 0), "RGA YUYV layout check");
        im_opt_t options{};
        options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
        const auto status = improcess(src, dst, {}, sr, dr, {}, -1, nullptr, &options, IM_SYNC);
        if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
            // 无法确认硬件是否仍访问源帧时，不展开栈并提前 QBUF/释放内存。
            std::cerr << "fatal: RGA YUYV execution failed: " << imStrError(status)
                      << "; terminating before DMA memory release\n";
            std::terminate();
        }
        return makeDmaInput(f.width, f.height, g);
    }
    TensorInput prepareNv12(const Nv12FrameView& f, const cv::Rect& source_roi) {
        const auto roi=nv12SourceRegion(f,source_roi);
        requireAvailableInput();
        const auto g = resizeGeometry(f.width, f.height, config);
        ensureRgbDestination();
        initializePadding(g);
        auto src = wrapbuffer_handle(f.dma->handle, f.width, f.height, RK_FORMAT_YCbCr_420_SP,
                                     f.width_stride, f.height_stride);
        auto dst = wrapbuffer_handle(destination->handle, config.input_width, config.input_height,
                                     RK_FORMAT_RGB_888, layout.width_stride, layout.height_stride);
        dst.color_space_mode = f.full_range ? IM_YUV_TO_RGB_BT601_FULL : IM_YUV_TO_RGB_BT601_LIMIT;
        // Model coordinates describe the virtual full-sized canvas after cropping/zooming.
        const im_rect sr{roi.x, roi.y, roi.width, roi.height}, dr{g.left, g.top, g.width, g.height};
        checkRga(imcheck(src, dst, sr, dr, 0), "RGA NV12 layout check");
        im_opt_t options{}; options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
        const auto rc = improcess(src, dst, {}, sr, dr, {}, -1, nullptr, &options, IM_SYNC);
        if (rc != IM_STATUS_SUCCESS && rc != IM_STATUS_NOERROR) {
            std::cerr << "fatal: RGA NV12 completion uncertain: " << imStrError(rc) << '\n';
            std::terminate();
        }
        return makeDmaInput(f.width, f.height, g);
    }

private:
    void requireAvailableInput() const {
        // RKNN 注册缓存也持有 allocation，不能用 destination.use_count() 判断帧占用。
        if (!frame_lease.expired()) throw std::runtime_error("previous DMA input is still leased");
    }

    void ensureRgbDestination(const char* sync_error = "RGB destination sync failure") {
        // 本机 cached 目标的 RKNN FD 路径曾不稳定，检测目标继续使用已验证的 uncached heap。
        // 按模型容量首次分配后复用；CPU 上传源和 LK 缓冲仍使用默认 cached heap。
        if (!destination)
            destination = std::make_shared<DmaBuffer>(layout.min_capacity, "/dev/dma_heap/system-uncached");
        if (!destination->healthy) throw std::runtime_error(sync_error);
    }

    void initializePadding(const ResizeGeometry& geometry) {
        const cv::Rect region(geometry.left, geometry.top, geometry.width, geometry.height);
        if (region == initialized_region) return;

        // RK3588 color fill 受 RGA2/4 GiB 地址限制，由 CPU 初始化 114。
        // 仅首次或有效区域变化时初始化；之后 RGA 覆盖有效区域，补边可复用。
        CpuAccess write(destination, DMA_BUF_SYNC_WRITE);
        std::memset(destination->data, 114, destination->size);
        write.finish();
        initialized_region = region;
    }

    // 只在同步 RGA 成功后发布：不打开 CPU READ 区间，也不复制 RGB 像素。
    TensorInput makeDmaInput(int image_width, int image_height, const ResizeGeometry& geometry) {
        TensorInput input;
        input.image_width = image_width;
        input.image_height = image_height;
        input.letterbox = geometry.transform;
        input.owner = std::make_shared<FrameLease>(destination);
        frame_lease = input.owner;
        input.dma = DmaTensorInput{destination, destination->fd, destination->data, destination->size,
                                  config.input_width, config.input_height,
                                  layout.width_stride, layout.height_stride};
        return input;
    }

    AppConfig config;
    InputMemoryLayout layout;
    std::weak_ptr<void> frame_lease; // 只观察是否仍有帧使用者，不主动延长当前帧占用。
    std::shared_ptr<DmaBuffer> source;      // CPU 上传暂存，直接 YUYV/NV12 路径不使用。
    std::shared_ptr<DmaBuffer> destination; // 当前 Worker 独占的模型 RGB 输入。
    cv::Rect initialized_region;
};
ImagePreprocessor::ImagePreprocessor(const AppConfig& c, InputMemoryLayout l) : impl_(std::make_unique<Impl>(c, l)) {}
ImagePreprocessor::~ImagePreprocessor() = default;
TensorInput ImagePreprocessor::prepare(const cv::Mat& bgr) { return impl_->prepare(bgr); }
void ImagePreprocessor::reserveBgrSourceCapacity(size_t bytes) { impl_->reserveBgrSourceCapacity(bytes); }
TensorInput ImagePreprocessor::prepareYuyv(const YuyvFrameView& f) { return impl_->prepareYuyv(f); }
TensorInput ImagePreprocessor::prepareNv12(const Nv12FrameView& f, const cv::Rect& roi) { return impl_->prepareNv12(f,roi); }

class TrackingPreprocessor::Impl {
public:
    explicit Impl(const AppConfig& c) : config(c) {}
    TrackingImage prepare(const cv::Mat& bgr) {
        if (bgr.empty() || bgr.type() != CV_8UC3 || bgr.cols > 16384 || bgr.rows > 16384)
            throw std::runtime_error("LK preprocess requires nonempty BGR UINT8 image <=16384");
        // NV12 要求可见宽高为偶数；跟踪坐标使用实际取整后的 X/Y 缩放比例。
        const int width = std::max(2, cvRound(bgr.cols * config.lk_scale / 2) * 2);
        const int height = std::max(2, cvRound(bgr.rows * config.lk_scale / 2) * 2);
        // LK 同时需要前后两帧；只选择没有外部读租约的槽，不扩展成无限帧池。
        std::shared_ptr<DmaBuffer>* selected = nullptr;
        for (auto& slot : output) if (!slot || slot.use_count() == 1) { selected = &slot; break; }
        if (!selected) throw std::runtime_error("both LK DMA frames are still leased");
        const int source_stride = (bgr.cols + 15) & ~15;
        const int stride = (width + 15) & ~15;
        const size_t source_bytes = size_t(source_stride) * bgr.rows * 3;
        const size_t bytes = size_t(stride) * height * 3 / 2;
        if (!source || source->size != source_bytes) source = std::make_shared<DmaBuffer>(source_bytes);
        auto& destination = *selected;
        if (!destination || destination->size != bytes) destination = std::make_shared<DmaBuffer>(bytes);
        if (!source->healthy || !destination->healthy) throw std::runtime_error("LK DMA cache sync previously failed");
        uploadBgr(bgr, source, source_stride);
        auto src = wrapbuffer_handle(source->handle, bgr.cols, bgr.rows, RK_FORMAT_BGR_888, source_stride, bgr.rows);
        // RGA3 支持 NV12 和 40 位寻址；选其 Y 平面作灰度，避免 RGA2 专有 Y400 的地址限制。
        auto dst = wrapbuffer_handle(destination->handle, width, height, RK_FORMAT_YCbCr_420_SP, stride, height);
        // 使用 BT.601 full range，匹配 OpenCV 灰度的 0..255 范围。
        imsetColorSpace(&src, IM_RGB_FULL_RANGE);
        imsetColorSpace(&dst, IM_YUV_BT601_FULL_RANGE);
        checkRga(imcheck(src, dst, {}, {}, 0), "LK RGA layout check");
        im_opt_t options{};
        options.core = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
        checkRga(improcess(src, dst, {}, {}, {}, {}, -1, nullptr, &options, IM_SYNC), "LK RGA resize/grayscale");
        auto read_lease = std::make_shared<CpuAccess>(destination, DMA_BUF_SYNC_READ);
        // gray 只是 DMA-BUF 的带 stride 的 Y 平面视图，没有再次复制灰度像素。
        // 与检测目标不同，LK 在 CPU 上读取 gray，READ 租约必须覆盖 gray/金字塔使用期。
        return {std::move(read_lease), cv::Mat(height, width, CV_8UC1, destination->data, stride)};
    }
    AppConfig config;
    std::shared_ptr<DmaBuffer> source;
    std::array<std::shared_ptr<DmaBuffer>, 2> output;
};
TrackingPreprocessor::TrackingPreprocessor(const AppConfig& c) : impl_(std::make_unique<Impl>(c)) {}
TrackingPreprocessor::~TrackingPreprocessor() = default;
TrackingImage TrackingPreprocessor::prepare(const cv::Mat& bgr) { return impl_->prepare(bgr); }
}  // namespace yolo

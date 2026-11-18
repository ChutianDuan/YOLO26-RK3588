#include "model/yolo_engine.h"
#include "model/yolo_postprocess.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <rknn_api.h>

namespace yolo {
namespace {
void check(int code, const char* operation) {
    if (code != RKNN_SUCC) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}
double elapsed(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
struct Context {
    rknn_context value = 0;
    void reset() noexcept {
        if (!value) return;
        if (rknn_destroy(value) != RKNN_SUCC) {
            std::cerr << "fatal: RKNN context cleanup failed; retaining device memory until process exit\n";
            std::terminate();
        }
        value = 0;
    }
    ~Context() { reset(); }
};
// RKNN 的外部内存包装：只管理注册，底层 FD/mmap 的所有权仍由 buffer.allocation 保持。
// 本对象属于创建它的 context，必须在 context 销毁前撤销注册。
struct ImportedInput {
    rknn_context context;
    DmaTensorInput buffer;
    rknn_tensor_mem* memory = nullptr;
    ImportedInput(rknn_context ctx, const DmaTensorInput& b) : context(ctx), buffer(b) {
        // 导入 RGA 刚写完的同一块 DMA-BUF，offset=0；不是重新分配或复制一张输入图。
        memory = rknn_create_mem_from_fd(ctx, b.fd, b.data, static_cast<uint32_t>(b.capacity), 0);
        if (!memory) throw std::runtime_error("rknn_create_mem_from_fd failed");
    }
    ImportedInput(const ImportedInput&) = delete;
    ImportedInput& operator=(const ImportedInput&) = delete;
    ~ImportedInput() {
        if (rknn_destroy_mem(context, memory) != RKNN_SUCC) {
            std::cerr << "fatal: RKNN imported memory cleanup failed\n";
            std::terminate();
        }
    }
    bool matches(const DmaTensorInput& b) const {
        // FD 数字可能被复用；同时比较 shared_ptr 所有者身份、地址和布局，不能只按 fd 缓存。
        return !buffer.allocation.owner_before(b.allocation) && !b.allocation.owner_before(buffer.allocation) &&
            buffer.fd == b.fd && buffer.data == b.data && buffer.capacity == b.capacity &&
            buffer.width == b.width && buffer.height == b.height &&
            buffer.width_stride == b.width_stride && buffer.height_stride == b.height_stride;
    }
};
// 配对 outputs_get()/outputs_release()；即使后处理抛异常，也要归还本次输出。
struct OutputLease {
    rknn_context context;
    std::vector<rknn_output>& outputs;
    bool active = true;
    void finish() {
        const int rc = rknn_outputs_release(context, static_cast<uint32_t>(outputs.size()), outputs.data());
        active = false;
        check(rc, "rknn_outputs_release");
    }
    ~OutputLease() {
        if (!active) return;
        const int rc = rknn_outputs_release(context, static_cast<uint32_t>(outputs.size()), outputs.data());
        if (rc != RKNN_SUCC) std::cerr << "rknn_outputs_release failed: " << rc << '\n';
    }
};
}
class YoloEngine::Impl {
public:
    explicit Impl(const AppConfig& c, int core_mask, bool collect_perf) : config(c), collect_performance(collect_perf) {
        validateConfig(c);
        std::ifstream file(c.model_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open model: " + c.model_path);
        const auto size = file.tellg();
        if (size <= 0 || size > 512 * 1024 * 1024) throw std::runtime_error("invalid RKNN model size");
        std::vector<char> model(static_cast<size_t>(size));
        file.seekg(0);
        if (!file.read(model.data(), size)) throw std::runtime_error("cannot read complete model");
        // 模型只初始化一次；默认 flags=0，仅诊断 context 显式开启性能采集，保留默认缓存同步。
        const uint32_t flags = collect_performance ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
        check(rknn_init(&context.value, model.data(), static_cast<uint32_t>(model.size()), flags, nullptr), "rknn_init");
        if (core_mask != -1)
            check(rknn_set_core_mask(context.value, static_cast<rknn_core_mask>(core_mask)), "rknn_set_core_mask");
        rknn_sdk_version version{};
        check(rknn_query(context.value, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)), "RKNN version query");
        std::cerr << "RKNN runtime " << version.api_version << ", driver " << version.drv_version << '\n';
        rknn_input_output_num io{};
        check(rknn_query(context.value, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "RKNN IO count");
        if (io.n_input != 1 || io.n_output != 2) throw std::runtime_error("expected 1 input and 2 YOLO26 outputs");
        rknn_tensor_attr input{};
        check(rknn_query(context.value, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "RKNN input attribute");
        attrs.resize(2);
        for (size_t i = 0; i < attrs.size(); ++i) {
            attrs[i].index = static_cast<uint32_t>(i);
            check(rknn_query(context.value, RKNN_QUERY_OUTPUT_ATTR, &attrs[i], sizeof(attrs[i])), "RKNN output attribute");
        }
        // 当前后处理只接受约定的 YOLO26 双输出，不能直接套用其他 YOLO 版本的 tensor。
        validateModelLayout(input, attrs, c);
        check(rknn_query(context.value, RKNN_QUERY_NATIVE_INPUT_ATTR, &input_attr, sizeof(input_attr)), "RKNN native input attribute");
        validateModelLayout(input_attr, attrs, c);
        // 逻辑属性用于验证模型契约，native 属性用于取得实际内存 stride/容量。
        const auto stride = input_attr.w_stride ? input_attr.w_stride : static_cast<uint32_t>(c.input_width);
        if (stride < static_cast<uint32_t>(c.input_width) || stride > 16384)
            throw std::runtime_error("unsupported RKNN input width stride");
        layout.width_stride = static_cast<int>(stride);
        layout.height_stride = c.input_height;
        // 容量必须同时满足 RGB UINT8 布局和模型报告要求；不能因应用提供 UINT8 就缩小 native 容量。
        layout.min_capacity = std::max({size_t(stride) * c.input_height * 3,
                                       size_t(input_attr.size_with_stride), size_t(input_attr.size)});
        if (layout.min_capacity > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("RKNN input allocation exceeds RGA import limit");
        // 应用提供 RGB UINT8/NHWC，pass_through=0 让 Runtime 按模型处理归一化/量化。
        // 不在 CPU 上提前除以 255，也不把 UINT8 直接冒充模型的 FP16/INT8 native 数据。
        input_attr.type = RKNN_TENSOR_UINT8;
        input_attr.fmt = RKNN_TENSOR_NHWC;
        input_attr.pass_through = 0;
        input_attr.h_stride = 0;  // 当前协商的高度 stride 等于模型高度，API 中 0 表示默认高度。
        std::cerr << "input RGB UINT8 NHWC: stride=" << stride << "x" << c.input_height
                  << ", DMA capacity=" << layout.min_capacity << ", pass_through=0\n";
        // 两个 FP32 输出缓冲一次分配、逐帧复用；want_float=1 由 Runtime 处理 FP16/INT8 转换。
        buffers.resize(2);
        outputs.resize(2);
        for (size_t i = 0; i < attrs.size(); ++i) {
            buffers[i].resize(attrs[i].n_elems);
            outputs[i] = {};
            outputs[i].index = static_cast<uint32_t>(i);
            outputs[i].want_float = 1;
            outputs[i].is_prealloc = 1;
            outputs[i].buf = buffers[i].data();
            outputs[i].size = static_cast<uint32_t>(buffers[i].size() * sizeof(float));
            if (attrs[i].dims[1] == 4 &&
                (c.num_classes != 4 || std::strncmp(attrs[i].name, "boxes", sizeof(attrs[i].name)) == 0)) boxes_index = i;
            else scores_index = i;
            std::cerr << "output " << i << ": [1," << attrs[i].dims[1] << "," << attrs[i].dims[2]
                      << "] " << get_type_string(attrs[i].type) << " zp=" << attrs[i].zp << " scale=" << attrs[i].scale << '\n';
        }
    }
    ~Impl() {
        // 执行失败后不能假设设备已经停止；先确认完成，期间保留 active_input 的帧租约。
        drain();
        // 临时再持有 allocation，保证释放顺序为：RKNN 注册 → context → 底层 FD/mmap/RGA Handle。
        auto allocation = imported ? imported->buffer.allocation : std::shared_ptr<void>{};
        imported.reset();
        context.reset();
        active_input = {};
    }
    void drain() noexcept {
        if (!device_pending) return;
        rknn_run_extend wait{};
        wait.timeout_ms = config.inference_timeout_ms;
        wait.fence_fd = -1;
        // 等待失败时直接终止进程，不继续析构并提前释放仍可能被硬件访问的内存。
        if (rknn_wait(context.value, &wait) != RKNN_SUCC) {
            std::cerr << "fatal: RKNN completion could not be confirmed; terminating before DMA memory release\n";
            std::terminate();
        }
        device_pending = false;
    }
    void validateInput(const TensorInput& input) const {
        if (!input.owner) throw std::runtime_error("missing DMA input frame lease");
        const auto& b = input.dma;
        // 分配和布局不变时复用注册；稳定描述只在首次导入或更换输入时校验。
        if (imported && imported->matches(b)) return;
        if (!b.allocation || !b.data || b.data == MAP_FAILED || b.fd < 0)
            throw std::runtime_error("invalid DMA input ownership/address/FD");
        if (b.width != config.input_width || b.height != config.input_height ||
            b.width_stride != layout.width_stride || b.height_stride != layout.height_stride)
            throw std::runtime_error("DMA RGB input dimensions/stride do not match RKNN");
        if (b.capacity < layout.min_capacity || b.capacity > size_t(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error("DMA input capacity does not satisfy RKNN");
        struct stat status{};
        if (fstat(b.fd, &status) != 0 || status.st_size <= 0 || uint64_t(status.st_size) < b.capacity)
            throw std::runtime_error("DMA input FD is invalid or smaller than the declared capacity");
    }
    void bindDmaInput(const DmaTensorInput& buffer) {
        if (!imported || !imported->matches(buffer)) {
            // 上一任务已结束：先撤旧注册再导入，同时保留旧 allocation 至绑定成功。
            // 重复导入可能复用底层句柄（包括换 owner 或 dup FD），不能先新建再撤旧。
            transition_allocation = imported ? imported->buffer.allocation : std::shared_ptr<void>{};
            imported.reset();
            imported = std::make_unique<ImportedInput>(context.value, buffer);
        }
        // 输入始终为 DMA-BUF；沿用 Runtime 的逐帧 IO 绑定。
        check(rknn_set_io_mem(context.value, imported->memory, &input_attr), "rknn_set_io_mem input");
        transition_allocation.reset();
    }

    InferResult infer(const TensorInput& input, int64_t frame_index, NpuPerformance* performance) {
        // 锁覆盖绑定、执行、输出及解码，防止共享 context 和复用输出缓冲被并发改写。
        std::lock_guard<std::mutex> lock(mutex);
        if (failed) throw std::runtime_error("RKNN engine is unusable after an execution/binding failure");
        validateInput(input); // 非法元数据在硬件调用前拒绝，不污染正常 engine 状态。
        const auto start = std::chrono::steady_clock::now();
        active_input = input; // 复制描述和所有权，不复制像素；异常后也能继续持有当前帧。
        try {
            // 1. 绑定输入不表示执行完成；从提交前开始保留设备在途状态。
            bindDmaInput(input.dma);
            rknn_run_extend run{};
            run.timeout_ms = config.inference_timeout_ms;
            run.fence_fd = -1;
            // RGA 已用 IM_SYNC 完成；这里采用阻塞执行。视频异步来自外层 Worker，并非此处 non_block。
            device_pending = true;
            check(rknn_run(context.value, &run), "rknn_run");
            check(rknn_outputs_get(context.value, 2, outputs.data(), nullptr), "rknn_outputs_get");
            device_pending = false;

            // 2. 设备已完成，但输出包装和当前输入租约继续覆盖后处理。
            OutputLease lease{context.value, outputs};
            InferResult result;
            result.frame_index = frame_index;
            result.infer_ms = elapsed(start);
            const auto post = std::chrono::steady_clock::now();
            // 输入模型的 boxes/scores 已由 Runtime 转为 FP32，后处理只做筛选和坐标恢复。
            result.detections = decodeYolo26(buffers[boxes_index].data(), buffers[scores_index].data(),
                                           attrs[boxes_index].dims[2], input, config);
            result.postprocess_ms = elapsed(post);
            NpuPerformance collected;
            if (performance) {
                // 两种性能查询均要求 outputs_get 成功；查询与字符串复制不计入上面的阶段耗时。
                rknn_perf_run run_performance{};
                check(rknn_query(context.value, RKNN_QUERY_PERF_RUN, &run_performance, sizeof(run_performance)),
                      "RKNN_QUERY_PERF_RUN");
                if (run_performance.run_duration <= 0)
                    throw std::runtime_error("RKNN_QUERY_PERF_RUN returned a non-positive run duration");
                collected.run_duration_us = run_performance.run_duration;
                if (collect_performance) {
                    rknn_perf_detail detail{};
                    check(rknn_query(context.value, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)),
                          "RKNN_QUERY_PERF_DETAIL");
                    constexpr uint64_t max_detail_bytes = 64ULL * 1024 * 1024;
                    if (!detail.perf_data || !detail.data_len || detail.data_len > max_detail_bytes)
                        throw std::runtime_error("RKNN_QUERY_PERF_DETAIL returned empty/invalid data or exceeded 64 MiB");
                    // perf_data 属于 Runtime，不 free；锁内立即复制，避免跨下一次推理保留借用指针。
                    collected.detail.assign(detail.perf_data, static_cast<size_t>(detail.data_len));
                }
            }
            lease.finish();
            if (performance) *performance = std::move(collected);

            // 3. 只结束 engine 的本帧持有；注册继续缓存，调用方仍可保留自己的帧租约。
            active_input = {}; // 本次设备使用结束；调用方仍持有 input 时，预处理器仍不能覆盖该帧。
            return result;
        } catch (...) {
            failed = true;
            // 任一绑定/执行阶段失败后禁止继续推理；保持帧租约，交给析构中的等待与清理路径。
            throw;
        }
    }
    AppConfig config;
    const bool collect_performance;
    Context context;
    InputMemoryLayout layout;
    rknn_tensor_attr input_attr{};
    std::unique_ptr<ImportedInput> imported;   // 长期注册，不持有当前帧租约。
    std::shared_ptr<void> transition_allocation; // 换注册失败时继续保留旧分配。
    TensorInput active_input;                 // 当前帧占用，异常后也保留到安全清理。
    bool device_pending = false;              // 失败清理是否仍需 rknn_wait 确认完成。
    bool failed = false;                      // 绑定/执行失败后拒绝再次使用 engine。
    std::mutex mutex;
    std::vector<rknn_tensor_attr> attrs;
    std::vector<std::vector<float>> buffers;
    std::vector<rknn_output> outputs;
    size_t boxes_index = 0, scores_index = 1;
};
YoloEngine::YoloEngine(const AppConfig& c, int core_mask, bool collect_perf) : impl_(std::make_unique<Impl>(c, core_mask, collect_perf)) {}
YoloEngine::~YoloEngine() = default;
InferResult YoloEngine::infer(const TensorInput& in, int64_t index, NpuPerformance* performance) { return impl_->infer(in, index, performance); }
InputMemoryLayout YoloEngine::inputMemoryLayout() const { return impl_->layout; }
}  // namespace yolo

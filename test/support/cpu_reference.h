#pragma once
// CPU reference implementations belong only to validation, never to yolo_core.
#include "image/image_processing.h"
#include "model/yolo_postprocess.h"
#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>

namespace yolo::test {
inline cv::Mat referenceRgb(const cv::Mat& bgr, const AppConfig& config) {
    const auto g = resizeGeometry(bgr.cols, bgr.rows, config);
    cv::Mat rgb(config.input_height, config.input_width, CV_8UC3, cv::Scalar::all(114)), resized;
    cv::resize(bgr, resized, {g.width, g.height});
    cv::cvtColor(resized, rgb(cv::Rect(g.left, g.top, g.width, g.height)), cv::COLOR_BGR2RGB);
    return rgb;
}

class CpuReferenceEngine {
public:
    explicit CpuReferenceEngine(const AppConfig& config) : config_(config) {
        std::ifstream file(config.model_path, std::ios::binary | std::ios::ate);
        const auto bytes = file.tellg();
        if (bytes <= 0) throw std::runtime_error("reference model read failed");
        std::vector<char> model(static_cast<size_t>(bytes));
        file.seekg(0);
        if (!file.read(model.data(), bytes)) throw std::runtime_error("reference model read failed");
        check(rknn_init(&context_, model.data(), model.size(), 0, nullptr));
        try {
            for (unsigned i = 0; i < 2; ++i) {
                attrs_[i].index = i;
                check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attrs_[i], sizeof(attrs_[i])));
                pixels_[i].resize(attrs_[i].n_elems);
                outputs_[i].index = i;
                outputs_[i].want_float = 1;
                outputs_[i].is_prealloc = 1;
                outputs_[i].buf = pixels_[i].data();
                outputs_[i].size = pixels_[i].size() * sizeof(float);
                if (attrs_[i].dims[1] == 4) boxes_ = i;
            }
        } catch (...) { rknn_destroy(context_); throw; }
    }
    ~CpuReferenceEngine() { rknn_destroy(context_); }
    CpuReferenceEngine(const CpuReferenceEngine&) = delete;
    CpuReferenceEngine& operator=(const CpuReferenceEngine&) = delete;

    InferResult infer(const cv::Mat& rgb, const TensorInput& geometry, int64_t index = -1) {
        const auto start = std::chrono::steady_clock::now();
        rknn_input input{};
        input.buf = rgb.data; input.size = rgb.total() * rgb.elemSize();
        input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
        check(rknn_inputs_set(context_, 1, &input));
        check(rknn_run(context_, nullptr));
        check(rknn_outputs_get(context_, 2, outputs_.data(), nullptr));
        struct Release {
            rknn_context context; rknn_output* outputs;
            ~Release() { rknn_outputs_release(context, 2, outputs); }
        } release{context_, outputs_.data()};
        InferResult result;
        result.frame_index = index;
        result.infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        result.detections = decodeYolo26(pixels_[boxes_].data(), pixels_[1-boxes_].data(),
                                       attrs_[boxes_].dims[2], geometry, config_);
        return result;
    }
private:
    static void check(int rc) {
        if (rc != RKNN_SUCC) throw std::runtime_error("CPU reference RKNN error: " + std::to_string(rc));
    }
    AppConfig config_;
    rknn_context context_ = 0;
    std::array<rknn_tensor_attr, 2> attrs_{};
    std::array<rknn_output, 2> outputs_{};
    std::array<std::vector<float>, 2> pixels_;
    int boxes_ = 0;
};
} // namespace yolo::test
